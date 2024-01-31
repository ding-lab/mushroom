import logging
import os
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from torchio.transforms import Resize
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

import mushroom.visualization.utils as vis_utils
from mushroom.model.sae import SAEargs
from mushroom.model.model import LitSpore, WandbImageCallback, VariableTrainingCallback
from mushroom.data.datasets import get_learner_data, construct_training_batch, construct_inference_batch
from mushroom.model.integration import integrate_volumes
import mushroom.utils as utils


class Mushroom(object):
    def __init__(
            self,
            sections,
            dtype_to_chkpt=None,
            sae_kwargs=None,
            trainer_kwargs=None,
        ):
        self.sections = sections
        self.dtype_to_chkpt = dtype_to_chkpt
        self.sae_kwargs = sae_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.input_ppm = self.trainer_kwargs['input_ppm']
        self.target_ppm = self.trainer_kwargs['target_ppm']

        self.section_ids = [(entry['id'], d['dtype']) for entry in sections
                       for d in entry['data']]
        self.dtypes = sorted({x for _, x in self.section_ids})

        self.dtype_to_spore = {}
        for dtype in self.dtypes:
            logging.info(f'loading spore for {dtype}')
            dtype_sections = [entry for entry in sections if dtype in [item['dtype'] for item in entry['data']]]
            for i, entry in enumerate(dtype_sections):
                entry['data'] = [item for item in entry['data'] if item['dtype'] == dtype]
                dtype_sections[i] = entry
            
            # eventually add custom paramters for data types
            out_dir = self.trainer_kwargs['out_dir']
            trainer_kwargs['save_dir'] = os.path.join(out_dir, f'{dtype}_chkpts')
            trainer_kwargs['log_dir'] = os.path.join(out_dir, f'{dtype}_logs')

            chkpt_filepath = self.dtype_to_chkpt[dtype] if self.dtype_to_chkpt is not None else None

            spore = Spore(dtype_sections, chkpt_filepath=chkpt_filepath, sae_kwargs=sae_kwargs, trainer_kwargs=trainer_kwargs)

            self.dtype_to_spore[dtype] = spore

        self.integrated_clusters = None
        self.dtype_to_volume, self.dtype_to_volume_probs = None, None
        self.section_positions = None

    @staticmethod
    def from_config(mushroom_config, accelerator=None):
        if isinstance(mushroom_config, str):
            mushroom_config = yaml.safe_load(open(mushroom_config))

        if accelerator is not None:
            mushroom_config['trainer_kwargs']['accelerator'] = accelerator

        mushroom = Mushroom(
            mushroom_config['sections'],
            dtype_to_chkpt=mushroom_config['dtype_to_chkpt'],
            sae_kwargs=mushroom_config['sae_kwargs'],
            trainer_kwargs=mushroom_config['trainer_kwargs'],
        )

        if mushroom_config['dtype_to_chkpt'] is not None:
            logging.info(f'chkpt files detected, embedding to spores')
            mushroom.embed_sections()

        return mushroom

    def train(self, dtypes=None):
        dtypes = dtypes if dtypes is not None else self.dtypes
        if self.dtype_to_chkpt is None:
            self.dtype_to_chkpt = {}

        for dtype in dtypes:
            logging.info(f'starting training for {dtype}')
            spore = self.dtype_to_spore[dtype]
            spore.train()

            # save chkpts
            chkpt_dir = os.path.join(self.trainer_kwargs['out_dir'], f'{dtype}_chkpts')
            fps = [fp for fp in os.listdir(chkpt_dir) if 'last' in fp]
            if len(fps) == 1:
                chkpt_fp = os.path.join(chkpt_dir, 'last.ckpt')
            else:
                val = np.max([int(re.sub(r'^last-v([0-9]+).ckpt$', r'\1', fp)) for fp in fps])
                chkpt_fp = os.path.join(chkpt_dir, f'last-v{val}.ckpt')

            logging.info(f'finished training {dtype}, saved chkpt to {chkpt_fp}')
            self.dtype_to_chkpt[dtype] = chkpt_fp

    def embed_sections(self, dtypes=None):
        dtypes = dtypes if dtypes is not None else self.dtypes

        for dtype in dtypes:
            logging.info(f'embedding {dtype} spore')
            spore = self.dtype_to_spore[dtype]
            spore.embed_sections()

    def save(self, output_dir=None):
        if output_dir is None:
            output_dir = self.trainer_kwargs['out_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logging.info(f'saving config and outputs to {output_dir}')

        config = {
            'sections': self.sections,
            'dtype_to_chkpt': self.dtype_to_chkpt,
            'sae_kwargs': self.sae_kwargs,
            'trainer_kwargs': self.trainer_kwargs
        }

        # clusters and cluster probs
        dtype_to_clusters = {
            'integrated': self.integrated_clusters,
        }
        dtype_to_cluster_probs, dtype_to_cluster_probs_all = {}, {}
        for dtype, spore in self.dtype_to_spore.items():
            dtype_to_clusters[dtype] = spore.clusters
            dtype_to_cluster_probs = spore.cluster_probs
            dtype_to_cluster_probs_all = spore.cluster_probs_all

        # cluster intensities
        dtype_to_cluster_intensities = {
            'dtype_specific': [
                self.calculate_cluster_intensities(level=level)
                for level in range(len(next(iter(self.dtype_to_spore.values())).clusters))
            ]
        }
        if self.integrated_clusters is not None:
            dtype_to_cluster_intensities['integrated'] = [
                    self.calculate_cluster_intensities(use_integrated=True, level=level) if self.integrated_clusters[level] is not None else None
                    for level in range(len(self.integrated_clusters))
            ]

        outputs = {
            'section_positions': self.section_positions,
            'section_ids': self.section_ids,
            'dtype_to_volume': self.dtype_to_volume,
            'dtype_to_volume_probs': self.dtype_to_volume_probs,
            'dtype_to_clusters': dtype_to_clusters,
            'dtype_to_cluster_probs': dtype_to_cluster_probs,
            'dtype_to_cluster_probs_all': dtype_to_cluster_probs_all,
            'dtype_to_cluster_intensities': dtype_to_cluster_intensities,
        }

        yaml.safe_dump(
            config,
            open(os.path.join(output_dir, f'config.yaml'), 'w')
        )
        np.save(
            os.path.join(output_dir, f'outputs.npy'),
            outputs
        )

    def calculate_cluster_intensities(self, use_integrated=False, use_predicted=True, level=-1):
        dtype_to_df = {}

        for dtype, spore in self.dtype_to_spore.items():
            if use_integrated:
                assert self.integrated_clusters is not None, 'Must generate integrated volume first'
                input_clusters = [c for c, (sid, dtype) in zip(self.integrated_clusters, self.section_ids) if dtype==dtype]
                dtype_to_df[dtype] = spore.get_cluster_intensities(use_predicted=use_predicted, level=level, input_clusters=input_clusters)[dtype]
            else:
                dtype_to_df[dtype] = spore.get_cluster_intensities(use_predicted=use_predicted, level=level)[dtype]

        return dtype_to_df

    def generate_interpolated_volumes(self, z_scaler=.1, level=-1, use_probs=True, integrate=True, dist_thresh=.5, n_iterations=10, resolution=2., dtype_to_weight=None, kernel=None, kernel_size=5):
        dtypes, spores = zip(*self.dtype_to_spore.items())
        if self.integrated_clusters is None:
            self.integrated_clusters = [None for i in range(len(next(iter(self.dtype_to_spore.values())).clusters))]

        section_positions = []
        sids = []
        for spore in spores:
            section_positions += [entry['position'] for entry in spore.sections]
            sids += spore.section_ids
        section_positions, sids = zip(*sorted([(p, tup) for p, tup in zip(section_positions, sids)], key=lambda x: x[0]))

        section_positions = (np.asarray(section_positions) * z_scaler).astype(int)
        for i, (val, (ident, dtype)) in enumerate(zip(section_positions, sids)):
            if i > 0:
                old = section_positions[i-1]
                old_ident = sids[i-1][0]
                if old == val and old_ident != ident:
                    section_positions[i:] = section_positions[i:] + 1

        start, stop = section_positions[0], section_positions[-1]
        dtype_to_volume = {}
        for dtype, spore in zip(dtypes, spores):
            logging.info(f'generating volume for {dtype} spore')
            positions = [p for p, (_, dt) in zip(section_positions, sids) if dt==dtype]

            if use_probs:
                clusters = spore.cluster_probs[level].copy()
            else:
                clusters = spore.clusters[level].copy()

            if positions[0] != start:
                positions.insert(0, start)
                clusters = np.concatenate((clusters[:1], clusters))
            if positions[-1] != stop:
                positions.append(stop)
                clusters = np.concatenate((clusters, clusters[-1:]))

            if use_probs:
                clusters = rearrange(clusters, 'n h w c -> c n h w')
                volume = utils.get_interpolated_volume(clusters, positions, method='linear')
                volume = rearrange(volume, 'c n h w -> n h w c')
            else:
                volume = utils.get_interpolated_volume(clusters, positions)
            dtype_to_volume[dtype] = volume

        if integrate:
            logging.info(f'generating integrated volume')
            dtype_to_cluster_intensities = self.calculate_cluster_intensities(level=level)
            integrated = integrate_volumes(dtype_to_volume, dtype_to_cluster_intensities, are_probs=use_probs, dist_thresh=dist_thresh, n_iterations=n_iterations, resolution=resolution, dtype_to_weight=dtype_to_weight, kernel=kernel, kernel_size=kernel_size)
            logging.info(f'finished integration, found {integrated.max()} clusters')
            dtype_to_volume['integrated'] = integrated
            self.integrated_clusters[level] = np.stack([integrated[i] for i in section_positions])

        if use_probs:
            self.dtype_to_volume_probs = dtype_to_volume
            self.dtype_to_volume = {dtype:probs.argmax(-1) if dtype!='integrated' else probs
                                    for dtype, probs in self.dtype_to_volume_probs.items()}
        else:
            self.dtype_to_volume = dtype_to_volume
        self.section_positions = section_positions
        return dtype_to_volume

    def display_predicted_pixels(self, dtype, channel, level=-1, figsize=None):
        spore = self.dtype_to_spore[dtype]
        return spore.display_predicted_pixels(channel, dtype, level=level, figsize=figsize)
    
    def display_cluster_probs(self, dtype, level=-1, return_axs=False):
        if dtype == 'integrated':
            raise RuntimeError(f'Probabilities are not caclulated for integrated clusters')
        else:
            spore = self.dtype_to_spore[dtype]
            return spore.display_cluster_probs(level=level, return_axs=return_axs)

    def display_clusters(self, dtype, level=-1, section_idxs=None, section_ids=None, cmap=None, figsize=None, horizontal=True, preserve_indices=True, return_axs=False):
        if dtype == 'integrated':
            clusters = self.integrated_clusters[level]
        else:
            clusters = self.dtype_to_spore[dtype].clusters[level]

        if section_ids is None and section_idxs is None:
            return vis_utils.display_clusters(
                clusters, cmap=cmap, figsize=figsize, horizontal=horizontal, preserve_indices=preserve_indices, return_axs=return_axs)
        else:
            if section_idxs is None:
                section_idxs = [i for i, sid in enumerate(self.section_ids) if sid in section_ids]
            return vis_utils.display_clusters(
                clusters[section_idxs], cmap=cmap, figsize=figsize, horizontal=horizontal, preserve_indices=preserve_indices, return_axs=return_axs)
        
    def display_volumes(self, positions=None, dtype_to_volume=None, figsize=None, return_axs=False):
        if dtype_to_volume is None:
            assert self.dtype_to_volume is not None, f'need to run generate_interpolated_volumes first'
            dtype_to_volume = self.dtype_to_volume

        dtypes, volumes = zip(*dtype_to_volume.items())

        if positions is not None:
            volumes = [v[positions] for v in volumes]

        ncols = len(volumes)
        if figsize is None:
            figsize = (ncols, volumes[0].shape[0])
        fig, axs = plt.subplots(nrows=volumes[0].shape[0], ncols=ncols, figsize=figsize)
        for i in range(volumes[0].shape[0]):
            for j, volume in enumerate(volumes):
                ax = axs[i, j]
                rgb = vis_utils.display_labeled_as_rgb(volume[i], preserve_indices=True)
                ax.imshow(rgb)
                ax.axis('off')

                if i==0:
                    ax.set_title(dtypes[j])

        if return_axs:
            return axs
        
    def assign_pts(self, pts, section_id, dtype, level=-1, scale=True, use_volume=False):
        """
        pts are (x, y)
        """
        dtype = section_id[1] if dtype is None else dtype

        if scale:
            scaler = self.input_ppm / self.target_ppm
            pts = pts / scaler
            pts = pts.astype(int)

        if dtype == 'integrated':
            section_idx = self.section_ids.index(section_id)            
            nbhds = self.integrated_clusters[level][section_idx]
        else:
            if use_volume:
                section_idx = self.section_ids.index(section_id)   
                position = self.section_positions[section_idx]
                nbhds = self.dtype_to_volume[dtype][position]

            else:
                spore = self.dtype_to_spore[dtype]
                section_idx = spore.section_ids.index(section_id)            
                nbhds = spore.clusters[level][section_idx]
            
        max_h, max_w = nbhds.shape[0] - 1, nbhds.shape[1] - 1

        pts[pts[:, 0] > max_w, 0] = max_w
        pts[pts[:, 1] > max_h, 1] = max_h

        labels = nbhds[pts[:, 1], pts[:, 0]]

        return labels


class Spore(object):
    def __init__(
            self,
            sections,
            chkpt_filepath=None,
            sae_kwargs=None,
            trainer_kwargs=None,
        ):
        self.sections = sections
        self.chkpt_filepath = chkpt_filepath
        self.sae_kwargs = sae_kwargs
        self.trainer_kwargs = trainer_kwargs

        self.channel_mapping = self.trainer_kwargs['channel_mapping']
        self.input_ppm = self.trainer_kwargs['input_ppm']
        self.target_ppm = self.trainer_kwargs['target_ppm']
        self.contrast_pct = self.trainer_kwargs['contrast_pct']
        self.pct_expression = self.trainer_kwargs['pct_expression']

        self.sae_args = SAEargs(**self.sae_kwargs) if self.sae_kwargs is not None else {}
        self.size = (self.sae_args.size, self.sae_args.size)

        self.learner_data = get_learner_data(self.sections, self.input_ppm, self.target_ppm, self.sae_args.size,
                                             channel_mapping=self.channel_mapping, contrast_pct=self.contrast_pct, pct_expression=self.pct_expression)
        self.section_ids = self.learner_data.train_ds.section_ids
        self.dtypes = self.learner_data.dtypes
        self.dtype_to_channels = self.learner_data.dtype_to_channels
        self.batch_size = self.trainer_kwargs['batch_size']
        self.num_workers = self.trainer_kwargs['num_workers']

        # by default dataset is infinite, change to desired num steps
        self.learner_data.train_ds.n = self.trainer_kwargs['steps_per_epoch'] * self.batch_size

        logging.info('creating data loaders')
        self.train_dl = DataLoader(
            self.learner_data.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=construct_training_batch
        )
        self.inference_dl = DataLoader(
            self.learner_data.inference_ds, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=construct_inference_batch
        )

        self.model = LitSpore(
            self.sae_args,
            self.learner_data,
            lr=self.trainer_kwargs['lr'],
            total_steps=self.trainer_kwargs['max_epochs'] * self.trainer_kwargs['steps_per_epoch'],
        )

        logging.info('model initialized')

        Path(self.trainer_kwargs['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.trainer_kwargs['save_dir']).mkdir(parents=True, exist_ok=True)

        callbacks = []
        if self.trainer_kwargs['logger_type'] == 'wandb':
            logger = pl_loggers.WandbLogger(
                project=self.trainer_kwargs['logger_project'],
                save_dir=self.trainer_kwargs['log_dir'],
            )
            logger.experiment.config.update({
                'trainer_kwargs': self.trainer_kwargs,
                'sae_kwargs': self.sae_kwargs,
                'sections': self.sections
            })

            logging_callback = WandbImageCallback(
                logger, self.learner_data, self.inference_dl)
            callbacks.append(logging_callback)

        else:
            logger = pl_loggers.TensorBoardLogger(
                save_dir=self.trainer_kwargs['log_dir'],
            )
        chkpt_callback = ModelCheckpoint(
            dirpath=self.trainer_kwargs['save_dir'],
            save_last=True,
            # save_top_k=-1,
            every_n_epochs=self.trainer_kwargs['save_every'],
        )
        callbacks.append(chkpt_callback)

        vt_callback = VariableTrainingCallback()
        callbacks.append(vt_callback)

        self.trainer = self.initialize_trainer(logger, callbacks)

        if self.chkpt_filepath is not None:
            logging.info(f'loading checkpoint: {self.chkpt_filepath}')
            state_dict = torch.load(self.chkpt_filepath)['state_dict']
            self.model.load_state_dict(state_dict)
            
        self.predicted_pixels, self.scaled_predicted_pixels = None, None
        self.true_pixels, self.scaled_true_pixels = None, None
        self.clusters, self.agg_clusters, self.cluster_probs_agg = None, None, None
        self.cluster_probs, self.cluster_probs_all = None, None
        self.cluster_to_agg = None


    @staticmethod
    def from_config(config, chkpt_filepath=None, accelerator=None):
        if isinstance(config, str):
            config = yaml.safe_load(open(config))

        if accelerator is not None:
            config['trainer_kwargs']['accelerator'] = 'cpu'

        spore = Spore(
            config['sections'],
            sae_kwargs=config['sae_kwargs'],
            trainer_kwargs=config['trainer_kwargs'],
            chkpt_filepath=chkpt_filepath
        )

        return spore
    
    def _calculate_probs(self):
        n_levels = len(self.clusters)
        # probs for all clusters
        level_to_probs_all = []
        for level in range(n_levels):
            chars = utils.CHARS[:level + 1]
            ein_exp = ','.join([f'nhw{x}' for x in chars])
            ein_exp += f'->nhw{chars}'
            probs = np.einsum(ein_exp, *self.cluster_probs_agg[:level + 1])
            
            if level:
                new_probs = np.zeros_like(probs)
                for label, cluster in self.cluster_to_agg[level].items():
                    mask = self.clusters[level]==label
                    values = probs[mask] # (n, z)
                    empty = np.zeros_like(values)

                    selections = tuple([slice(None)] + list(cluster)[:-1])
                    empty[selections] = values[selections]
                    new_probs[mask] = empty
            else:
                new_probs = probs

            level_to_probs_all.append(new_probs)

        # probs for labeled clusters only
        level_to_probs = []
        for level in range(n_levels):
            probs = level_to_probs_all[level]
            
            if level:
                labeled_probs = np.zeros((probs.shape[0], probs.shape[1], probs.shape[2], len(self.cluster_to_agg[level])))
                for label, cluster in self.cluster_to_agg[level].items():
                    mask = self.clusters[level]==label
                    values = probs[mask] # (n, a, b, c)

                    window = labeled_probs[mask]
                    labels, cs = zip(*[(l, c) for l, c in self.cluster_to_agg[level].items()
                                       if list(c)[:-1] == list(cluster)[:-1]])
                    cs = np.asarray(cs)
                    tups = [tuple(cs[:, i]) for i in range(cs.shape[1])]

                    selections = tuple([slice(None)] + tups)
                    window[:, labels] = values[selections]
                    labeled_probs[mask] = window
            else:
                labels = np.unique(probs.argmax(axis=-1))
                labeled_probs = probs[..., labels]
            
            level_to_probs.append(labeled_probs)

        return level_to_probs, level_to_probs_all

    
    def initialize_trainer(
            self,
            logger,
            callbacks
        ):

        return Trainer(
            devices=self.trainer_kwargs['devices'],
            accelerator=self.trainer_kwargs['accelerator'],
            enable_checkpointing=self.trainer_kwargs['enable_checkpointing'],
            log_every_n_steps=self.trainer_kwargs['log_every_n_steps'],
            max_epochs=self.trainer_kwargs['max_epochs'],
            callbacks=callbacks,
            logger=logger
        )

    
    def train(self):
        self.trainer.fit(self.model, self.train_dl)

    def embed_sections(self):
        outputs = self.trainer.predict(self.model, self.inference_dl)
        formatted = self.model.format_prediction_outputs(outputs)
        self.predicted_pixels = [[z.cpu().clone().detach().numpy() for z in x] for x in formatted['predicted_pixels']]
        self.true_pixels = [x.cpu().clone().detach().numpy() for x in formatted['true_pixels']]
        self.clusters = [x for x in formatted['clusters']]
        self.agg_clusters = [x.cpu().clone().detach().numpy().astype(int) for x in formatted['agg_clusters']]
        self.cluster_probs_agg = [x.cpu().clone().detach().numpy() for x in formatted['cluster_probs']]
        self.cluster_to_agg = [x for x in formatted['label_to_original']]

        self.cluster_probs, self.cluster_probs_all = self._calculate_probs()

    def get_cluster_intensities(self, use_predicted=True, level=-1, input_clusters=None):
        if input_clusters is None:
            input_clusters = self.clusters
        
        dtype_to_df = {}
        imgs = self.predicted_pixels[level] if use_predicted else self.true_pixels
        for dtype in self.dtypes:
            sections, clusters = [], []
            for (sid, dt), img, labeled in zip(self.section_ids, imgs, input_clusters[level]):
                if dt == dtype:
                    sections.append(img)
                    clusters.append(labeled)
            sections = np.stack(sections) # (n, h, w, c)
            clusters = np.stack(clusters) # (n, h, w)

            data = []
            for cluster in np.unique(clusters):
                mask = clusters==cluster
                data.append(sections[mask, :].mean(axis=0))
            df = pd.DataFrame(data=data, columns=self.learner_data.dtype_to_channels[dtype], index=np.unique(clusters))
            dtype_to_df[dtype] = df
        return dtype_to_df


    def generate_interpolated_volume(self, z_scaler=.1, level=-1, use_probs=False):
        section_positions = [entry['position'] for entry in self.sections]
        section_positions = (np.asarray(section_positions) * z_scaler).astype(int)
        for i, val in enumerate(section_positions):
            if i > 0:
                old = section_positions[i-1]
                if old == val:
                    section_positions[i:] = section_positions[i:] + 1

        if use_probs:
            probs = rearrange(self.cluster_probs[level], 'n h w c -> c n h w')
            volume = utils.get_interpolated_volume(probs, section_positions, method='linear')
            volume = rearrange(volume, 'c n h w -> n h w c')
        else:
            volume = utils.get_interpolated_volume(self.clusters[level], section_positions, method='label_gaussian')
        return volume

    def display_predicted_pixels(self, channel, dtype, level=-1, figsize=None, return_axs=False):
        if self.predicted_pixels is None:
            raise RuntimeError(
                'Must train model and embed sections before displaying. To embed run .embed_sections()')
        pred, true, sids = [], [], []
        for (sid, dt), pred_imgs, true_imgs in zip(self.section_ids, self.predicted_pixels[level], self.true_pixels):
            if dt == dtype:
                pred.append(pred_imgs)
                true.append(true_imgs)
                sids.append(sid)
        pred = np.stack(pred) # (n, h, w, c)
        true = np.stack(true) # (n, h, w, c)

        fig, axs = plt.subplots(nrows=2, ncols=pred.shape[0], figsize=figsize)

        for sid, img, ax in zip(sids, pred, axs[0, :]):
            ax.imshow(img[..., self.learner_data.dtype_to_channels[dtype].index(channel)])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(sid)

        for sid, img, ax in zip(sids, true, axs[1, :]):
            ax.imshow(img[..., self.learner_data.dtype_to_channels[dtype].index(channel)])
            ax.set_xticks([])
            ax.set_yticks([])
        axs[0, 0].set_ylabel('predicted')
        axs[1, 0].set_ylabel('true')

        if return_axs:
            return axs
    
    def display_cluster_probs(self, level=-1, prob_type='clusters', return_axs=False):

        if prob_type == 'clusters_agg':
            cluster_probs = self.cluster_probs_agg[level]
        else:
            cluster_probs = self.cluster_probs[level]
        fig, axs = plt.subplots(
            nrows=cluster_probs.shape[-1],
            ncols=cluster_probs.shape[0],
            figsize=(cluster_probs.shape[0], cluster_probs.shape[-1])
        )
        for c in range(cluster_probs.shape[0]):
            for r in range(cluster_probs.shape[-1]):
                ax = axs[r, c]
                ax.imshow(cluster_probs[c, ..., r])
                ax.set_yticks([])
                ax.set_xticks([])
                if c == 0: ax.set_ylabel(r, rotation=90)
        
        if return_axs:
            return axs

    def display_clusters(self, level=-1, section_idxs=None, section_ids=None, cmap=None, figsize=None, horizontal=True, preserve_indices=True, return_axs=False):
        if section_ids is None and section_idxs is None:
            return vis_utils.display_clusters(
                self.clusters[level], cmap=cmap, figsize=figsize, horizontal=horizontal, preserve_indices=preserve_indices, return_axs=return_axs)
        else:
            if section_idxs is None:
                section_idxs = [i for i, sid in enumerate(self.section_ids) if sid in section_ids]
            return vis_utils.display_clusters(
                self.clusters[level][section_idxs], cmap=cmap, figsize=figsize, horizontal=horizontal, preserve_indices=preserve_indices, return_axs=return_axs)
        
    def assign_pts(self, pts, section_id=None, section_idx=None, level=-1, scale=True):
        """
        pts are (x, y)
        """
        assert section_id is not None or section_idx is not None, f'either section id or section index must be given'
        if scale:
            scaler = self.input_ppm / self.target_ppm
            pts = pts / scaler
            pts = pts.astype(int)

        section_idx = self.section_ids.index(section_id) if section_id is not None else section_idx
        
        nbhds = self.clusters[level][section_idx]
        max_h, max_w = nbhds.shape[0] - 1, nbhds.shape[1] - 1

        pts[pts[:, 0] > max_w, 0] = max_w
        pts[pts[:, 1] > max_h, 1] = max_h

        labels = nbhds[pts[:, 1], pts[:, 0]]

        return labels