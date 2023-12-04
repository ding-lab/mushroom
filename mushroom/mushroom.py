import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from einops import rearrange
from torch.utils.data import DataLoader
from torchio.transforms import Resize
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

import mushroom.data.multiplex as multiplex
import mushroom.data.xenium as xenium
import mushroom.data.visium as visium
import mushroom.visualization.utils as vis_utils
from mushroom.model.sae import SAEargs
from mushroom.model.model import LitMushroom, WandbImageCallback
import mushroom.utils as utils


class Mushroom(object):
    def __init__(
            self,
            dtype,
            sections,
            chkpt_filepath=None,
            sae_kwargs=None,
            trainer_kwargs=None,
        ):
        self.dtype = dtype
        self.sections = sections
        self.chkpt_filepath = chkpt_filepath
        self.sae_kwargs = sae_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.section_ids = [entry['id'] for entry in self.sections
                       if dtype in [d['dtype'] for d in entry['data']]]

        self.dtype = dtype
        self.channel_mapping = self.trainer_kwargs['channel_mapping']
        self.scale = self.trainer_kwargs['scale']
        self.channels = self.trainer_kwargs['channels']
        self.contrast_pct = self.trainer_kwargs['contrast_pct']
        self.pct_expression = self.trainer_kwargs['pct_expression']

        self.sae_args = SAEargs(**self.sae_kwargs) if self.sae_kwargs is not None else {}
        self.size = (self.sae_args.size, self.sae_args.size)

        

        logging.info(f'generating inputs for {self.dtype} tissue sections')
        if self.dtype == 'multiplex':
            self.learner_data = multiplex.get_learner_data(
                self.sections, self.scale, self.size, self.sae_args.patch_size,
                channels=self.channels, channel_mapping=self.channel_mapping, contrast_pct=self.contrast_pct,
            )
        elif self.dtype == 'xenium':
            self.learner_data = xenium.get_learner_data(
                self.sections, self.scale, self.size, self.sae_args.patch_size,
                channels=self.channels, channel_mapping=self.channel_mapping,
            )
        elif self.dtype == 'he':
            pass
        elif self.dtype == 'visium':
            self.learner_data = visium.get_learner_data(
                self.sections, self.scale, self.size, self.sae_args.patch_size,
                channels=self.channels, channel_mapping=self.channel_mapping, pct_expression=self.pct_expression,
            )
        else:
            raise RuntimeError(f'dtype must be one of the following: \
["multiplex", "he", "visium", "xenium"], got {self.dtype}')
        self.channels = self.learner_data.channels
        self.batch_size = self.trainer_kwargs['batch_size']
        self.num_workers = self.trainer_kwargs['num_workers']

        # by default dataset is infinite, change to desired num steps
        self.learner_data.train_ds.n = self.trainer_kwargs['steps_per_epoch'] * self.batch_size

        logging.info('creating data loaders')
        self.train_dl = DataLoader(
            self.learner_data.train_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
        self.inference_dl = DataLoader(
            self.learner_data.inference_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

        self.model = LitMushroom(
            self.sae_args,
            self.learner_data,
            lr=self.trainer_kwargs['lr'],
        )

        logging.info('model initialized')

        # make a groundtruth reconstruction for original images
        self.true_imgs = torch.stack(
            [self.learner_data.inference_ds.image_from_tiles(self.learner_data.inference_ds.section_to_tiles[s])
            for s in self.learner_data.inference_ds.sections]
        ).cpu().detach().numpy()
        if self.dtype in ['visium']:
            self.true_imgs = torch.stack(
                [visium.format_expression(
                    img, self.learner_data.inference_ds.section_to_adata[sid], self.learner_data.sae_args.patch_size
                ) for sid, img in zip(self.section_ids, self.true_imgs)]
            ).cpu().detach().numpy()

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
                logger, self.learner_data, self.inference_dl, self.true_imgs,
                channel=self.trainer_kwargs['logger_channel']
            )
            callbacks.append(logging_callback)

        else:
            logger = pl_loggers.TensorBoardLogger(
                save_dir=self.trainer_kwargs['log_dir'],
            )
        chkpt_callback = ModelCheckpoint(
            dirpath=self.trainer_kwargs['save_dir'],
            save_last=True,
            every_n_epochs=self.trainer_kwargs['save_every'],
        )
        callbacks.append(chkpt_callback)

        self.trainer = self.initialize_trainer(logger, callbacks)
            
        self.predicted_pixels, self.scaled_predicted_pixels = None, None
        self.clusters, self.cluster_probs = None, None


    @staticmethod
    def from_config(mushroom_config, chkpt_filepath=None, accelerator=None):
        if isinstance(mushroom_config, str):
            mushroom_config = yaml.safe_load(open(mushroom_config))

        if accelerator is not None:
            mushroom_config['trainer_kwargs']['accelerator'] = 'cpu'

        mushroom = Mushroom(
            mushroom_config['dtype'],
            mushroom_config['sections'],
            sae_kwargs=mushroom_config['sae_kwargs'],
            trainer_kwargs=mushroom_config['trainer_kwargs'],
        )

        if chkpt_filepath is not None:
            logging.info(f'loading checkpoint: {chkpt_filepath}')
            state_dict = torch.load(chkpt_filepath)['state_dict']
            mushroom.model.load_state_dict(state_dict)

        return mushroom

    
    def _get_section_imgs(self, args):
        emb_size = int(self.true_imgs.shape[-2] / self.learner_data.train_transform.output_patch_size)
        section_imgs = TF.resize(self.true_imgs, (emb_size, emb_size), antialias=True)

        if args.background_channels is None and args.mask_background:
            logging.info('no background channel detected, defaulting to mean of all channels')
            section_imgs = section_imgs.mean(1)
        elif args.background_channels is not None:
            idxs = [self.learner_data.channels.index(channel) for channel in args.background_channels]
            section_imgs = section_imgs[:, idxs].mean(1)
        else:
            section_imgs = None

        return section_imgs
    
    
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
        self.predicted_pixels = formatted['predicted_pixels'].cpu().clone().detach().numpy()
        self.clusters = formatted['clusters'].cpu().clone().detach().numpy().astype(int)
        self.cluster_probs = formatted['cluster_probs'].cpu().clone().detach().numpy()

        scalers = np.amax(self.predicted_pixels, axis=(-2, -1))
        self.scaled_predicted_pixels = self.predicted_pixels / rearrange(scalers, 'n c -> n c 1 1')

    def get_cluster_intensities(self):
        data = []
        x = rearrange(self.scaled_predicted_pixels, 'n c h w -> n h w c')
        for cluster in np.unique(self.clusters):
            mask = self.clusters==cluster
            data.append(x[mask, :].mean(axis=0))
        df = pd.DataFrame(data=data, columns=self.learner_data.channels, index=np.unique(self.clusters))
        return df

    def generate_interpolated_volume(self, z_scaler=.1):
        section_positions = [entry['position'] for entry in self.sections
                             if entry['data'][0]['dtype']=='multiplex']
        section_positions = (np.asarray(section_positions) * z_scaler).astype(int)
        cluster_volume = utils.get_interpolated_volume(self.clusters, section_positions)
        return cluster_volume

    def display_predicted_pixels(self, channel=None, figsize=None):
        if self.predicted_pixels is None:
            raise RuntimeError(
                'Must train model and embed sections before displaying. To embed run .embed_sections()')
        channel = channel if channel is not None else self.learner_data.channels[0]
        fig, axs = plt.subplots(nrows=2, ncols=self.predicted_pixels.shape[0], figsize=figsize)

        for sid, img, ax in zip(self.section_ids, self.predicted_pixels, axs[0, :]):
            ax.imshow(img[self.learner_data.channels.index(channel)])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(sid)

        for sid, img, ax in zip(self.section_ids, self.true_imgs, axs[1, :]):
            ax.imshow(img[self.learner_data.channels.index(channel)])
            ax.set_xticks([])
            ax.set_yticks([])
        axs[0, 0].set_ylabel('predicted')
        axs[1, 0].set_ylabel('true')

        return axs
    
    def display_cluster_probs(self):
        fig, axs = plt.subplots(
            nrows=self.cluster_probs.shape[1],
            ncols=self.cluster_probs.shape[0],
            figsize=(self.cluster_probs.shape[0], self.cluster_probs.shape[1])
        )
        for c in range(self.cluster_probs.shape[0]):
            for r in range(self.cluster_probs.shape[1]):
                ax = axs[r, c]
                ax.imshow(self.cluster_probs[c, r])
                ax.set_yticks([])
                ax.set_xticks([])
                if c == 0: ax.set_ylabel(r, rotation=90)

    def display_clusters(self, cmap=None, figsize=None, horizontal=True, preserve_indices=False):
        vis_utils.display_clusters(
            self.clusters, cmap=None, figsize=None, horizontal=True, preserve_indices=False)
