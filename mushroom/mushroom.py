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
from mushroom.data.datasets import get_learner_data, construct_training_batch, construct_inference_batch
import mushroom.utils as utils


class Mushroom(object):
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

        self.model = LitMushroom(
            self.sae_args,
            self.learner_data,
            lr=self.trainer_kwargs['lr'],
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
                logger, self.learner_data, self.inference_dl,
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
        self.true_pixels, self.scaled_true_pixels = None, None
        self.clusters, self.agg_clusters, self.cluster_probs = None, None, None


    @staticmethod
    def from_config(mushroom_config, chkpt_filepath=None, accelerator=None):
        if isinstance(mushroom_config, str):
            mushroom_config = yaml.safe_load(open(mushroom_config))

        if accelerator is not None:
            mushroom_config['trainer_kwargs']['accelerator'] = 'cpu'

        mushroom = Mushroom(
            mushroom_config['sections'],
            sae_kwargs=mushroom_config['sae_kwargs'],
            trainer_kwargs=mushroom_config['trainer_kwargs'],
        )

        if chkpt_filepath is not None:
            logging.info(f'loading checkpoint: {chkpt_filepath}')
            state_dict = torch.load(chkpt_filepath)['state_dict']
            mushroom.model.load_state_dict(state_dict)

        return mushroom

    
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
        self.cluster_probs = [x.cpu().clone().detach().numpy() for x in formatted['cluster_probs']]

        # scalers = [np.amax(x, axis=(-2, -1)) for x in self.predicted_pixels]
        # self.scaled_predicted_pixels = [xp / rearrange(xs, 'c -> c 1 1') for xp, xs in zip(self.predicted_pixels, scalers)]

        # scalers = np.amax(self.true_pixels, axis=(-2, -1))
        # self.scaled_true_pixels = self.true_pixels / rearrange(scalers, 'n c -> n c 1 1')

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
