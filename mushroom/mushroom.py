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
from lightning.pytorch.callbacks import ModelCheckpoint

import mushroom.data.multiplex as multiplex
import mushroom.data.xenium as xenium
import mushroom.data.visium as visium
from mushroom.model.sae import SAEargs
from mushroom.model.model import LitMushroom
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

        Path(self.trainer_kwargs['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.trainer_kwargs['save_dir']).mkdir(parents=True, exist_ok=True)
        logger = pl_loggers.TensorBoardLogger(
            save_dir=self.trainer_kwargs['log_dir'],
        )
        chkpt_callback = ModelCheckpoint(
            dirpath=self.trainer_kwargs['save_dir'],
            save_last=True,
            every_n_train_steps=self.trainer_kwargs['save_every'],
        )
        self.trainer = self.initialize_trainer(logger=logger, chkpt_callback=chkpt_callback)


        # make a groundtruth reconstruction for original images
        self.true_imgs = torch.stack(
            [self.learner_data.inference_ds.image_from_tiles(self.learner_data.inference_ds.section_to_tiles[s])
            for s in self.learner_data.inference_ds.sections]
        )
        if self.dtype in ['visium']:
            self.true_imgs = torch.stack(
                [visium.format_expression(
                    img, self.learner_data.inference_ds.section_to_adata[sid], self.learner_data.sae_args.patch_size
                ) for sid, img in zip(self.section_ids, self.true_imgs)]
            )
            
        self.recon_embs, self.recon_imgs, self.recon_cluster_ids, self.recon_cluster_probs = None, None, None, None

        self.cluster_probs, self.cluster_ids, self.scaled_recon_imgs = None, None, None

        

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
            chkpt_callback=None,
            logger=None
        ):
        
        if logger is None:
            logger = pl_loggers.TensorBoardLogger()
        if chkpt_callback is None:
            chkpt_callback = ModelCheckpoint()

        return Trainer(
            devices=self.trainer_kwargs['devices'],
            accelerator=self.trainer_kwargs['accelerator'],
            enable_checkpointing=self.trainer_kwargs['enable_checkpointing'],
            # log_every_n_epochs=self.trainer_kwargs['log_every_n_epochs'],
            log_every_n_steps=1,
            max_epochs=self.trainer_kwargs['max_epochs'],
            callbacks=[chkpt_callback],
            logger=logger
        )

    
    def train(self):
        self.trainer.fit(self.model, self.train_dl)
    
    # def embed_sections(self):
    #     if self.chkpt_filepath is None:
    #         raise RuntimeError('Must either train model or load a model checkpoint. To train, run .train()')

    #     self.recon_imgs, self.recon_embs, self.recon_cluster_ids, self.recon_cluster_probs = self.model.embed_sections()
    #     self.recon_cluster_ids = self.recon_cluster_ids.to(torch.long)

    #     self.cluster_ids = utils.relabel(self.recon_cluster_ids)
    #     self.cluster_probs = self.recon_cluster_probs[:, torch.unique(self.recon_cluster_ids)]

    #     scalers = torch.amax(self.recon_imgs, dim=(-2, -1))
    #     self.scaled_recon_imgs = self.recon_imgs / rearrange(scalers, 'n c -> n c 1 1')

    def get_cluster_intensities(self, cluster_ids):
        data = []
        x = rearrange(self.scaled_recon_imgs, 'n c h w -> n h w c').clone().detach().cpu().numpy()
        for cluster in np.unique(cluster_ids):
            mask = cluster_ids==cluster
            data.append(x[mask, :].mean(axis=0))
        df = pd.DataFrame(data=data, columns=self.learner_data.channels, index=np.unique(cluster_ids))
        return df

    def display_predicted_pixels(self, channel=None, figsize=None):
        if self.recon_imgs is None:
            raise RuntimeError(
                'Must train model and embed sections before displaying. To embed run .embed_sections()')
        channel = channel if channel is not None else self.learner_data.channels[0]
        fig, axs = plt.subplots(nrows=2, ncols=self.recon_imgs.shape[0], figsize=figsize)

        for sid, img, ax in zip(self.section_ids, self.recon_imgs, axs[0, :]):
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
