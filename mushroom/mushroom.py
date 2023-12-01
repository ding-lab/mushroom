import logging
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from einops import rearrange
from torchio.transforms import Resize

from mushroom.model.sae import SAEargs
from mushroom.model.learners import SAELearner
from mushroom.data.visium import format_expression
import mushroom.utils as utils


class Mushroom(object):
    def __init__(
            self,
            dtype,
            sections,
            chkpt_filepath=None,
            sae_kwargs=None,
            learner_kwargs=None,
            train_kwargs=None,
            cluster_kwargs=None,
        ):
        self.dtype = dtype
        self.sections = sections
        self.chkpt_filepath = chkpt_filepath
        self.sae_kwargs = sae_kwargs
        self.learner_kwargs = learner_kwargs
        self.train_kwargs = train_kwargs
        self.section_ids = [entry['id'] for entry in self.sections
                       if dtype in [d['dtype'] for d in entry['data']]]

        self.learner = self.initialize_learner()

        logging.info('learner initialized')

        # make a groundtruth reconstruction for original images
        self.true_imgs = torch.stack(
            [self.learner.inference_ds.image_from_tiles(self.learner.inference_ds.section_to_tiles[s])
            for s in self.learner.inference_ds.sections]
        )
        if self.dtype in ['visium']:
            self.true_imgs = torch.stack(
                [format_expression(
                    img, self.learner.inference_ds.section_to_adata[sid], self.learner.sae_args.patch_size
                ) for sid, img in zip(self.section_ids, self.true_imgs)]
            )
            
        self.recon_embs, self.recon_imgs, self.recon_cluster_ids, self.recon_cluster_probs = None, None, None, None

        self.cluster_probs, self.cluster_ids, self.scaled_recon_imgs = None, None, None

    @staticmethod
    def from_config(mushroom_config):
        if isinstance(mushroom_config, str):
            mushroom_config = yaml.safe_load(open(mushroom_config))
        return Mushroom(
            mushroom_config['dtype'],
            mushroom_config['sections'],
            chkpt_filepath=mushroom_config['chkpt_filepath'],
            sae_kwargs=mushroom_config['sae_kwargs'],
            learner_kwargs=mushroom_config['learner_kwargs'],
            train_kwargs=mushroom_config['train_kwargs'],
        )
    
    def _get_section_imgs(self, args):
        emb_size = int(self.true_imgs.shape[-2] / self.learner.train_transform.output_patch_size)
        section_imgs = TF.resize(self.true_imgs, (emb_size, emb_size), antialias=True)

        if args.background_channels is None and args.mask_background:
            logging.info('no background channel detected, defaulting to mean of all channels')
            section_imgs = section_imgs.mean(1)
        elif args.background_channels is not None:
            idxs = [self.learner.channels.index(channel) for channel in args.background_channels]
            section_imgs = section_imgs[:, idxs].mean(1)
        else:
            section_imgs = None

        return section_imgs

    def initialize_learner(self):
        learner = SAELearner(
            self.sections,
            self.dtype,
            sae_args=SAEargs(**self.sae_kwargs) if self.sae_kwargs is not None else {},
            **self.learner_kwargs if self.learner_kwargs is not None else {}
        )

        if self.chkpt_filepath is not None:
            learner.sae.load_state_dict(torch.load(self.chkpt_filepath))

        return learner

    def save_config(self, filepath): 
        mushroom_config = {
            'dtype': self.dtype,
            'sections': self.sections,
            'chkpt_filepath': self.chkpt_filepath,
            'sae_kwargs': self.sae_kwargs if self.sae_kwargs is not None else {},
            'learner_kwargs': self.learner_kwargs if self.learner_kwargs is not None else {},
            'train_kwargs': self.train_kwargs if self.train_kwargs is not None else {},
        }
        yaml.safe_dump(mushroom_config, open(filepath, 'w'))

    def save_outputs(self, filepath):
        obj = {
            'recon_embs': self.recon_embs,
            'recon_imgs': self.recon_imgs,
            'true_imgs': self.true_imgs,
            'cluster_sims': self.cluster_sims,
            'cluster_ids': self.cluster_ids,
        }
        torch.save(obj, filepath)
    
    def train(self, **kwargs):
        self.train_kwargs.update(kwargs)
        self.learner.train(**self.train_kwargs)
        self.chkpt_filepath = (os.path.join(self.train_kwargs['save_dir'], 'final.pt') # final ckpt is final.pt
                               if 'save_dir' in self.train_kwargs else 'final.pt')
    
    def embed_sections(self):
        if self.chkpt_filepath is None:
            raise RuntimeError('Must either train model or load a model checkpoint. To train, run .train()')

        self.recon_imgs, self.recon_embs, self.recon_cluster_ids, self.recon_cluster_probs = self.learner.embed_sections()
        self.recon_cluster_ids = self.recon_cluster_ids.to(torch.long)

        self.cluster_ids = utils.relabel(self.recon_cluster_ids)
        self.cluster_probs = self.recon_cluster_probs[:, torch.unique(self.recon_cluster_ids)]

        scalers = torch.amax(self.recon_imgs, dim=(-2, -1))
        self.scaled_recon_imgs = self.recon_imgs / rearrange(scalers, 'n c -> n c 1 1')

    def get_cluster_intensities(self, cluster_ids):
        data = []
        x = rearrange(self.scaled_recon_imgs, 'n c h w -> n h w c').clone().detach().cpu().numpy()
        for cluster in np.unique(cluster_ids):
            mask = cluster_ids==cluster
            data.append(x[mask, :].mean(axis=0))
        df = pd.DataFrame(data=data, columns=self.learner.channels, index=np.unique(cluster_ids))
        return df

    def display_predicted_pixels(self, channel=None, figsize=None):
        if self.recon_imgs is None:
            raise RuntimeError(
                'Must train model and embed sections before displaying. To embed run .embed_sections()')
        channel = channel if channel is not None else self.learner.channels[0]
        fig, axs = plt.subplots(nrows=2, ncols=self.recon_imgs.shape[0], figsize=figsize)

        for sid, img, ax in zip(self.section_ids, self.recon_imgs, axs[0, :]):
            ax.imshow(img[self.learner.channels.index(channel)])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(sid)

        for sid, img, ax in zip(self.section_ids, self.true_imgs, axs[1, :]):
            ax.imshow(img[self.learner.channels.index(channel)])
            ax.set_xticks([])
            ax.set_yticks([])
        axs[0, 0].set_ylabel('predicted')
        axs[1, 0].set_ylabel('true')

        return axs
