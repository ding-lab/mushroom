import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import yaml

from mushroom.model.sae import SAEargs
from mushroom.model.learners import SAELearner
from mushroom.clustering import EmbeddingClusterer


class Mushroom(object):
    def __init__(
            self,
            dtype,
            sections,
            chkpt_filepath=None,
            sae_kwargs=None,
            learner_kwargs=None,
            train_kwargs=None,
        ):
        self.dtype = dtype
        self.sections = sections
        self.chkpt_filepath = chkpt_filepath
        self.sae_kwargs = sae_kwargs
        self.learner_kwargs = learner_kwargs
        self.train_kwargs = train_kwargs
        self.section_ids = [entry['id'] for entry in self.sections
                       if dtype in [d['dtype'] for d in entry['data']]]

        self.learner = self.initialize_learner(
            self.sections,
            self.dtype,
            sae_kwargs=self.sae_kwargs,
            learner_kwargs=self.learner_kwargs,
            chkpt_filepath=self.chkpt_filepath
        )

        self.recon_embs, self.recon_imgs, self.true_imgs = None, None, None
        self.clusterer = None
        self.dists, self.cluster_ids, self.dists_volume = None, None, None

    @staticmethod
    def from_config(mushroom_config):
        return Mushroom(
            mushroom_config['dtype'],
            mushroom_config['sections'],
            chkpt_filepath=mushroom_config['chkpt_filepath'],
            sae_kwargs=mushroom_config['sae_kwargs'],
            learner_kwargs=mushroom_config['learner_kwargs'],
            train_kwargs=mushroom_config['train_kwargs']
        )

    def initialize_learner(self, section_config, dtype, sae_kwargs=None, learner_kwargs=None, chkpt_filepath=None):
        learner = SAELearner(
            section_config,
            dtype,
            sae_args=SAEargs(**sae_kwargs) if sae_kwargs is not None else {},
            **learner_kwargs if learner_kwargs is not None else {}
        )

        if chkpt_filepath is not None:
            learner.sae.load_state_dict(chkpt_filepath)

        return learner
    
    def save(self, filepath):
        mushroom_config = {
            'dtype': self.dtype,
            'sections': self.sections,
            'chkpt_filepath': self.chkpt_filepath,
            'sae_kwargs': self.sae_kwargs if self.sae_kwargs is not None else {},
            'learner_kwargs': self.learner_kwargs if self.learner_kwargs is not None else {},
            'train_kwargs': self.train_kwargs if self.train_kwargs is not None else {}
        }
        yaml.safe_dump(mushroom_config, open(filepath, 'w'))
    
    def train(self, **kwargs):
        self.train_kwargs.update(kwargs)
        self.learner.train(**self.train_kwargs)
        self.chkpt_filepath = (os.path.join(self.train_kwargs['save_dir'], 'final.pt')
                               if 'save_dir' in self.train_kwargs else 'final.pt')
    
    def embed_sections(self):
        if self.chkpt_filepath is None:
            raise RuntimeError('Must either train model or load a model checkpoint. To train, run .train()')

        self.recon_imgs, self.recon_embs = self.learner.embed_sections()

        # make a groundtruth reconstruction for original images
        true_imgs = torch.stack(
            [self.learner.inference_ds.image_from_tiles(self.learner.inference_ds.section_to_tiles[s])
            for s in self.learner.inference_ds.sections]
        )
        self.true_imgs = true_imgs

    def cluster_sections(
            self,
            num_clusters=20,
            mask_background=True,
            add_background_cluster=True,
            margin=.05,
            background_channels=None,
            section_masks=None,
            span_all_sections=False,
        ):
        section_imgs = TF.resize(self.true_imgs, self.recon_embs.shape[-2:], antialias=True)
        if background_channels is None and mask_background:
            logging.info('no background channel detected, defaulting to mean of all channels')
            section_imgs = section_imgs.mean(1)
        elif background_channels is not None:
            idxs = [self.learner.channels.index(channel) for channel in background_channels]
            section_imgs = section_imgs[:, idxs].mean(1)
        else:
            section_imgs = None
        
        self.clusterer = EmbeddingClusterer(
            n_clusters=num_clusters, section_imgs=section_imgs, section_masks=section_masks, margin=margin
        )
        self.dists, self.cluster_ids = self.clusterer.fit_transform(
            self.recon_embs, mask_background=mask_background, add_background_cluster=add_background_cluster
        )

        section_positions = np.asarray(
            [entry['position'] for entry in self.sections if entry['id'] in self.section_ids]
        )
        if span_all_sections:
            all_positions = np.asarray([entry['position'] for entry in self.sections])
            section_range = (all_positions.min(), all_positions.max())
        else:
            section_range = None
        self.dists_volume = self.clusterer.interpolate_distances(
            self.dists, section_positions, section_range=section_range
        )

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
