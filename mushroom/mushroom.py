import logging
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import yaml
from einops import rearrange

from mushroom.model.sae import SAEargs
from mushroom.model.learners import SAELearner
from mushroom.clustering import EmbeddingClusterer, ClusterArgs
from mushroom.data.visium import format_expression


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
        self.cluster_kwargs = cluster_kwargs
        self.section_ids = [entry['id'] for entry in self.sections
                       if dtype in [d['dtype'] for d in entry['data']]]

        self.learner = self.initialize_learner()

        logging.info('learner initialized')

        # print(self.learner.inference_ds.section_to_tiles[s].shape)
        # make a groundtruth reconstruction for original images
        self.true_imgs = torch.stack(
            [self.learner.inference_ds.image_from_tiles(self.learner.inference_ds.section_to_tiles[s])
            for s in self.learner.inference_ds.sections]
        )

        if self.dtype in ['visium']:
            # self.true_imgs = torch.cat(
            #     [format_expression(
            #         img, self.learner.inference_ds.section_to_adata[sid], self.learner.sae_args.patch_size
            #     ) for sid, img in zip(self.section_ids, self.true_imgs)]
            # )
            self.true_imgs = torch.stack(
                [format_expression(
                    img, self.learner.inference_ds.section_to_adata[sid], self.learner.sae_args.patch_size
                ) / rearrange(self.learner.inference_ds.section_to_adata[sid].X.max(0), 'n -> n 1 1') for sid, img in zip(self.section_ids, self.true_imgs)]
            )
            # self.true_imgs = torch.stack(
            #     [format_expression(
            #         img, self.learner.inference_ds.section_to_adata[sid], self.learner.sae_args.patch_size
            #     ) for sid, img in zip(self.section_ids, self.true_imgs)]
            # )
            # print(self.true_imgs.shape)
            
        self.recon_embs, self.recon_imgs, self.recon_embs_prequant = None, None, None

        logging.info('initializing clusterer')
        self.clusterer = self.initialize_clusterer()

        self.dists, self.cluster_ids, self.dists_volume = None, None, None

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
            cluster_kwargs=mushroom_config['cluster_kwargs']
        )
    
    def _get_section_imgs(self, args):
        # if self.recon_imgs is None:
        emb_size = int(self.true_imgs.shape[-2] / self.learner.train_transform.output_patch_size)
        section_imgs = TF.resize(self.true_imgs, (emb_size, emb_size), antialias=True)
        # else:
        #     emb_size = int(self.recon_imgs.shape[-2] / self.learner.train_transform.output_patch_size)
        #     section_imgs = TF.resize(self.recon_imgs, (emb_size, emb_size), antialias=True)

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
    
    def initialize_clusterer(self):
        args = ClusterArgs(**self.cluster_kwargs)
        section_imgs = self._get_section_imgs(args)

        init = 'k-means++' if args.centroids is None else np.asarray(args.centroids)
        clusterer = EmbeddingClusterer(
            n_clusters=args.num_clusters, section_imgs=section_imgs, section_masks=args.section_masks, margin=args.margin, init=init
        )

        return clusterer
    
    def save_config(self, filepath): 
        mushroom_config = {
            'dtype': self.dtype,
            'sections': self.sections,
            'chkpt_filepath': self.chkpt_filepath,
            'sae_kwargs': self.sae_kwargs if self.sae_kwargs is not None else {},
            'learner_kwargs': self.learner_kwargs if self.learner_kwargs is not None else {},
            'train_kwargs': self.train_kwargs if self.train_kwargs is not None else {},
            'cluster_kwargs': self.cluster_kwargs if self.cluster_kwargs is not None else {},
        }
        yaml.safe_dump(mushroom_config, open(filepath, 'w'))

    def save_outputs(self, filepath):
        centroids = torch.tensor(self.cluster_kwargs['centroids']) if self.cluster_kwargs['centroids'] is not None else None
        obj = {
            'recon_embs': self.recon_embs,
            'recon_imgs': self.recon_imgs,
            'true_imgs': self.true_imgs,
            'cluster_distances': self.dists,
            'cluster_distance_volume': self.dists_volume,
            'cluster_centroids': centroids,
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

        self.recon_imgs, self.recon_embs, self.recon_embs_prequant = self.learner.embed_sections()

    def cluster_sections(self, recluster=True, **kwargs):
        self.cluster_kwargs.update(kwargs)
        args = ClusterArgs(**self.cluster_kwargs)
        section_imgs = self._get_section_imgs(args)

        if recluster:
            init = 'k-means++' if args.centroids is None else args.centroids
            self.clusterer = EmbeddingClusterer(
                n_clusters=args.num_clusters, section_imgs=section_imgs, section_masks=args.section_masks, margin=args.margin, init=init
            )
            self.dists, self.cluster_ids = self.clusterer.fit_transform(
                self.recon_embs, mask_background=args.mask_background, add_background_cluster=args.add_background_cluster
            )
            self.cluster_kwargs['centroids'] = self.clusterer.kmeans.cluster_centers_.tolist()
        else:
            self.dists, self.cluster_ids = self.clusterer.transform(
                self.recon_embs, add_background_cluster=args.add_background_cluster
            )

        section_positions = np.asarray(
            [entry['position'] for entry in self.sections if entry['id'] in self.section_ids]
        )
        if args.span_all_sections:
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
