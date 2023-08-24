import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import KMeans
from torchio.transforms import Resize

from mushroom.visualization.utils import display_labeled_as_rgb


class EmbeddingClusterer(object):
    def __init__(self, n_clusters=20, section_imgs=None, section_masks=None, margin=.05):
        self.kmeans = KMeans(n_clusters=n_clusters, n_init='auto')

        self.section_masks = self.autogenerate_section_masks(
            section_imgs, margin=margin) if section_imgs is not None else section_masks
        
    def autogenerate_section_masks(
            self,
            imgs, # (n_sections, height, width), where height, width is same size as embeddings
            kernel_size=(3, 3),
            margin=.05,
        ):
        imgs /= rearrange(imgs.std((-2, -1)), 'n -> n 1 1')
        imgs /= imgs.max()
        imgs = TF.gaussian_blur(imgs, kernel_size=kernel_size).numpy()

        thresholds = rearrange(
            np.asarray([threshold_otsu(img) - margin for img in imgs]),
            'n -> n 1 1'
        )

        masks = imgs > thresholds
        masks = torch.stack([
            torch.tensor(binary_fill_holes(mask))
            for mask in masks
        ])
        return masks
    
    def set_section_masks(self, section_masks):
        self.section_masks = section_masks

    def fit_transform(self, recon_embs, mask_background=True, add_background_cluster=True):
        if (mask_background or add_background_cluster) and self.section_masks is None:
            raise RuntimeError('To mask background pixels section masks indicating section background pixels must be set with either .autogenerate_section_mask or .set_section_masks.')
        
        x = rearrange(recon_embs, 'n c h w -> n h w c')
        n, h, w = x.shape[:-1]
        
        if mask_background:
            x = x[self.section_masks].numpy()
        else:
            x = rearrange(x, 'n h w c -> (n h w) c').numpy()

        self.kmeans.fit(x)
        dists = self.kmeans.transform(rearrange(recon_embs, 'n c h w -> (n h w) c').numpy())
        cluster_ids = torch.tensor(dists.argmin(1))

        dists = rearrange(torch.tensor(dists), '(n h w) d -> n h w d', n=n, h=h, w=w)
        cluster_ids = rearrange(cluster_ids, '(n h w) -> n h w', n=n, h=h, w=w)

        if add_background_cluster and mask_background:
            cluster_ids[~self.section_masks] = cluster_ids.max() + 1
            dists[~self.section_masks] = dists.amax((0, 1, 2))

        return dists, cluster_ids
    
    def interpolate_distances(
            self,
            dists, # (n h w d)
            section_positions, # (n,)
            section_range=None # (start, stop), defaults to (min(section_positions), max(section_positions))
        ):
        if section_range is None:
            section_range = (section_positions.min(), section_positions.max())

        dist_volume = torch.full((section_range[-1], dists.shape[1], dists.shape[2], dists.shape[3]), dists.max())
        for i in range(dists.shape[0] - 1):
            l1, l2 = section_positions[i], section_positions[i+1]
            stack = rearrange(dists[i:i+2], 'n h w d -> d n h w')
            transform = Resize((l2 - l1, stack.shape[-2], stack.shape[-1]))
            resized = transform(stack)
            dist_volume[l1:l2] = rearrange(resized, 'd n h w -> n h w d')

        if section_range[1] > section_positions.max():
            x = dist_volume[-1]
            x = repeat(x, 'h w d -> n h w d', n=section_range[1] - section_positions.max())
            dist_volume = torch.cat((dist_volume, x))

        return dist_volume
    
    def display_section_masks(self):
        fig, axs = plt.subplots(ncols=self.section_masks.shape[0])
        for mask, ax in zip(self.section_masks, axs):
            ax.imshow(mask)
            ax.axis('off')

    def display_cluster_ids(self, cluster_ids, cmap=None, figsize=None):
        if cmap is None:
            cmap = sns.color_palette('tab20') + sns.color_palette('tab20b') + sns.color_palette('tab20c')
        fig, axs = plt.subplots(ncols=cluster_ids.shape[0], figsize=figsize)
        for i, labeled in enumerate(cluster_ids):
            axs[i].imshow(display_labeled_as_rgb(labeled, cmap=cmap))
            axs[i].set_xticks([])
            axs[i].set_yticks([])


    def display_distances(self, dists, figsize=None, cmap='viridis_r'):
        fig, axs = plt.subplots(ncols=dists.shape[0], nrows=dists.shape[-1], figsize=figsize)
        for r in range(dists.shape[-1]):
            row_vmax = dists[..., r].max()
            row_vmin = dists[..., r].min()
            for c in range(dists.shape[0]):
                ax = axs[r, c]
                img = dists[c, ..., r]
                ax.imshow(img, cmap=cmap, vmax=row_vmax, vmin=row_vmin)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(r)
