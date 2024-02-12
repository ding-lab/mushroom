import contextlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
import tifffile
from einops import rearrange, repeat
from PIL import Image
from skimage.exposure import adjust_gamma


def get_cmap(n):
    if n < 10:
        return sns.color_palette()
    if n < 20:
        return sns.color_palette('tab20')
    
    iters = (n // 20) + 2
    cmap = []
    for i in range(iters):
        cmap += sns.color_palette('tab20')
    
    return cmap


def display_labeled_as_rgb(labeled, cmap=None, preserve_indices=False):
    if isinstance(labeled, torch.Tensor):
        labeled = labeled.numpy()
    
    if preserve_indices:
        cmap = get_cmap(labeled.max() + 1) if cmap is None else cmap
    else:
        cmap = get_cmap(len(np.unique(labeled))) if cmap is None else cmap

    labels = sorted(np.unique(labeled))
    if len(cmap) < len(labels):
        raise RuntimeError('cmap is too small')
    new = np.zeros((labeled.shape[0], labeled.shape[1], 3))
    for i, l in enumerate(labels):
        if preserve_indices:
            c = cmap[l]
        else:
            c = cmap[i]
        new[labeled==l] = c
    return new


def display_clusters(clusters, cmap=None, figsize=None, horizontal=True, preserve_indices=False, return_axs=False):
    if figsize is None:
        figsize = (clusters.shape[0] * 2, 5)
        if not horizontal:
            figsize = (figsize[1], figsize[0])

    if horizontal:
        fig, axs = plt.subplots(ncols=clusters.shape[0] + 1, figsize=figsize)
    else:
        fig, axs = plt.subplots(nrows=clusters.shape[0] + 1, figsize=figsize)

    if cmap is None:
        cmap = get_cmap(len(np.unique(clusters)))
    elif isinstance(cmap, str):
        cmap = sns.color_palette(cmap)

    for i, labeled in enumerate(clusters):
        axs[i].imshow(display_labeled_as_rgb(labeled, cmap=cmap, preserve_indices=preserve_indices))
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    display_legend(np.unique(clusters), cmap, ax=axs[-1])
    axs[-1].axis('off')

    if return_axs:
        return axs


def display_legend(labels, cmap, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.legend(
        handles=[mpatches.Patch(color=color, label=label)
                 for label, color in zip(labels, cmap)],
        loc='center'
    )
    # return ax


def show_groups(clusters, groups):
    mapping = {i:i for i in np.unique(clusters)}
    mapping.update({i:-1 for i in np.unique(clusters) if i not in groups})
    neigh_ids = np.vectorize(mapping.get)(clusters)
    display_clusters(neigh_ids)




def volume_to_gif(volume, is_probs, filepath, axis=0, duration=200):
    if is_probs:
        imgs = [repeat(img, 'h w -> h w c', c=3)
                    for img in volume.swapaxes(0, axis)]
    else:
        imgs = [display_labeled_as_rgb(labeled, preserve_indices=True)
                for labeled in volume.swapaxes(0, axis)]
    imgs = [((img / img.max()) * 255.).astype(np.uint8) for img in imgs]

    ims = [Image.fromarray(x, mode="RGB")
                for x in imgs]
    ims[0].save(fp=filepath, format='GIF', append_images=ims,
             save_all=True, duration=duration, loop=0)
    
def save_reference_gif(rgb, filepath, axis=0, duration=200, color=(1., 0., 0.), thickness=1):
    if not isinstance(color, np.ndarray):
        color = np.asarray(color)
    if rgb.max() <= 1.:
        rgb /= rgb.max()
        rgb *= 255.
        rgb = rgb.astype(np.uint8)
        color = (color * 255.).astype(np.uint8)
    
    n_frames = rgb.shape[axis]
    imgs = []
    for frame in range(n_frames):
        selections = tuple([slice(None) if i!=axis else slice(i, i+thickness) for i in range(n_frames)])
        selections = tuple([slice(None) if i!=axis else slice(frame, frame + thickness) for i in range(len(rgb.shape))])
        new = rgb.copy()
        new[selections] = color
        imgs.append(new)
        
    ims = [Image.fromarray(x, mode="RGB")
                for x in imgs]
    ims[0].save(fp=filepath, format='GIF', append_images=ims,
             save_all=True, duration=duration, loop=0)
      