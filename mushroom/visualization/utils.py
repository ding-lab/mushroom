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
from matplotlib.colors import CSS4_COLORS, LinearSegmentedColormap
from PIL import Image
from skimage.exposure import adjust_gamma

SEQUENTIAL_CMAPS = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

COLORS = [
    np.asarray([52, 40, 184]) / 255., # blue
    np.asarray([204, 137, 4]) / 255., # orange
    np.asarray([25, 117, 5]) / 255., # green
    np.asarray([161, 19, 14]) / 255., # red
    np.asarray([109, 12, 148]) / 255., # purple
    np.asarray([92, 60, 1]) / 255., # brown
    np.asarray([224, 54, 185]) / 255., # pink
    np.asarray([2, 194, 191]) / 255., # cyan
]
COLORS += CSS4_COLORS


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

def get_hierarchical_cmap(label_to_hierarchy):
    aggs = np.stack(list(label_to_hierarchy.values()))
    n_clusts = aggs.max(0) + 1

    n_maps = n_clusts[0]

    color_endpoints = COLORS[:n_maps]
    
    label_to_color = {}
    for label, agg in label_to_hierarchy.items():
        if len(agg) == 1:
            label_to_color[label] = LinearSegmentedColormap.from_list('a', ['white', color_endpoints[label]], N=100)(.8)
        elif len(agg) == 2:
            n_colors = n_clusts[1]
            val = (agg[-1] + 1) / n_colors
            label_to_color[label] = LinearSegmentedColormap.from_list('a', ['white', color_endpoints[agg[0]]], N=100)(val)
        else:
            n_colors = np.product(n_clusts[1:])
            arr = np.arange(n_colors)
            for i in range(1, len(agg) - 1):
                x, max_c = agg[i], n_clusts[i + 1]
                arr = arr[x * max_c:(x + 1) * max_c]
            idx = arr[agg[-1]]
            val = (idx + 1) / n_colors
            label_to_color[label] = LinearSegmentedColormap.from_list('a', ['white', color_endpoints[agg[0]]], N=100)(val)
    label_to_color = {k:v[:3] for k, v in label_to_color.items()}
    return label_to_color

def display_labeled_as_rgb(labeled, cmap=None, preserve_indices=True, label_to_hierarchy=None):
    if isinstance(labeled, torch.Tensor):
        labeled = labeled.numpy()
    
    if label_to_hierarchy is not None:
        cmap = get_hierarchical_cmap(label_to_hierarchy)
    elif preserve_indices:
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


def display_clusters(clusters, cmap=None, figsize=None, horizontal=True, preserve_indices=False, return_axs=False, label_to_hierarchy=None):
    if figsize is None:
        figsize = (clusters.shape[0] * 2, 5)
        if not horizontal:
            figsize = (figsize[1], figsize[0])

    if horizontal:
        fig, axs = plt.subplots(ncols=clusters.shape[0] + 1, figsize=figsize)
    else:
        fig, axs = plt.subplots(nrows=clusters.shape[0] + 1, figsize=figsize)

    if cmap is None:
        if label_to_hierarchy is not None:
            cmap = get_hierarchical_cmap(label_to_hierarchy)
        else:
            cmap = get_cmap(len(np.unique(clusters)))
    elif isinstance(cmap, str):
        cmap = sns.color_palette(cmap)

    for i, labeled in enumerate(clusters):
        axs[i].imshow(display_labeled_as_rgb(labeled, cmap=cmap, preserve_indices=preserve_indices, label_to_hierarchy=label_to_hierarchy))
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    display_legend(np.unique(clusters), cmap, ax=axs[-1], label_to_hierarchy=label_to_hierarchy)
    axs[-1].axis('off')

    if return_axs:
        return axs


def display_legend(labels, cmap, ax=None, label_to_hierarchy=None):
    if ax is None:
        fig, ax = plt.subplots()

    if label_to_hierarchy is not None:
        xs = [agg + (l,) for l, agg in label_to_hierarchy.items()]
        order = [x[-1] for x in sorted(xs)]
        ax.legend(
            handles=[mpatches.Patch(color=cmap[label], label=label)
                    for label in order],
            loc='center'
        )
    else:
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
      