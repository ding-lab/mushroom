import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
import tifffile
from einops import rearrange

import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
import mushroom.data.he as he

def get_cmap(n):
    if n <= 10:
        return sns.color_palette()
    if n < 20:
        return sns.color_palette('tab20')
    
    iters = (n // 20) + 1
    cmap = []
    for i in range(iters):
        cmap += sns.color_palette('tab20')
    
    return cmap

def calculate_target_visualization_shape(config, output_data, pixels_per_micron=None, microns_per_section=5):
    scaled_size = output_data['true_imgs'].shape
    sections = config['sections']
    if pixels_per_micron is None:
        for s in sections:
            for entry in s['data']:
                if entry['dtype'] == 'visium':
                    try:
                        pixels_per_micron = visium.pixels_per_micron(entry['filepath'])
                    except:
                        pass
                if entry['dtype'] == 'multiplex':
                    try:
                        pixels_per_micron = multiplex.pixels_per_micron(entry['filepath'])
                    except:
                        pass
                
                if pixels_per_micron is not None: break
            if pixels_per_micron is not None: break

    step_size = pixels_per_micron * microns_per_section * config['learner_kwargs']['scale']
    z_max = int(step_size * np.max([entry['position'] for entry in sections]))
    
    target_shape = (scaled_size[-1], scaled_size[-2], z_max)

    return target_shape

 
def display_sections(config, multiplex_cmap=None, gene='EPCAM', dtype_order=None):
    def rescale(x, scale=.1):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = rearrange(x, 'h w c -> c h w')
        x = TF.resize(x, (int(x.shape[-2] * scale), int(x.shape[-1] * scale)), antialias=True)
        x = rearrange(x.numpy(), 'c h w -> h w c')    
        return x

    if dtype_order is None:
        dtype_order = sorted({d['dtype'] for entry in config for d in entry['data']})

    fig, axs = plt.subplots(ncols=len(config), nrows=len(dtype_order))
    axs_mask = np.zeros_like(axs, dtype=bool)
    config = sorted(config, key=lambda x: x['position'])
    for c, entry in enumerate(config):
        for d in entry['data']:
            dtype, filepath = d['dtype'], d['filepath']
            r = dtype_order.index(dtype)
            ax = axs[r, c]

            if dtype == 'multiplex':
                channel_to_img = multiplex.extract_ome_tiff(filepath, channels=list(multiplex_cmap.keys()))
                channel_to_img = {channel:np.squeeze(rescale(np.expand_dims(img, -1), scale=.1))
                     for channel, img in channel_to_img.items()}

                pseudo = multiplex.make_pseudo(channel_to_img, cmap=multiplex_cmap, contrast_pct=90.)
                pseudo /= pseudo.max()
                ax.imshow(pseudo)
            elif dtype == 'he':
                he = tifffile.imread(filepath)
                he = rescale(he, .1)
                ax.imshow(he)
            elif dtype == 'visium':
                adata = sc.read_h5ad(filepath)
                d = next(iter(adata.uns['spatial'].values()))
                scale = d['scalefactors']['tissue_hires_scalef']
                h, w = int(d['images']['hires'].shape[0] / scale), int(d['images']['hires'].shape[1] / scale)
                ax = sc.pl.spatial(
                    adata, color=gene, alpha_img=.0, ax=ax, show=False, title='',
                    colorbar_loc=None, crop_coord=(0, w, 0, h))[0]
                axs[r, c] = ax
                
            axs_mask[r, c] = True
        
    for ax in axs[~axs_mask].flatten():
        ax.imshow(np.full((1, 1, 3), .8))
        ax.set_xticks([])
        ax.set_yticks([])
        
    for ax in axs[axs_mask].flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        
    for ax, dtype in zip(axs[:, 0], dtype_order):
        ax.set_ylabel(dtype)
    for ax, sid in zip(axs[0, :], [item['id'] for item in config]):
        ax.set_title(sid)


def display_labeled_as_rgb(labeled, cmap=None, preserve_indices=False):
    if isinstance(labeled, torch.Tensor):
        labeled = labeled.numpy()
    
    if preserve_indices:
        cmap = get_cmap(labeled.max()) if cmap is None else cmap
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


def display_clusters(clusters, cmap=None, figsize=None, horizontal=True, preserve_indices=False):
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
    