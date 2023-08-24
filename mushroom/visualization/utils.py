import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torchvision.transforms.functional as TF
import tifffile
from einops import rearrange

from mushroom.data.multiplex import extract_ome_tiff, make_pseudo

 
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
                channel_to_img = extract_ome_tiff(filepath, channels=list(multiplex_cmap.keys()))
                channel_to_img = {channel:np.squeeze(rescale(np.expand_dims(img, -1), scale=.1))
                     for channel, img in channel_to_img.items()}

                pseudo = make_pseudo(channel_to_img, cmap=multiplex_cmap, contrast_pct=90.)
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


def display_labeled_as_rgb(labeled, cmap=None):
    if isinstance(labeled, torch.Tensor):
        labeled = labeled.numpy()
    cmap = sns.color_palette() if cmap is None else cmap
    labels = sorted(np.unique(labeled))
    if len(cmap) < len(labels):
        raise RuntimeError('cmap is too small')
    new = np.zeros((labeled.shape[0], labeled.shape[1], 3))
    for l in labels:
        c = cmap[l]
        new[labeled==l] = c
    return new