import os
import re

import numpy as np
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
import numpy as np
from einops import rearrange
from skimage.exposure import rescale_intensity
from tifffile import TiffFile
from ome_types import from_tiff, from_xml


def listfiles(folder, regex=None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_means_and_stds(adatas):
    means, stds = None, None
    for a in adatas:
        x = rearrange(next(iter(a.uns['spatial'].values()))['images']['lowres'], 'h w c -> c h w')

        if means is None:
            means = x.mean((1, 2))
            stds = x.std((1, 2))
        else:
            means = (means + x.mean((1, 2))) / 2
            stds = (stds + x.std((1, 2))) / 2
    return means, stds


def extract_he_from_adata(adata):
    """Extract hires H&E from adata object"""
    return next(iter(adata.uns['spatial'].values()))['images']['hires']


def rescale_img(img, scale=.5, shape=None):
    if shape is None:
        h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    else:
        h, w = shape
    scaled = TF.resize(rearrange(torch.Tensor(img), 'h w c -> c h w'), size=(h, w))
    scaled = rearrange(scaled, 'c h w -> h w c').numpy()
    
    if scaled.max() > 1.:
        return scaled.astype(np.uint8)
    return scaled


def rescale_with_pad(img, scale=.5, shape=None, padding_mode='reflect'):
    if shape is None:
        h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    else:
        h, w = shape
    r, c = img.shape[0], img.shape[1]
    
    right_pad = r-c if r>c else 0
    bottom_pad = c-r if c>r else 0
    padded = TF.pad(rearrange(torch.Tensor(img), 'h w c -> c h w'),
                    padding=[0, 0, right_pad, bottom_pad], padding_mode=padding_mode)
    
    scaled = TF.resize(padded, size=(h, w))
    return rearrange(scaled, 'c h w -> h w c').numpy().astype(np.uint8)


def project_expression(labeled, exp, voxel_idxs):

    new = torch.zeros((labeled.shape[-2], labeled.shape[-1], exp.shape[1]), dtype=exp.dtype)
    for i, idx in enumerate(voxel_idxs):
        new[labeled.squeeze()==idx] = exp[i]
    return new


def construct_tile_expression(padded_exp, masks, n_voxels, normalize=True):
    tile = torch.zeros((masks.shape[0], masks.shape[-2], masks.shape[-1], padded_exp.shape[-1]),
                       device=padded_exp.device)
    for b in range(tile.shape[0]):
        for exp, m in zip(padded_exp[b], masks[b]):
            tile[b, :, :][m==1] = exp.to(torch.float32)
            
    tile = rearrange(tile, 'b h w c -> b c h w')
    tile = tile.detach().cpu().numpy()
    
    tile /= np.expand_dims(tile.max(axis=(0, -2, -1)), (0, -2, -1))

    return rearrange(tile, 'b c h w -> b h w c')


def extract_ome_tiff(fp, channels=None):   
    tif = TiffFile(fp)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    d = {}
    img_channels = []
    for c, p in zip(im.pixels.channels, tif.pages):
        img_channels.append(c.name)

        if channels is None:
            img = p.asarray()
            d[c.name] = img
        elif c.name in channels:
            img = p.asarray()
            d[c.name] = img

    if channels is not None and len(set(channels).intersection(set(img_channels))) != len(channels):
        raise RuntimeError(f'Not all channels were found in ome tiff: {channels} | {img_channels}')

    return d


def get_ome_tiff_channels(fp):   
    tif = TiffFile(fp)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    return [c.name for c in im.pixels.channels]


def make_pseudo(channel_to_img, cmap=None, contrast_pct=20.):
    cmap = sns.color_palette('tab10') if cmap is None else cmap

    new = np.zeros_like(next(iter(channel_to_img.values())))
    img_stack = []
    for i, (channel, img) in enumerate(channel_to_img.items()):
        color = cmap[i] if not isinstance(cmap, dict) else cmap[channel]
        new = img.copy().astype(np.float32)
        new -= new.min()
        new /= new.max()

        try:
            vmax = np.percentile(new[new>0], (contrast_pct)) if np.count_nonzero(new) else 1.
            new = rescale_intensity(new, in_range=(0., vmax))
        except IndexError:
            pass

        new = np.repeat(np.expand_dims(new, -1), 3, axis=-1)
        new *= color
        img_stack.append(new)
    stack = np.mean(np.asarray(img_stack), axis=0)
    stack -= stack.min()
    stack /= stack.max()
    return stack