import os
import logging
import re
import sys

import numpy as np
import seaborn as sns
import scanpy as sc
import tifffile
import torch
import torchvision.transforms.functional as TF
import numpy as np
from einops import rearrange
from skimage.exposure import rescale_intensity
from tifffile import TiffFile
from ome_types import from_tiff, from_xml, to_xml, model


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


def flexible_rescale(img, scale=.5, size=None):
    if size is None:
        size = int(img.shape[0] * scale), int(img.shape[1] * scale)

    if not isinstance(img, torch.Tensor):
        is_tensor = False
        img = torch.tensor(img)
    else:
        is_tensor = True

    if img.shape[0] not in [1, 3]:
        channel_first = False
        img = rearrange(img, 'h w c -> c h w')
    else:
        channel_first = True

    img = TF.resize(img, size=size)

    if not channel_first:
        img = rearrange(img, 'c h w -> h w c')

    if not is_tensor:
        img = img.numpy()

    return img


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


def adata_from_visium(fp):
    sid = fp.split('/')[-1]
    a = sc.read_visium(fp)
    a.var_names_make_unique()
    a.obsm['spatial'] = a.obsm['spatial'].astype(int)
    return a


def save_ome_tiff(channel_to_img, filepath, bbox=None, pixel_type='uint8', subresolutions=4):
    """
    Generate an ome tiff from channel to image map
    """
    n_channels = len(channel_to_img)
    shape = next(iter(channel_to_img.values())).shape


    if bbox is None:
        data = np.zeros((shape[1], shape[0], n_channels, 1, 1),
                        dtype=np.uint8 if pixel_type=='uint8' else np.uint16)
    else:
        logging.info(f'bbox detected: {bbox}')
        data = np.zeros((bbox[3] - bbox[2], bbox[1] - bbox[0], n_channels, 1, 1),
                        dtype=np.uint8 if pixel_type=='uint8' else np.uint16)
    
    biomarkers = sorted(channel_to_img.keys())
    for i, biomarker in enumerate(biomarkers):
        img = channel_to_img[biomarker]
        img = tifffile.imread(fp)
        if bbox is not None:
            r1, r2, c1, c2 = bbox
            y = r2 - r1
            x = c2 - c1
            img = img[r1:r2, c1:c2]
        img = img.astype(np.float32)
        img -= img.min()
        img /= img.max()
        if pixel_type == 'uint8':
            img *= 255
            img = img.astype(np.uint8)
        else:
            img *= 65535
            img = img.astype(np.uint16) 

        if data is None:
            if bbox is None:
                data = np.zeros((img.shape[1], img.shape[0], n_channels, 1, 1),
                                dtype=np.uint8 if pixel_type=='uint8' else np.uint16)
            else:
                logging.info(f'bbox detected, cropping to {bbox}')
                data = np.zeros((bbox[3] - bbox[2], bbox[1] - bbox[0], n_channels, 1, 1),
                                dtype=np.uint8 if pixel_type=='uint8' else np.uint16)
        else:
            data[..., i, 0, 0] = np.swapaxes(img, 0, 1)
    
    o = model.OME()
    o.images.append(
        model.Image(
            id='Image:0',
            pixels=model.Pixels(
                dimension_order='XYCZT',
                size_c=n_channels,
                size_t=1,
                size_x=data.shape[0],
                size_y=data.shape[1],
                size_z=data.shape[3],
                type=pixel_type,
                big_endian=False,
                channels=[model.Channel(id=f'Channel:{i}', name=c) for i, c in enumerate(biomarkers)],
                # physical_size_x=1 / PHENOCYCLER_PIXELS_PER_MICRON,
                # physical_size_y=1 / PHENOCYCLER_PIXELS_PER_MICRON,
                # physical_size_x_unit='µm',
                # physical_size_y_unit='µm'
            )
        )
    )

    im = o.images[0]
    for i in range(len(im.pixels.channels)):
        im.pixels.planes.append(model.Plane(the_c=i, the_t=0, the_z=0))
    im.pixels.tiff_data_blocks.append(model.TiffData(plane_count=len(im.pixels.channels)))

    with tifffile.TiffWriter(filepath, ome=True, bigtiff=True) as out_tif:
        opts = {
            'compression': 'LZW',
        }
        out_tif.write(
            rearrange(data, 'x y c z t -> t c y x z'),
            subifds=subresolutions,
            **opts
        )
        for level in range(subresolutions):
            mag = 2**(level + 1)
            x = torch.tensor(rearrange(data[..., 0, 0], 'w h c -> c h w'))
            sampled = rearrange(
                TF.resize(x, (int(x.shape[-2] / mag), int(x.shape[-1] / mag)), antialias=True),
                'c h w -> w h c 1 1'
            )
            sampled = sampled.numpy().astype(np.uint8)
            out_tif.write(
                rearrange(sampled, 'x y c z t -> t c y x z'),
                subfiletype=1,
                **opts
            )
        xml_str = to_xml(o)
        out_tif.overwrite_description(xml_str.encode())


def coords_to_labeled_slices(xyz, clusters):
    cluster_pool = np.unique(clusters)
    z_pool = np.unique(xyz[:, -1])
    slices = []
    for slice in z_pool:
        f_xyz, f_clusters = xyz[xyz[:, -1]==slice], clusters[xyz[:, -1]==slice]
        img = np.zeros((xyz[:, 1].max() + 1, xyz[:, 0].max() + 1))
        for cluster in cluster_pool:
            for x, y, z in f_xyz[f_clusters==cluster]:
                img[y, x] = int(cluster)
        slices.append(img) 
    return slices


def display_labeled_as_rgb(labeled, cmap=None):
    if isinstance(labeled, torch.Tensor):
        labeled = labeled.numpy()
    cmap = sns.color_palette() if cmap is None else cmap
    labels = sorted(np.unique(labeled))
    if len(cmap) < len(labels):
        raise RuntimeError('cmap is too small')
    new = np.zeros((labeled.shape[0], labeled.shape[1], 3))
    for l, c in zip(labels, cmap):
        new[labeled==l] = c
    return new


class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        