import logging
from collections import Counter

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import tifffile
from einops import rearrange
from ome_types import from_xml, model, to_xml
from skimage.exposure import rescale_intensity
from tifffile import TiffFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Normalize, RandomCrop, Compose

import mushroom.utils as utils
from mushroom.data.inference import InferenceTransform, InferenceSectionDataset
from mushroom.data.utils import LearnerData



def pixels_per_micron(filepath):
    tif = TiffFile(filepath)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    return im.pixels.physical_size_x

def extract_ome_tiff(filepath, channels=None, as_dict=True, flexibility='strict', scale=None, bbox=None):
    tif = TiffFile(filepath)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    d = {}
    img_channels, imgs = [], []

    for c, p in zip(im.pixels.channels, tif.pages):
        if channels is None or c.name in channels:
            img = p.asarray()

            if bbox is not None:
                r1, r2, c1, c2 = bbox
                img = img[r1:r2, c1:c2]

            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                img /= img.max()
                img *= 255.
                img = img.astype(np.uint8)

            if scale is not None:
                img = torch.tensor(img).unsqueeze(0)
                target_size = [int(x * scale) for x in img.shape[-2:]]
                img = TF.resize(img, size=target_size, antialias=True).squeeze().numpy()
            
            d[c.name] = img

            imgs.append(img)
            img_channels.append(c.name)

    if all([
        channels is not None and len(set(channels).intersection(set(img_channels))) != len(channels),
        flexibility=='strict'
        ]):
        raise RuntimeError(f'Not all channels were found in ome tiff: {channels} | {img_channels}')

    if as_dict:
        return d
    
    return img_channels, np.stack(imgs)


def get_ome_tiff_channels(filepath):   
    tif = TiffFile(filepath)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    return [c.name for c in im.pixels.channels]

def get_size(filepath):
    tif = TiffFile(filepath)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    return (im.pixels.size_c, im.pixels.size_y, im.pixels.size_x)


def write_basic_ome_tiff(filepath, data, channels, microns_per_pixel=1.):
    """
    data - (n_channels, height, width)
    """
    assert data.shape[0] == len(channels), f'number of channels is {len(channels)}, must be same length as first dimension of data which is {data.shape}'

    dtype_name = str(np.dtype(data.dype))
    if 'int' in dtype_name:
        assert dtype_name in ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']
    elif 'float' in dtype_name:
        dtype_name = 'float'
    elif dtype_name == 'bool':
        dtype_name = 'bit'
    else:
        raise ValueError(f'dtype {data.dtype} was unable to be saved as ome')

    o = model.OME()
    o.images.append(
        model.Image(
            id='Image:0',
            pixels=model.Pixels(
                dimension_order='XYCZT',
                size_c=len(channels),
                size_t=1,
                size_x=data.shape[2],
                size_y=data.shape[1],
                type=dtype_name,
                big_endian=False,
                channels=[model.Channel(id=f'Channel:{i}', name=c) for i, c in enumerate(channels)],
                physical_size_x=microns_per_pixel,
                physical_size_y=microns_per_pixel,
                physical_size_x_unit='µm',
                physical_size_y_unit='µm'
            )
        )
    )

    im = o.images[0]
    for i in range(len(im.pixels.channels)):
        im.pixels.planes.append(model.Plane(the_c=i, the_t=0, the_z=0))
    im.pixels.tiff_data_blocks.append(model.TiffData(plane_count=len(im.pixels.planes)))

    with tifffile.TiffWriter(filepath, ome=True, bigtiff=True) as out_tif:
        opts = {
            'compression': 'LZW',
        }
        out_tif.write(
            rearrange(data, 'c y x -> 1 1 c y x'),
            **opts
        )
        xml_str = to_xml(o)
        out_tif.overwrite_description(xml_str.encode())

    with tifffile.TiffWriter(filepath, bigtiff=True) as tif:
        metadata={
            'axes': 'TCYXS',
            'Channel': {'Name': channels},
            'PhysicalSizeX': pix_per_micron,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pix_per_micron,
            'PhysicalSizeYUnit': 'µm',
        }
        tif.write(
            rearrange(data, 'c h w -> 1 c h w 1'),
            metadata=metadata,
            compression='LZW',
        )


def make_pseudo(channel_to_img, cmap=None, contrast_pct=20., contrast_mapping=None):
    cmap = sns.color_palette('tab10') if cmap is None else cmap

    new = np.zeros_like(next(iter(channel_to_img.values())))
    img_stack = []
    for i, (channel, img) in enumerate(channel_to_img.items()):
        color = cmap[i] if not isinstance(cmap, dict) else cmap[channel]
        new = img.copy().astype(np.float32)
        new -= new.min()
        new /= new.max()

        try:
            if contrast_mapping is not None:
                cp = contrast_mapping.get(channel, contrast_pct)
            else:
                cp = contrast_pct
            vmax = np.percentile(new[new>0], (cp)) if np.count_nonzero(new) else 1.
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


def get_common_channels(filepaths, channel_mapping=None):
    channel_mapping = channel_mapping if channel_mapping is not None else {}
    pool = []
    for filepath in filepaths:
        channels = get_ome_tiff_channels(filepath)
        channels = [channel_mapping.get(c, c) for c in channels]
        pool += channels
    counts = Counter(pool)
    channels = sorted([c for c, count in counts.items() if count==len(filepaths)])
    return channels

def get_channel_counts(filepaths, channel_mapping=None):
    channel_mapping = channel_mapping if channel_mapping is not None else {}
    pool = []
    for filepath in filepaths:
        channels = get_ome_tiff_channels(filepath)
        channels = [channel_mapping.get(c, c) for c in channels]
        pool += channels
    counts = Counter(pool)
    return counts.most_common()


def get_section_to_image(sid_to_filepaths, channels, channel_mapping=None, scale=.1, contrast_pct=None):
    if channel_mapping is None:
        channel_mapping = {}

    section_to_img = {}
    for sid, filepath in sid_to_filepaths.items():
        logging.info(f'generating image data for section {sid}')
        # cs, imgs = extract_ome_tiff(filepath, as_dict=False, scale=scale)
        cs, imgs = extract_ome_tiff(filepath, as_dict=False)
        imgs = np.stack([utils.rescale(x.astype(np.float32), scale=scale, dim_order='h w', target_dtype=np.float32) for x in imgs])
        # print(imgs[0].shape, np.unique(imgs[0]))
        cs = [channel_mapping.get(c, c) for c in cs]
        idxs = [cs.index(c) for c in channels]
        imgs = imgs[idxs]
        # imgs = imgs[idxs].astype(np.float32)
        # print(np.unique(imgs[0]))

        if contrast_pct is not None:
            for i, bw in enumerate(imgs):
                bw -= bw.min()
                bw /= bw.max()
                vmax = np.percentile(bw[bw>0], (contrast_pct), axis=-1) if np.count_nonzero(bw) else 1.
                imgs[i] = rescale_intensity(bw, in_range=(0., vmax))

        imgs = torch.tensor(imgs, dtype=torch.float32)
        
        section_to_img[sid] = imgs
    return section_to_img