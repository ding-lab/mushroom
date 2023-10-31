import logging
from collections import Counter

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import tifffile
from einops import rearrange
from ome_types import from_xml
from skimage.exposure import rescale_intensity
from tifffile import TiffFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Normalize, RandomCrop, Compose
from vit_pytorch import ViT

from mushroom.data.inference import InferenceTransform, InferenceSectionDataset
from mushroom.data.utils import LearnerData


def pixels_per_micron(filepath):
    tif = TiffFile(filepath)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    return im.pixels.physical_size_x

def extract_ome_tiff(filepath, channels=None, as_dict=True, flexibility='strict', scale=None):
    tif = TiffFile(filepath)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    d = {}
    img_channels, imgs = [], []

    for c, p in zip(im.pixels.channels, tif.pages):
        if channels is None or c.name in channels:
            img = p.asarray()
            d[c.name] = img

            if scale is not None:
                img = torch.tensor(img).unsqueeze(0)
                target_size = [int(x * scale) for x in img.shape[-2:]]
                img = TF.resize(img, size=target_size, antialias=True).squeeze()

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


def write_basic_ome_tiff(filepath, data, channels, pix_per_micron=1.):
    """
    data - (n_channels, height, width)
    """
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


def get_section_to_image(sid_to_filepaths, channels, channel_mapping=None, scale=.1, contrast_pct=95.):
    if channel_mapping is None:
        channel_mapping = {}

    section_to_img = {}
    for sid, filepath in sid_to_filepaths.items():
        cs, imgs = extract_ome_tiff(filepath, as_dict=False, scale=scale)
        cs = [channel_mapping.get(c, c) for c in cs]
        idxs = [cs.index(c) for c in channels]
        imgs = imgs[idxs].astype(np.float32)

        if contrast_pct is not None:
            for i, bw in enumerate(imgs):
                bw -= bw.min()
                bw /= bw.max()
                vmax = np.percentile(bw[bw>0], (contrast_pct), axis=-1) if np.count_nonzero(bw) else 1.
                imgs[i] = rescale_intensity(bw, in_range=(0., vmax))

        imgs = torch.tensor(imgs)
        
        section_to_img[sid] = imgs
    return section_to_img


def get_learner_data(config, scale, size, patch_size, channels=None, channel_mapping=None, contrast_pct=None):
    sid_to_filepaths = {
        entry['id']:d['filepath'] for entry in config for d in entry['data']
        if d['dtype']=='multiplex'
    }
    section_ids = [entry['id'] for entry in config
                   if 'multiplex' in [d['dtype'] for d in entry['data']]]

    if channels is None:
        fps = [d['filepath'] for entry in config for d in entry['data']
                if d['dtype']=='multiplex']
        channels = get_common_channels(fps, channel_mapping=channel_mapping)
    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info('processing sections')
    section_to_img = get_section_to_image(
        sid_to_filepaths, channels, channel_mapping=channel_mapping, scale=scale, contrast_pct=contrast_pct)

    means = torch.cat(
        [x.mean(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    stds = torch.cat(
        [x.std(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    normalize = Normalize(means, stds)

    train_transform = MultiplexTrainingTransform(size=size, patch_size=patch_size, normalize=normalize)
    inference_transform = InferenceTransform(normalize)

    logging.info('generating training dataset')
    train_ds = MultiplexSectionDataset(
        section_ids, section_to_img, transform=train_transform
    )
    logging.info('generating inference dataset')
    inference_ds = InferenceSectionDataset(
        section_ids, section_to_img, transform=inference_transform, size=size
    )

    learner_data = LearnerData(
        section_to_img=section_to_img,
        train_transform=train_transform,
        inference_transform=inference_transform,
        train_ds=train_ds,
        inference_ds=inference_ds,
        channels=channels
    )

    return learner_data


class MultiplexTrainingTransform(object):
    def __init__(self, size=(256, 256), patch_size=32, normalize=None):
        self.output_size = size
        self.output_patch_size = patch_size
        self.transforms = Compose([
            RandomCrop(size, padding_mode='reflect'),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            normalize if normalize is not None else nn.Identity()
        ])

    def __call__(self, tile):
        return self.transforms(tile)
    

class MultiplexSectionDataset(Dataset):
    def __init__(self, sections, section_to_img, transform=None):
        self.sections = sections
        self.section_to_img = section_to_img

        self.transform = transform if transform is not None else nn.Identity()

    def __len__(self):
        return np.iinfo(np.int64).max # make infinite

    def __getitem__(self, idx):
        section = np.random.choice(self.sections)
        idx = self.sections.index(section)

        tile = self.transform(self.section_to_img[section])

        return {
            'idx': idx,
            'tile': tile,
        }