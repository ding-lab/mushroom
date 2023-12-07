import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Normalize, RandomCrop, Compose

import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
import mushroom.data.xenium as xenium
import mushroom.utils as utils

DTYPES = ('multiplex', 'xenium', 'visium',)



def get_multiplex_section_to_img(config, ppm, target_ppm, channels=None, channel_mapping=None, contrast_pct=None):
    sid_to_filepaths = {
        entry['id']:d['filepath'] for entry in config for d in entry['data']
        if d['dtype']=='multiplex'
    }

    section_ids = [entry['id'] for entry in config
                   if 'multiplex' in [d['dtype'] for d in entry['data']]]

    if channels is None:
        fps = [d['filepath'] for entry in config for d in entry['data']
                if d['dtype']=='multiplex']
        channels = multiplex.get_common_channels(fps, channel_mapping=channel_mapping)
    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info('processing sections')
    section_to_img = multiplex.get_section_to_image(
        sid_to_filepaths, channels, channel_mapping=channel_mapping, scale=target_ppm / ppm, contrast_pct=contrast_pct)
    
    means = torch.cat(
        [x.mean(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    stds = torch.cat(
        [x.std(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    normalize = Normalize(means, stds)
    
    return section_to_img, normalize

def get_xenium_section_to_img(
        config, ppm, target_ppm, channels=None, channel_mapping=None
    ):
    sid_to_filepaths = {
        entry['id']:d['filepath'] for entry in config for d in entry['data']
        if d['dtype']=='xenium'
    }

    section_ids = [entry['id'] for entry in config
                   if 'xenium' in [d['dtype'] for d in entry['data']]]

    if channels is None:
        fps = [d['filepath'] for entry in config for d in entry['data']
                if d['dtype']=='xenium']
        channels = xenium.get_common_channels(
            fps, channel_mapping=channel_mapping
        )
    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info(f'processing sections')
    tiling_size = int(ppm / target_ppm)
    section_to_adata = {
        sid:xenium.adata_from_xenium(fp, normalize=True)
        for sid, fp in sid_to_filepaths.items()
    }
    section_to_img = {
        sid:xenium.to_multiplex(adata, tiling_size=tiling_size)
        for sid, adata in sid_to_filepaths.items()
    }
    # reorder channel dim
    section_to_img = {k:rearrange(img, 'h w c -> c h w') for k, img in section_to_img.items()}

    means = torch.cat(
        [x.mean(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    stds = torch.cat(
        [x.std(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    normalize = Normalize(means, stds)

    return section_to_img, section_to_adata, normalize


def get_visium_section_to_img(
        config, target_ppm, channels=None, channel_mapping=None, pct_expression=.02
    ):
    sid_to_filepaths = {
        entry['id']:d['filepath'] for entry in config for d in entry['data']
        if d['dtype']=='visium'
    }
    if not len(sid_to_filepaths):
        return None, None, None
    section_ids = [entry['id'] for entry in config
                   if 'visium' in [d['dtype'] for d in entry['data']]]

    if channels is None:
        fps = [d['filepath'] for entry in config for d in entry['data']
                if d['dtype']=='visium']
        channels = visium.get_common_channels(
            fps, channel_mapping=channel_mapping, pct_expression=pct_expression
        )
    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info(f'processing sections')
    section_to_adata = {sid:visium.adata_from_visium(fp, normalize=True) for sid, fp in sid_to_filepaths.items()}
    section_to_img, section_to_adata = visium.get_section_to_image( # labeled image where pixels represent location of barcodes, is converted by transform to actual exp image
        section_to_adata, channels, target_ppm=target_ppm, patch_size=1, channel_mapping=channel_mapping
    )

     # TODO: find a cleaner way to do this, is long because trying to avoid explicit sparse matrix conversion of .X
    means = np.asarray(np.vstack(
        [a.X.mean(0) for a in section_to_adata.values()]
    ).mean(0)).squeeze()
    stds = np.asarray(np.vstack(
        [a.X.std(0) for a in section_to_adata.values()]
    ).mean(0)).squeeze()
    normalize = Normalize(means, stds)

    return section_to_img, section_to_adata, normalize



def get_learner_data(config, ppm, target_ppm, tile_size, channels=None, channel_mapping=None, contrast_pct=None, pct_expression=.02):

    # all images must be same size
    dtypes = sorted({d['dtype'] for entry in config for d in entry['data']})
    section_ids = [entry['id'] for entry in config]

    dtype_to_section_to_img = {}
    dtype_to_norm = {}
    dtype_to_section_to_adata = {}

    for dtype in dtypes:
        if dtype == 'multiplex':
            section_to_img, norm = get_multiplex_section_to_img(config, ppm, target_ppm, channels=channels, channel_mapping=channel_mapping, contrast_pct=contrast_pct)
            section_to_adata = None
        elif dtype == 'xenium':
            section_to_img, section_to_adata, norm = get_xenium_section_to_img(config, ppm, target_ppm, channels=channels, channel_mapping=channel_mapping)
        elif dtype == 'visium':
            section_to_img, section_to_adata, norm = get_visium_section_to_img(config, target_ppm, channels=channels, channel_mapping=channel_mapping, pct_expression=pct_expression)
        else:
            raise RuntimeError(f'dtype {dtype} is not a valid data type')
        
        dtype_to_section_to_img[dtype] = section_to_img
        dtype_to_norm[dtype] = norm
        dtype_to_section_to_adata[dtype] = section_to_adata

    # image sizes are a few pixels off sometimes, adjusting for that
    sizes = [(img.shape[-2], img.shape[-1]) for section_to_img in dtype_to_section_to_img.values() for img in section_to_img.values()]
    idx = np.argmax([np.sum(x) for x in sizes])
    target_size = sizes[idx]
    for dtype, section_to_img in dtype_to_section_to_img.items():
        section_to_img = {sid:utils.rescale(img, size=target_size, dim_order='c h w', target_dtype=img.dtype) for sid, img in section_to_img.items()}
    assert len({(img.shape[-2], img.shape[-1]) for section_to_img in dtype_to_section_to_img.values() for img in section_to_img.values()}) == 1


    train_transform = ImageTrainingTransform(target_size[-2], target_size[-1], dtype_to_norm, size=(tile_size, tile_size))
    inference_transform = ImageInferenceTransform(target_size[-2], target_size[-1], dtype_to_norm, size=(tile_size, tile_size))
# sections, dtypes, dtype_to_section_to_imgs, transform=None, n=None
    logging.info('generating training dataset')
    train_ds = ImageTrainingDataset(
        section_ids, dtypes, dtype_to_section_to_img, transform=train_transform
    )
    logging.info('generating inference dataset')
    inference_ds = ImageInferenceDataset(
        section_ids, dtypes, dtype_to_section_to_img, transform=inference_transform
    )
    
    learner_data = LearnerData(
        dtype_to_section_to_img=dtype_to_section_to_img,
        train_transform=train_transform,
        inference_transform=inference_transform,
        train_ds=train_ds,
        inference_ds=inference_ds,
        channels=channels,
        dtypes=dtypes
    )

    return learner_data


class ImageTrainingTransform(object):
    def __init__(self, h, w, dtype_to_norm, size=(8, 8)):
        self.h, self.w = h, w
        self.output_size = size
        self.hs = np.arange(self.h - size[-2] - 1)
        self.ws = np.arange(self.w - size[-1] - 1)
        self.dtype_to_norm = dtype_to_norm

    def __call__(self, imgs, dtypes):
        h, w = np.random.choice(self.hs), np.random.choice(self.ws)

        tiles = [TF.crop(img, h, w, self.output_size[-2], self.output_size[-1]) for img in imgs]
        if np.random.rand() > .5:
            tiles = [TF.vflip(img) for img in tiles]
        if np.random.rand() > .5:
            tiles = [TF.hflip(img) for img in tiles]

        tiles = [self.dtype_to_norm[dtype](img) for dtype, img in zip(dtypes, tiles)]

        return tiles
    
class ImageInferenceTransform(object):
    def __init__(self, h, w, dtype_to_norm, size=(8, 8)):
        self.h, self.w = h, w
        self.output_size = size
        self.hs = np.arange(self.h - size[-2] - 1)
        self.ws = np.arange(self.w - size[-1] - 1)
        self.dtype_to_norm = dtype_to_norm

    def __call__(self, tile, dtype):
        return self.dtype_to_norm[dtype](tile)
    
class ImageTrainingDataset(Dataset):
    def __init__(self, sections, dtypes, dtype_to_section_to_imgs, transform=None, n=None):
        self.section_to_img = {(sid, dtype):img if len(img.shape)==3 else img.unsqueeze(0) for dtype, d in dtype_to_section_to_imgs.items() for sid, img in dtype_to_section_to_imgs.keys()}
        section_to_keys = {sid:(sid, dtype) for sid, dtype in self.section_to_img.keys()}
        self.section_ids = []
        for section in sections:
            self.section_ids += section_to_keys[section]
        self.dtypes = dtypes

        self.size = next(iter(self.section_to_img.values())).shape[-2:]

        self.transform = transform if transform is not None else ImageTrainingTransform(*self.size)

        self.n = np.iinfo(np.int64).max if n is None else n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        anchor_section = np.random.choice(self.section_ids)
        anchor_dtype = anchor_section[1]
        anchor_idx = self.section_ids.index(anchor_section)

        if anchor_idx == 0:
            pos_idx = 1
        elif anchor_idx == len(self.section_ids) - 1:
            pos_idx = anchor_idx - 1
        else:
            pos_idx = np.random.choice([anchor_idx - 1, anchor_idx + 1])
        pos_section = self.section_ids[pos_idx]
        pos_dtype = pos_section[1]

        anchor_tile, pos_tile = self.transform(
            [self.section_to_img[anchor_section], self.section_to_img[pos_section]],
            [anchor_dtype, pos_dtype]
        )

        anchor_dtype_idx = self.dtypes.index(anchor_dtype)
        pos_dtype_idx = self.dtypes.index(pos_dtype)

        return {
            'anchor_slide_idx': anchor_idx,
            'pos_slide_idx': pos_idx,
            'anchor_tile': anchor_tile,
            'pos_tile': pos_tile,
            'anchor_dtype_idx': anchor_dtype_idx,
            'pos_dtype_idx': pos_dtype_idx,
        }


class ImageInferenceDataset(Dataset):
    def __init__(self, sections, dtypes, dtype_to_section_to_imgs, tile_size=(8, 8), transform=None):
        """"""
        self.section_to_img = {(sid, dtype):img if len(img.shape)==3 else img.unsqueeze(0) for dtype, d in dtype_to_section_to_imgs.items() for sid, img in dtype_to_section_to_imgs.keys()}
        section_to_keys = {sid:(sid, dtype) for sid, dtype in self.section_to_img.keys()}
        self.section_ids = []
        for section in sections:
            self.section_ids += section_to_keys[section]
        self.dtypes = dtypes

        self.tile_size = tile_size
        self.size = next(iter(self.section_to_img.values())).shape[-2:]
        
        # tiles are (ph pw c h w)
        self.section_to_tiles = {s:self.to_tiles(x) for s, x in self.section_to_img.items()}
        self.pw, self.ph = self.section_to_tiles[self.section_ids[0]].shape[:2]
        
        self.n_tiles_per_image = self.pw * self.ph
        outs = torch.stack(torch.meshgrid(
            torch.arange(len(self.section_ids)),
            torch.arange(self.section_to_tiles[self.section_ids[0]].shape[0]),
            torch.arange(self.section_to_tiles[self.section_ids[0]].shape[1]),
            indexing='ij'
        ))
        self.idx_to_coord = rearrange(
            outs, 'b n_sections n_rows n_cols -> (n_sections n_rows n_cols) b')

        self.transform = transform if transform is not None else nn.Identity()
        
    def to_tiles(self, x, tile_size=None):
        tile_size = self.tile_size if tile_size is None else tile_size
        pad_h, pad_w = tile_size[-2] - x.shape[-2] % tile_size[-2], tile_size[-1] - x.shape[-1] % tile_size[-1]
        # left, top, right and bottom
        x = TF.pad(x, [pad_w // 2, pad_h // 2, pad_w // 2 + pad_w % 2, pad_h // 2 + pad_h % 2])
        x = x.unfold(-2, tile_size[-2], tile_size[-2] // 2)
        x = x.unfold(-2, tile_size[-1], tile_size[-1] // 2)

        x = rearrange(x, 'c ph pw h w -> ph pw c h w')

        return x

    def image_from_tiles(self, x):
        pad_h, pad_w = x.shape[-2] // 4, x.shape[-1] // 4
        x = x[..., pad_h:-pad_h, pad_w:-pad_w]
        return rearrange(x, 'ph pw c h w -> c (ph h) (pw w)')
    
    def section_from_tiles(self, x, section_idx, tile_size=None):
        """
        x - (n c h w)
        """
        tile_size = self.tile_size if tile_size is None else tile_size
        mask = self.idx_to_coord[:, 0]==section_idx
        tiles = x[mask]
        ph, pw = self.idx_to_coord[mask, 1].max() + 1, self.idx_to_coord[mask, 2].max() + 1
        
        out = torch.zeros(ph, pw, x.shape[1], tile_size[0], tile_size[1])
        for idx, (_, r, c) in enumerate(self.idx_to_coord[mask]):
            out[r, c] = tiles[idx]
        
        return self.image_from_tiles(out)

    def __len__(self):
        return self.idx_to_coord.shape[0]

    def __getitem__(self, idx):
        section_idx, row_idx, col_idx = self.idx_to_coord[idx]
        section = self.section_ids[section_idx]
        dtype = section[1]
        dtype_idx = self.dtypes.index(dtype)
        return {
            'idx': section_idx,
            'row_idx': row_idx,
            'col_idx': col_idx,
            'dtype_idx': dtype_idx,
            'tile': self.transform(self.section_to_tiles[section][row_idx, col_idx], dtype)
        }
    

@dataclass
class LearnerData:
    dtype_to_section_to_img: dict
    train_transform: ImageTrainingTransform
    inference_transform: ImageInferenceTransform
    train_ds: ImageTrainingDataset
    inference_ds: ImageInferenceDataset
    channels: Iterable
    dtypes: Iterable