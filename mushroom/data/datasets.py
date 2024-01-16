import logging
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Normalize, RandomCrop, Compose

import mushroom.data.he as he
import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
import mushroom.data.xenium as xenium
import mushroom.utils as utils

DTYPES = ('multiplex', 'xenium', 'visium', 'he',)


def get_config_info(config, name):
    assert name in DTYPES, f'data type must be one of {DTYPES}, got {name}'
    sid_to_filepaths = {
        entry['id']:d['filepath'] for entry in config for d in entry['data']
        if d['dtype']==name
    }

    section_ids = [entry['id'] for entry in config
                   if name in [d['dtype'] for d in entry['data']]]
    
    fps = [d['filepath'] for entry in config for d in entry['data']
                if d['dtype']==name]
    
    return sid_to_filepaths, section_ids, fps

def generate_norm_transform(section_to_img):
    means = torch.cat(
        [x.mean(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    stds = torch.cat(
        [x.std(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    normalize = Normalize(means, stds)
    return normalize



def get_multiplex_section_to_img(config, ppm, target_ppm, channels=None, channel_mapping=None, contrast_pct=None):
    logging.info(f'starting multiplex processing')
    sid_to_filepaths, section_ids, fps = get_config_info(config, 'multiplex')

    if channels is None:
        channels = multiplex.get_common_channels(fps, channel_mapping=channel_mapping)

    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info('processing sections')
    section_to_img = multiplex.get_section_to_image(
        sid_to_filepaths, channels, channel_mapping=channel_mapping, scale=target_ppm / ppm, contrast_pct=contrast_pct)
    
    normalize = generate_norm_transform(section_to_img)
    
    return section_to_img, normalize, channels

def get_he_section_to_img(config, ppm, target_ppm):
    logging.info(f'starting he processing')
    sid_to_filepaths, section_ids, fps = get_config_info(config, 'he')

    # channels = np.asarray(['red', 'green', 'blue'])
    channels = ['red', 'green', 'blue']

    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info('processing sections')
    section_to_img = he.get_section_to_image(
        sid_to_filepaths, scale=target_ppm / ppm)
    
    normalize = generate_norm_transform(section_to_img)
    
    return section_to_img, normalize, channels

def get_xenium_section_to_img(
        config, ppm, target_ppm, channels=None, channel_mapping=None
    ):
    logging.info(f'starting xenium processing')
    sid_to_filepaths, section_ids, fps = get_config_info(config, 'xenium')

    if channels is None:
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

    section_to_img = {}
    for sid, adata in section_to_adata.items():
        logging.info(f'generating image data for section {sid}')
        img = xenium.to_multiplex(adata, tiling_size=tiling_size)
        img = torch.tensor(rearrange(img, 'h w c -> c h w'), dtype=torch.float32)
        section_to_img[sid] = img
   
    normalize = generate_norm_transform(section_to_img)

    return section_to_img, section_to_adata, normalize, channels

def get_visium_section_to_img(
        config, ppm, target_ppm, channels=None, channel_mapping=None, pct_expression=.02,
    ):
    logging.info(f'starting visium processing')
    sid_to_filepaths, section_ids, fps = get_config_info(config, 'visium')

    if channels is None:
        channels = visium.get_common_channels(
            fps, channel_mapping=channel_mapping, pct_expression=pct_expression
        )

    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info(f'processing sections')
    tiling_size = int(ppm / target_ppm)
    section_to_adata = {
        sid:visium.adata_from_visium(fp, normalize=True)
        for sid, fp in sid_to_filepaths.items()
    }

    section_to_adata = {sid:adata[:, channels] for sid, adata in section_to_adata.items()}

    section_to_img = {}
    for sid, adata in section_to_adata.items():
        logging.info(f'generating image data for section {sid}')
        img = visium.to_multiplex(adata, tiling_size=tiling_size)
        img = torch.tensor(rearrange(img, 'h w c -> c h w'), dtype=torch.float32)
        section_to_img[sid] = img
   
    normalize = generate_norm_transform(section_to_img)

    return section_to_img, section_to_adata, normalize, channels
# def get_visium_section_to_img(
#         config, target_ppm, channels=None, channel_mapping=None, pct_expression=.02
#     ):
#     logging.info(f'starting visium processing')
#     sid_to_filepaths = {
#         entry['id']:d['filepath'] for entry in config for d in entry['data']
#         if d['dtype']=='visium'
#     }
#     if not len(sid_to_filepaths):
#         return None, None, None
#     section_ids = [entry['id'] for entry in config
#                    if 'visium' in [d['dtype'] for d in entry['data']]]

#     if channels is None:
#         fps = [d['filepath'] for entry in config for d in entry['data']
#                 if d['dtype']=='visium']
#         channels = visium.get_common_channels(
#             fps, channel_mapping=channel_mapping, pct_expression=pct_expression
#         )
#     logging.info(f'using {len(channels)} channels')
#     logging.info(f'{len(section_ids)} sections detected: {section_ids}')

#     logging.info(f'processing sections')
#     section_to_adata = {sid:visium.adata_from_visium(fp, normalize=True) for sid, fp in sid_to_filepaths.items()}
#     section_to_img, section_to_adata = visium.get_section_to_image( # labeled image where pixels represent location of barcodes, is converted by transform to actual exp image
#         section_to_adata, channels, target_ppm=target_ppm, patch_size=1, channel_mapping=channel_mapping
#     )

#      # TODO: find a cleaner way to do this, is long because trying to avoid explicit sparse matrix conversion of .X
#     means = np.asarray(np.vstack(
#         [a.X.mean(0) for a in section_to_adata.values()]
#     ).mean(0)).squeeze()
#     stds = np.asarray(np.vstack(
#         [a.X.std(0) for a in section_to_adata.values()]
#     ).mean(0)).squeeze()
#     normalize = Normalize(means, stds)

#     return section_to_img, section_to_adata, normalize, channels


def get_learner_data(config, ppm, target_ppm, tile_size, channel_mapping=None, contrast_pct=None, pct_expression=.02):

    # all images must be same size
    dtypes = sorted({d['dtype'] for entry in config for d in entry['data']})
    section_ids = [entry['id'] for entry in config]

    dtype_to_section_to_img = {}
    dtype_to_norm = {}
    dtype_to_section_to_adata = {}
    dtype_to_channels = {}

    for dtype in dtypes:
        if dtype == 'multiplex':
            section_to_img, norm, channels = get_multiplex_section_to_img(config, ppm, target_ppm, channel_mapping=channel_mapping, contrast_pct=contrast_pct)
            section_to_adata = None
        elif dtype == 'he':
            section_to_img, norm, channels = get_he_section_to_img(config, ppm, target_ppm)
            section_to_adata = None
        elif dtype == 'xenium':
            section_to_img, section_to_adata, norm, channels = get_xenium_section_to_img(config, ppm, target_ppm, channel_mapping=None)
        elif dtype == 'visium':
            section_to_img, section_to_adata, norm, channels = get_visium_section_to_img(config, ppm, target_ppm, channel_mapping=None, pct_expression=pct_expression)
        else:
            raise RuntimeError(f'dtype {dtype} is not a valid data type')
        
        dtype_to_section_to_img[dtype] = section_to_img
        dtype_to_norm[dtype] = norm
        dtype_to_section_to_adata[dtype] = section_to_adata
        dtype_to_channels[dtype] = channels

    # image sizes are a few pixels off sometimes, adjusting for that
    sizes = [(img.shape[-2], img.shape[-1]) for section_to_img in dtype_to_section_to_img.values() for img in section_to_img.values()]
    idx = np.argmax([np.sum(x) for x in sizes])
    target_size = sizes[idx]
    for dtype, section_to_img in dtype_to_section_to_img.items():
        section_to_img = {sid:utils.rescale(img, size=target_size, dim_order='c h w', target_dtype=img.dtype) for sid, img in section_to_img.items()}
        dtype_to_section_to_img[dtype] = section_to_img
    assert len(set((img.shape[-2], img.shape[-1]) for section_to_img in dtype_to_section_to_img.values() for img in section_to_img.values())) == 1


    train_transform = ImageTrainingTransform(target_size[-2], target_size[-1], dtype_to_norm, size=(tile_size, tile_size))
    inference_transform = ImageInferenceTransform(target_size[-2], target_size[-1], dtype_to_norm, size=(tile_size, tile_size))
    logging.info('generating training dataset')
    train_ds = ImageTrainingDataset(
        section_ids, dtypes, dtype_to_section_to_img, transform=train_transform
    )
    logging.info('generating inference dataset')
    inference_ds = ImageInferenceDataset(
        section_ids, dtypes, dtype_to_section_to_img, transform=inference_transform
    )

    logging.info(f'total of {len(train_ds.section_ids)} sections detected: {train_ds.section_ids}')
    
    dtype_to_n_channels = {dtype:len(channels) for dtype, channels in dtype_to_channels.items()}
    
    learner_data = LearnerData(
        dtype_to_section_to_img=dtype_to_section_to_img,
        train_transform=train_transform,
        inference_transform=inference_transform,
        train_ds=train_ds,
        inference_ds=inference_ds,
        dtypes=dtypes,
        dtype_to_n_channels=dtype_to_n_channels,
        dtype_to_channels=dtype_to_channels,
    )

    return learner_data

def construct_training_batch(batch):
    dtype_pool = sorted(set([entry['anchor_dtype_idx'] for entry in batch]).union([entry['pos_dtype_idx'] for entry in batch]))
    to_return = {
        'tiles': {x:[] for x in dtype_pool},
        'slides': {x:[] for x in dtype_pool},
        'dtypes': {x:[] for x in dtype_pool},
        'pairs': {x:[] for x in dtype_pool},
        'is_anchor': {x:[] for x in dtype_pool},
    }

    for i, entry in enumerate(batch):
        dtype_idx = entry['anchor_dtype_idx']
        to_return['tiles'][dtype_idx].append(entry['anchor_tile'])
        to_return['slides'][dtype_idx].append(entry['anchor_slide_idx'])
        to_return['dtypes'][dtype_idx].append(entry['anchor_dtype_idx'])
        to_return['pairs'][dtype_idx].append(i)
        to_return['is_anchor'][dtype_idx].append(True)

        dtype_idx = entry['pos_dtype_idx']
        to_return['tiles'][dtype_idx].append(entry['pos_tile'])
        to_return['slides'][dtype_idx].append(entry['pos_slide_idx'])
        to_return['dtypes'][dtype_idx].append(entry['pos_dtype_idx'])
        to_return['pairs'][dtype_idx].append(i)
        to_return['is_anchor'][dtype_idx].append(False)

    to_return['tiles'] = [torch.stack(to_return['tiles'][k]) for k in dtype_pool]
    to_return['slides'] = [torch.tensor(to_return['slides'][k], dtype=torch.long) for k in dtype_pool]
    to_return['dtypes'] = [torch.tensor(to_return['dtypes'][k], dtype=torch.long) for k in dtype_pool]
    to_return['pairs'] = [torch.tensor(to_return['pairs'][k], dtype=torch.long) for k in dtype_pool]
    to_return['is_anchor'] = [torch.tensor(to_return['is_anchor'][k], dtype=torch.bool) for k in dtype_pool]

    return to_return

def construct_inference_batch(batch):
    dtype_pool = sorted(set([entry['dtype_idx'] for entry in batch]))
    to_return = {
        'tiles': {x:[] for x in dtype_pool},
        'slides': {x:[] for x in dtype_pool},
        'dtypes': {x:[] for x in dtype_pool},
    }

    for i, entry in enumerate(batch):
        dtype_idx = entry['dtype_idx']
        to_return['tiles'][dtype_idx].append(entry['tile'])
        to_return['slides'][dtype_idx].append(entry['idx'])
        to_return['dtypes'][dtype_idx].append(entry['dtype_idx'])

    to_return['tiles'] = [torch.stack(to_return['tiles'][k]) for k in dtype_pool]
    to_return['slides'] = [torch.tensor(to_return['slides'][k], dtype=torch.long) for k in dtype_pool]
    to_return['dtypes'] = [torch.tensor(to_return['dtypes'][k], dtype=torch.long) for k in dtype_pool]  

    return to_return


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
        self.section_to_img = {}
        for dtype, d in dtype_to_section_to_imgs.items():
            for sid, img in d.items():
                img = img if len(img.shape)==3 else img.unsqueeze(0)
                self.section_to_img[(sid, dtype)] = img
        self.section_ids = []
        for section in sections:
            self.section_ids += [(s, dtype) for s, dtype in self.section_to_img.keys() if s==section]
        self.section_idxs = np.arange(len(self.section_ids))
        self.dtypes = dtypes

        self.size = next(iter(self.section_to_img.values())).shape[-2:]

        self.transform = transform if transform is not None else ImageTrainingTransform(*self.size)

        self.n = np.iinfo(np.int64).max if n is None else n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        anchor_idx = np.random.choice(self.section_idxs)
        anchor_section = self.section_ids[anchor_idx]
        anchor_dtype = anchor_section[1]

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
    
    def display_batch(self, batch, idx, dtype_to_channels, display_channels={'xenium': 'EPCAM', 'multiplex': 'E-Cadherin'}):
        items = []
        for position, dtype in enumerate(self.dtypes):
            if idx in batch['pairs'][position]:
                mask = batch['pairs'][position]==idx
                items.append({k:v[position][mask][0] for k, v in batch.items()})

        fig, axs = plt.subplots(ncols=len(items))
        for ax, item in zip(axs, items):
            dtype = self.dtypes[item['dtypes']]
            slide = item['slides']
            is_anchor = item['is_anchor']
            channel = display_channels[dtype]
            ax.imshow(item['tiles'][dtype_to_channels[dtype].index(channel)])
            ax.axis('off')
            ax.set_title(f'{dtype} {slide} {is_anchor} {channel}')


class ImageInferenceDataset(Dataset):
    def __init__(self, sections, dtypes, dtype_to_section_to_imgs, tile_size=(8, 8), transform=None):
        """"""
        self.section_to_img = {}
        for dtype, d in dtype_to_section_to_imgs.items():
            for sid, img in d.items():
                img = img if len(img.shape)==3 else img.unsqueeze(0)
                self.section_to_img[(sid, dtype)] = img
        self.section_ids = []
        for section in sections:
            self.section_ids += [(s, dtype) for s, dtype in self.section_to_img.keys() if s==section]

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
        x - (n h w c) or (n hw c)
        """
        tile_size = self.tile_size if tile_size is None else tile_size
        mask = self.idx_to_coord[:, 0]==section_idx

        if isinstance(x, list):
            tiles = torch.stack([val for val, keep in zip(x, mask) if keep])
        else:
            tiles = x[mask]

        if len(tiles.shape) == 3:
            tiles = rearrange(tiles, 'n (h w) c -> n h w c', h=tile_size[0], w=tile_size[1])
        
        ph, pw = self.idx_to_coord[mask, 1].max() + 1, self.idx_to_coord[mask, 2].max() + 1
        
        out = torch.zeros(ph, pw, tile_size[0], tile_size[1], tiles.shape[-1], dtype=tiles.dtype)
        for idx, (_, r, c) in enumerate(self.idx_to_coord[mask]):
            out[r, c] = tiles[idx]
        
        out = rearrange(out, 'ph pw h w c -> ph pw c h w')
        return rearrange(self.image_from_tiles(out), 'c h w -> h w c')

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
    dtype_to_section_to_img: Mapping
    train_transform: ImageTrainingTransform
    inference_transform: ImageInferenceTransform
    train_ds: ImageTrainingDataset
    inference_ds: ImageInferenceDataset
    dtypes: Iterable
    dtype_to_n_channels: Mapping
    dtype_to_channels: Mapping