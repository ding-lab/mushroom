import logging
from collections import Counter

import anndata
import numpy as np
import torch
import torch.nn as nn
import scanpy as sc
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, Compose, Normalize

from mushroom.data.utils import LearnerData
from mushroom.data.inference import InferenceSectionDataset

def pixels_per_micron(adata):
    if isinstance(adata, str):
        adata = adata_from_visium(adata)
    scalefactors = next(iter(adata.uns['spatial'].values()))['scalefactors']
    return scalefactors['spot_diameter_fullres'] / 65. # each spot is 65 microns wide

def get_fullres_size(adata):
    d = next(iter(adata.uns['spatial'].values()))
    img = d['images']['hires']
    fullres_size = (
        img.shape[0] / d['scalefactors']['tissue_hires_scalef'],
        img.shape[1] / d['scalefactors']['tissue_hires_scalef']
    )
    return fullres_size

def adata_from_visium(filepath):
    ext = filepath.split('.')[-1]
    if ext == 'h5ad':
        adata = sc.read_h5ad(filepath)
    else:
        adata = sc.read_visium(filepath)

    adata.var_names_make_unique()

    # some versions of scanpy don't load in as ints
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(int)

    # if sparse, then convert
    if 'sparse' in str(type(adata.X)).lower():
        adata.X = adata.X.toarray()
    
    return adata


def format_expression(tiles, adatas, patch_size):
    # add batch dim if there is none
    if len(tiles.shape) == 2:
        tiles = tiles.unsqueeze(0)
    if isinstance(adatas, anndata.AnnData):
        adatas = [adatas]
    
    exp_imgs = []
    for tile, adata in zip(tiles, adatas):
        # tile = rearrange(tile, '(ph h) (pw w) -> h w (ph pw)', ph=patch_size, pw=patch_size)
        tile = tile.unfold(-2, patch_size, patch_size)
        tile = tile.unfold(-2, patch_size, patch_size)
        tile = rearrange(tile, 'h w h1 w1 -> h w (h1 w1)')
        x = torch.unique(tile, dim=-1)

        exp = torch.zeros(x.shape[0], x.shape[1], adata.shape[1], dtype=torch.float32)
        l2b = adata.uns['label_to_barcode']
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                labels = x[i, j]
                labels = labels[labels!=0]
                if len(labels):
                    barcodes = [l2b[l.item()] for l in labels]
                    exp[i, j] = torch.tensor(adata[barcodes].X.mean(0))
        exp = rearrange(exp, 'h w c -> c h w')
        exp_imgs.append(exp)
    
    return torch.stack(exp_imgs)


def get_common_channels(filepaths, channel_mapping=None, pct_expression=.02):
    channel_mapping = channel_mapping if channel_mapping is not None else {}
    pool = []
    for filepath in filepaths:
        adata = adata_from_visium(filepath)

        if pct_expression is not None:
            spot_count = (adata.X>0).sum(0)
            mask = spot_count > pct_expression * adata.shape[0]
            adata = adata[:, mask]

        channels = adata.var.index.to_list()
        channels = [channel_mapping.get(c, c) for c in channels]
        pool += channels
    counts = Counter(pool)
    channels = sorted([c for c, count in counts.items() if count==len(filepaths)])

    return channels


def get_section_to_image(
        section_to_adata,
        channels,
        channel_mapping=None,
        scale=.1,
        log=True,
        fullres_size=None
    ):
    if channel_mapping is None:
        channel_mapping = {}

    section_to_img = {}
    for i, (sid, adata) in enumerate(section_to_adata.items()):
        # filter genes/channels
        adata = adata[:, channels]

        # scale coords
        adata.obsm['spatial_scaled'] = (adata.obsm['spatial'] * scale).astype(np.int32)

        # assign each barcode an integer label > 0
        adata.uns['label_to_barcode'] = {i+1:x for i, x in enumerate(adata.obs.index)}
        adata.uns['barcode_to_label'] = {v:k for k, v in adata.uns['label_to_barcode'].items()}

        # logp1
        if log:
            sc.pp.log1p(adata)

        # create labeled image
        if fullres_size is None and i==0:
            fullres_size = get_fullres_size(adata)
    
        scaled_size = (
            int(fullres_size[0] * scale),
            int(fullres_size[1] * scale),
        )

        labeled_locations = torch.zeros(*scaled_size, dtype=torch.long)
        for barcode, (c, r) in zip(adata.obs.index, adata.obsm['spatial_scaled']):
            labeled_locations[r, c] = adata.uns['barcode_to_label'][barcode]
        
        section_to_img[sid] = labeled_locations.unsqueeze(0)
        section_to_adata[sid] = adata
    return section_to_img, section_to_adata


def get_learner_data(
        config, scale, size, patch_size,
        channels=None, channel_mapping=None, fullres_size=None, pct_expression=.02
    ):
    sid_to_filepaths = {
        entry['id']:d['filepath'] for entry in config for d in entry['data']
        if d['dtype']=='visium'
    }
    section_ids = [entry['id'] for entry in config
                   if 'visium' in [d['dtype'] for d in entry['data']]]

    if channels is None:
        fps = [d['filepath'] for entry in config for d in entry['data']
                if d['dtype']=='visium']
        channels = get_common_channels(
            fps, channel_mapping=channel_mapping, pct_expression=pct_expression
        )
    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info(f'processing sections')
    section_to_adata = {sid:adata_from_visium(fp) for sid, fp in sid_to_filepaths.items()}
    section_to_img, section_to_adata = get_section_to_image( # labeled image where pixels represent location of barcodes, is converted by transform to actual exp image
        section_to_adata, channels, channel_mapping=channel_mapping, scale=scale, fullres_size=fullres_size
    )

    # TODO: find a cleaner way to do this, is long because trying to avoid explicit sparse matrix conversion of .X
    means = np.asarray(np.vstack(
        [a.X.mean(0) for a in section_to_adata.values()]
    ).mean(0)).squeeze()
    stds = np.asarray(np.vstack(
        [a.X.std(0) for a in section_to_adata.values()]
    ).mean(0)).squeeze()

    normalize = Normalize(means, stds)
    train_transform = VisiumTrainingTransform(size=size, patch_size=patch_size, normalize=normalize)
    inference_transform = VisiumInferenceTransform(size=size, patch_size=patch_size, normalize=normalize)

    logging.info('generating training dataset')
    train_ds = VisiumSectionDataset(
        section_ids, section_to_adata, section_to_img, transform=train_transform
    )
    logging.info('generating inference dataset')
    inference_ds = VisiumInferenceSectionDataset(
        section_ids, section_to_img, section_to_adata, transform=inference_transform, size=size
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


class VisiumTrainingTransform(object):
    def __init__(self, size=(256, 256), patch_size=32, normalize=None):
        self.size = size
        self.patch_size = patch_size
        self.output_size = (self.size[0] // self.patch_size, self.size[1] // self.patch_size)
        self.output_patch_size = 1
        self.transforms = Compose([
            RandomCrop(size, padding_mode='reflect'),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ])

        self.normalize = normalize if normalize is not None else nn.Identity()

    def __call__(self, anchor, pos, neg, anchor_adata, pos_adata, neg_adata):
        anchor, pos = self.transforms(torch.stack((anchor, pos)))
        neg = self.transforms(neg.unsqueeze(0)).squeeze()

        anchor = format_expression(anchor, anchor_adata, patch_size=self.patch_size)
        pos = format_expression(pos, pos_adata, patch_size=self.patch_size)
        neg = format_expression(neg, neg_adata, patch_size=self.patch_size)

        anchor, pos, neg = self.normalize(anchor), self.normalize(pos), self.normalize(neg)
        anchor, pos, neg = anchor.squeeze(0), pos.squeeze(0), neg.squeeze(0)
        return torch.stack((anchor, pos, neg))
        
    
class VisiumInferenceTransform(object):
    def __init__(self, size=(256, 256), patch_size=32, normalize=None):
        self.size = size
        self.patch_size = patch_size
        self.normalize = normalize if normalize is not None else nn.Identity()

    def __call__(self, x, adata):
        tile = x.squeeze()
        tile = format_expression(tile, adata, patch_size=self.patch_size)
        tile = self.normalize(tile)
        return tile.squeeze(0)
    
class VisiumSectionDataset(Dataset):
    def __init__(self, sections, section_to_adata, section_to_img, transform=None):
        self.sections = sections
        self.section_to_adata = section_to_adata
        self.section_to_img = section_to_img

        self.transform = transform if transform is not None else nn.Identity()

    def __len__(self):
        return np.iinfo(np.int64).max # make infinite

    def __getitem__(self, idx):
        anchor_section = np.random.choice(self.sections)
        anchor_idx = self.sections.index(anchor_section)

        if anchor_idx == 0:
            pos_idx = 1
        elif anchor_idx == len(self.sections) - 1:
            pos_idx = anchor_idx - 1
        else:
            pos_idx = np.random.choice([anchor_idx - 1, anchor_idx + 1])
        pos_section = self.sections[pos_idx]

        neg_section = np.random.choice(self.sections)
        neg_idx = self.sections.index(anchor_section)

        anchor, pos, neg = [
            self.section_to_img[section] for section in [anchor_section, pos_section, neg_section]
        ]
        anchor_adata, pos_adata, neg_adata = [
            self.section_to_adata[section] for section in [anchor_section, pos_section, neg_section]
        ]
        anchor_img, pos_img, neg_img = self.transform(
            anchor, pos, neg, anchor_adata, pos_adata, neg_adata
        )

        return {
            'anchor_idx': anchor_idx,
            'pos_idx': pos_idx,
            'neg_idx': neg_idx,
            'anchor_img': anchor_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
        }
    
class VisiumInferenceSectionDataset(InferenceSectionDataset):
    def __init__(
            self, sections, section_to_img, section_to_adata,
            size=(256, 256), transform=None
        ):
        super().__init__(sections, section_to_img, size=size, transform=transform)
        self.section_to_adata = section_to_adata

    def image_from_tiles(self, x, to_expression=False, adata=None):
        pad_h, pad_w = x.shape[-2] // 4, x.shape[-1] // 4
        x = x[..., pad_h:-pad_h, pad_w:-pad_w]
        
        if to_expression:
            ps = self.transform.patch_size
            new = torch.zeros(x.shape[0], x.shape[1], adata.shape[1], x.shape[-2] // ps, x.shape[-1] // ps,
                             dtype=torch.float32)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    new[i, j] = format_expression(x[i, j, 0], adata, ps)
            x = new
        x = rearrange(x, 'ph pw c h w -> c (ph h) (pw w)')
        return x

    def __getitem__(self, idx):
        section_idx, row_idx, col_idx = self.idx_to_coord[idx]
        section = self.sections[section_idx]
        img = self.section_to_tiles[section][row_idx, col_idx]
        adata = self.section_to_adata[section]
        return {
            'section_idx': section_idx,
            'row_idx': row_idx,
            'col_idx': col_idx,
            'img': self.transform(img, adata)
        }