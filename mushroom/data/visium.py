import logging
from collections import Counter

import anndata
import numpy as np
import torch
import torch.nn as nn
import scanpy as sc
import squidpy as sq
from einops import rearrange
from sklearn.neighbors import NearestNeighbors
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
        int(img.shape[0] / d['scalefactors']['tissue_hires_scalef']),
        int(img.shape[1] / d['scalefactors']['tissue_hires_scalef'])
    )
    return fullres_size

def adata_from_visium(filepath, normalize=False, base=np.e):
    ext = filepath.split('.')[-1]
    if ext == 'h5ad':
        adata = sc.read_h5ad(filepath)
    else:
        # adata = sc.read_visium(filepath)
        adata = sq.read.visium(filepath)

    adata.var_names_make_unique()

    # some versions of scanpy don't load in as ints
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(int)

    # if sparse, then convert
    if 'sparse' in str(type(adata.X)).lower():
        adata.X = adata.X.toarray()

    if normalize:
        sc.pp.log1p(adata, base=base)

    adata.uns['ppm'] = pixels_per_micron(adata)
    
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
        spots = adata.obs.index.to_list()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                labels = x[i, j]
                labels = labels[labels!=0]
                if len(labels):
                    barcodes = {l2b[l.item()] for l in labels}
                    mask = [True if x in barcodes else False for x in spots]
                    exp[i, j] = torch.tensor(adata.X[mask].mean(0))

        exp = rearrange(exp, 'h w c -> c h w')


        exp_imgs.append(exp)
    
    return torch.stack(exp_imgs).squeeze(0)


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


def to_multiplex(adata, tiling_size=64, method='radius', radius_sf=1.):
    size = get_fullres_size(adata)
    n_rows, n_cols = size[-2] // tiling_size + 1, size[-1] // tiling_size + 1

    pts = adata.obsm['spatial'][:, [1, 0]]

    if method == 'radius':
        grid_pts = np.meshgrid(np.arange(n_rows), np.arange(n_cols)) # (2, n_rows, n_cols)
        grid_pts = rearrange(grid_pts, 'b h w -> (h w) b')
        
        nbhs = NearestNeighbors(radius=tiling_size * radius_sf)
        nbhs.fit(pts)
        transformed = grid_pts * tiling_size + tiling_size / 2
        dists, idxs = nbhs.radius_neighbors(transformed)
        dists = 1 - (dists / tiling_size)

        X = adata.X
        minimum = X.min(0)

        img = np.zeros((n_rows, n_cols, adata.shape[1]))
        for (r, c), distances, indices in zip(grid_pts, dists, idxs):
            if len(distances):
                vals = X[indices] * rearrange(distances, 'd -> d 1')
                img[r, c] = vals.sum(0)
            else:
                img[r, c] = minimum
    elif method == 'grid':
        img = np.zeros((n_rows, n_cols, adata.shape[1]))
        for r in range(n_rows):
            r1, r2 = r * tiling_size, (r + 1) * tiling_size
            row_mask = ((pts[:, 0] >= r1) & (pts[:, 0] < r2))
            row_adata, row_pts = adata[row_mask], pts[row_mask]
            for c in range(n_cols):
                c1, c2 = c * tiling_size, (c + 1) * tiling_size
                col_mask = ((row_pts[:, 1] >= c1) & (row_pts[:, 1] < c2))
                img[r, c] = row_adata[col_mask].X.sum(0)
    else:
        raise RuntimeError(f'method was {method}, can only be "grid" or "radius"')

    return img


def get_section_to_image(
        section_to_adata,
        channels,
        patch_size=1,
        channel_mapping=None,
        scale=.1,
        fullres_size=None
    ):
    if channel_mapping is None:
        channel_mapping = {}

    section_to_img = {}
    for i, (sid, adata) in enumerate(section_to_adata.items()):
        logging.info(f'generating image data for section {sid}')
        # filter genes/channels
        adata = adata[:, channels]

        # scale coords
        adata.obsm['spatial_scaled'] = (adata.obsm['spatial'] * scale).astype(np.int32)

        # assign each barcode an integer label > 0
        adata.uns['label_to_barcode'] = {i+1:x for i, x in enumerate(adata.obs.index)}
        adata.uns['barcode_to_label'] = {v:k for k, v in adata.uns['label_to_barcode'].items()}

        # create labeled image
        if fullres_size is None and i==0:
            fullres_size = get_fullres_size(adata)
    
        scaled_size = (
            int(fullres_size[0] * scale) + 1,
            int(fullres_size[1] * scale) + 1,
        )

        labeled_locations = torch.zeros(*scaled_size, dtype=torch.long)
        for barcode, (c, r) in zip(adata.obs.index, adata.obsm['spatial_scaled']):
            labeled_locations[r, c] = adata.uns['barcode_to_label'][barcode]
        labeled_locations = labeled_locations.unsqueeze(0)

        # exp = format_expression(labeled_locations, adata, patch_size)
        
        section_to_img[sid] = labeled_locations
        section_to_adata[sid] = adata

    return section_to_img, section_to_adata
