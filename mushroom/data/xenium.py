import logging
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
import torch
import torchvision.transforms.functional as TF
import tifffile
from torchvision.transforms import Normalize

from mushroom.data.utils import LearnerData
import mushroom.data.visium as visium


def adata_from_xenium(filepath, scaler=.1, normalize=False):

    if filepath.split('.')[-1] == 'h5ad':
        adata = sc.read_h5ad(filepath)
    else:
        adata = sc.read_10x_h5(
            filename=os.path.join(filepath, 'cell_feature_matrix.h5')
        )

        df = pd.read_csv(
            os.path.join(filepath, 'cells.csv.gz')
        )
        df = df.set_index('cell_id')
        adata.obs = df.copy()
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy().astype(int)

        df = pd.read_csv(
            os.path.join(filepath, 'transcripts.csv.gz')
        )
        df = df.set_index('transcript_id')
        adata.uns['transcripts'] = df.copy()


        tf = tifffile.TiffFile(os.path.join(filepath, 'morphology_focus.ome.tif'))
        ppm = 1 / float(re.findall(r'PhysicalSizeX="(.*)".*PhysicalSizeY', tf.ome_metadata)[0])

        img = tifffile.imread(
        os.path.join(filepath, 'morphology_focus.ome.tif')).astype(np.float32)

        hires = TF.resize(
            torch.tensor(img).unsqueeze(0),
            (int(img.shape[0] * scaler), int(img.shape[1] * scaler)),
            antialias=True
        ).squeeze().numpy()

        if hires.max() > 255.: # check for uint16
            hires /= 65535.
            hires *= 255
            hires = hires.astype(np.uint8)

        adata.uns['spatial'] = {
            'key': {
                'images': {'hires': hires},
                'scalefactors': {
                    'tissue_hires_scalef': scaler * ppm,
                    'spot_diameter_fullres': 10.
                }
            }
        }

    # if sparse, then convert
    if 'sparse' in str(type(adata.X)).lower():
        adata.X = adata.X.toarray()

    if normalize:
        # sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    return adata


def get_common_channels(filepaths, channel_mapping=None):
    channel_mapping = channel_mapping if channel_mapping is not None else {}
    pool = []
    for filepath in filepaths:
        adata = adata_from_xenium(filepath)

        channels = adata.var.index.to_list()
        channels = [channel_mapping.get(c, c) for c in channels]
        pool += channels
    counts = Counter(pool)
    channels = sorted([c for c, count in counts.items() if count==len(filepaths)])

    return channels


def to_multiplex(adata, tiling_size=64):
    size = visium.get_fullres_size(adata)
    n_rows, n_cols = size[-2] // tiling_size + 1, size[-1] // tiling_size + 1
    pts = adata.obsm['spatial'][:, [1, 0]]

    img = np.zeros((n_rows, n_cols, adata.shape[1]))
    for r in range(n_rows):
        r1, r2 = r * tiling_size, (r + 1) * tiling_size
        row_mask = ((pts[:, 0] >= r1) & (pts[:, 0] < r2))
        row_adata, row_pts = adata[row_mask], pts[row_mask]
        for c in range(n_cols):
            c1, c2 = c * tiling_size, (c + 1) * tiling_size
            col_mask = ((row_pts[:, 1] >= c1) & (row_pts[:, 1] < c2))
            img[r, c] = row_adata[col_mask].X.sum(0)
    return img


def get_learner_data(
        config, scale, size, patch_size,
        channels=None, channel_mapping=None, fullres_size=None,
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
        channels = get_common_channels(
            fps, channel_mapping=channel_mapping
        )
    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info(f'processing sections')
    section_to_adata = {sid:adata_from_xenium(fp, normalize=True) for sid, fp in sid_to_filepaths.items()}
    section_to_img, section_to_adata = visium.get_section_to_image( # labeled image where pixels represent location of cells, is converted by transform to actual exp image
        section_to_adata, channels, patch_size=patch_size, channel_mapping=channel_mapping, scale=scale, fullres_size=fullres_size
    )

    # TODO: find a cleaner way to do this, is long because trying to avoid explicit sparse matrix conversion of .X
    means = np.asarray(np.vstack(
        [a.X.mean(0) for a in section_to_adata.values()]
    ).mean(0)).squeeze()
    stds = np.asarray(np.vstack(
        [a.X.std(0) for a in section_to_adata.values()]
    ).mean(0)).squeeze()
    normalize = Normalize(means, stds)


    train_transform = visium.VisiumTrainingTransform(size=size, patch_size=patch_size, normalize=normalize)
    inference_transform = visium.VisiumInferenceTransform(size=size, patch_size=patch_size, normalize=normalize)

    logging.info('generating training dataset')
    train_ds = visium.VisiumSectionDataset(
        section_ids, section_to_adata, section_to_img, transform=train_transform
    )
    logging.info('generating inference dataset')
    inference_ds = visium.VisiumInferenceSectionDataset(
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