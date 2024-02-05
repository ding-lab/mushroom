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

def get_fullres_size(adata):
    return visium.get_fullres_size(adata)

def adata_from_xenium(filepath, scaler=.1, normalize=False, transcripts=False):

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

        if transcripts:
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
                    'spot_diameter_fullres': 10.,
                }
            }
        }

        adata.uns['ppm'] = 1. # coords are in microns

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


def to_multiplex(adata, tiling_size=64, method='grid', radius_sf=1.):
    return visium.to_multiplex(adata, tiling_size=tiling_size, method=method, radius_sf=radius_sf)
