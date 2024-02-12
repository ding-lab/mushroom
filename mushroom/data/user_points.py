from collections import Counter

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

import mushroom.data.visium as visium

CELL_COL = 'cell_id'
X_COL = 'x'
Y_COL = 'y'
X_FULLRES = 'fullres_width'
Y_FULLRES = 'fullres_height'

def adata_from_point_based(df, normalize=False, base=np.e, hires_scale=.2):
    adata = None
    if isinstance(df, str):
        ext = df.split('/')[-1].split('.')[-1]
        if ext == 'h5ad':
            adata = sc.read_h5ad(df)
        elif ext == 'csv':
            df = pd.read_csv(df)
        elif ext == 'gz':
            df = pd.read_csv(df)
        else:
            raise RuntimeError(f'If supplying filepath to datatable extension must be .csv')

    assert X_COL in df.columns, f'x coordinates must be specfied under the column name "{X_COL}"'
    assert Y_COL in df.columns, f'y coordinates must be specfied under the column name "{Y_COL}"'
    assert X_FULLRES in df.columns, f'width of full-resolution image/data plane must be specfied under the column name "{X_FULLRES}"'
    assert Y_FULLRES in df.columns, f'height of full-resolution image/data plane must be specfied under the column name "{Y_FULLRES}"'
    assert CELL_COL in df.columns and len(set(df[CELL_COL]))==df.shape[0], f'cell ids must be unique and specfied under the column name "{CELL_COL}"'
    
    if adata is None:
        coords = df[[X_COL, Y_COL]].values.astype(int)

        exp = df[[c for c in df.columns if c not in [CELL_COL, X_COL, Y_COL, X_FULLRES, Y_FULLRES]]]

        adata = anndata.AnnData(X=exp.values)
        adata.obs.index = df[CELL_COL]
        adata.var.index = exp.columns
        adata.obsm['spatial'] = coords
        adata.uns['ppm'] = 1. # assumes unit is microns

        # we need to make a placeholder image so downstream size calculation is correct
        h, w = df[X_FULLRES][0] * hires_scale, df[Y_FULLRES][0] * hires_scale
        h, w = int(h), int(w)
        adata.uns['spatial'] = {
            'key': {
                'images': {'hires': np.zeros((h, w), dtype=np.uint8)},
                'scalefactors': {'tissue_hires_scalef': hires_scale}
            }
        }

    # if sparse, then convert
    if 'sparse' in str(type(adata.X)).lower():
        adata.X = adata.X.toarray()

    if normalize:
        sc.pp.log1p(adata, base=base)
    
    return adata


def get_common_channels(filepaths, channel_mapping=None, pct_expression=.02):
    channel_mapping = channel_mapping if channel_mapping is not None else {}
    pool = []
    for filepath in filepaths:
        adata = adata_from_point_based(filepath)

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

def to_multiplex(adata, tiling_size=64, method='grid', radius_sf=1.):
    return visium.to_multiplex(adata, tiling_size=tiling_size, method=method, radius_sf=radius_sf)
