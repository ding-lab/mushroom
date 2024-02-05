import json
import os
import re
from collections import Counter

import anndata
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import torch
import torchvision.transforms.functional as TF
import tifffile

import mushroom.data.visium as visium
import mushroom.utils as utils

def get_fullres_size(adata):
    return visium.get_fullres_size(adata)

def display_fovs(cosmx_dir):
    flatfiles_dir = os.path.join(cosmx_dir, 'flatfiles')
    flat_fps = sorted(utils.listfiles(flatfiles_dir))
    fov_metadata_fp = [filepath for filepath in flat_fps if 'fov_positions_file.csv.gz' in filepath][0]

    fov_metadata = pd.read_csv(fov_metadata_fp)

    sns.scatterplot(data=fov_metadata, x='X_mm', y='Y_mm')
    plt.gca().invert_yaxis()
    plt.axis('equal')

def adata_from_cosmx(filepath, img_channel='DNA', scaler=.1, normalize=False, sample_to_bbox=None):
    if filepath.split('.')[-1] == 'h5ad':
        sample_to_adata = {'sample': sc.read_h5ad(filepath)}
    else:
        if sample_to_bbox is None:
            sample_to_bbox['sample'] = (0, 1000, 0, 1000) # just make arbitrarily large
        print('start')
        fov_fps = sorted(utils.listfiles(filepath, regex=r'Morphology2D.*TIF$'))
        fov_to_fp = {int(re.sub(r'^.*F0*([1-9]+[0-9]*).TIF$', r'\1', fp)):fp for fp in fov_fps}
        print(fov_to_fp)

        metadata_fp = list(utils.listfiles(filepath, regex=r'metadata_file.csv.gz$'))[0]
        exp_fp = list(utils.listfiles(filepath, regex=r'exprMat_file.csv.gz$'))[0]
        fov_metadata_fp = list(utils.listfiles(filepath, regex=r'fov_positions_file.csv.gz$'))[0]

        print('here')

        metadata = pd.read_csv(metadata_fp, index_col='cell')
        fov_metadata = pd.read_csv(fov_metadata_fp)

        exp_df = pd.read_csv(exp_fp, index_col='cell')
        to_remove = ['fov', 'cell_ID', 'Negative', 'SystemControl']
        exp_df = exp_df[[c for c in exp_df.columns
                    if not len([x for x in to_remove if x in c])]]
        exp_df = exp_df.loc[metadata.index.to_list()]

        tf = tifffile.TiffFile(fov_fps[0])
        img_metadata = json.loads(next(iter(tf.pages)).description)

        img_ppm = img_metadata['PixelSize_um']
        channel_ids = img_metadata['ChannelOrder']
        channel_id_to_name = {d['Fluorophore']['ChannelId']:d['BiologicalTarget']
                            for d in img_metadata['MorphologyKit']['MorphologyReagents']}
        channels = [channel_id_to_name[cid] for cid in channel_ids]
        channel_idx = channels.index(img_channel)


        tile_size = (len(channel_ids), img_metadata['ImRows'], img_metadata['ImCols'])
        
        sample_to_adata = {}
        for sample, (r1, r2, c1, c2) in sample_to_bbox.items():
            print(sample)
            fov_mask = (
                (fov_metadata['X_mm'] >= c1) &
                (fov_metadata['X_mm'] <= c2) &
                (fov_metadata['Y_mm'] >= r1) &
                (fov_metadata['Y_mm'] <= r2)
            )
            fov_meta = fov_metadata[fov_mask]
            x_vals, y_vals = sorted(set(fov_meta['X_mm'].to_list())), sorted(set(fov_meta['Y_mm'].to_list()))
            fov_to_pos = {i:(y_vals.index(y), x_vals.index(x)) for i, y, x in zip(fov_meta['FOV'], fov_meta['Y_mm'], fov_meta['X_mm'])} 

            cell_mask = [True if fov in fov_to_pos else False for fov in metadata['fov'].to_list()]
            cell_meta = metadata[cell_mask]
            cell_exp = exp_df[cell_mask]

            stitched = np.zeros((tile_size[0], tile_size[1] * len(y_vals), tile_size[2] * len(x_vals)), dtype=np.uint8)
            cell_to_xy = {}
            for fov, (y, x) in fov_to_pos.items():
                print(fov)
                f = metadata[metadata['fov']==fov]

                tile = tifffile.imread(fov_to_fp[fov])
                tile = np.flip(tile, axis=1) # for some reason images are inverted on y-axis in the tif


                r1, r2 = y * tile.shape[-2], (y + 1) * tile.shape[-2]
                c1, c2 = x * tile.shape[-1], (x + 1) * tile.shape[-1]

                tile = tile.astype(np.float32)
                tile /= 65535.
                tile *= 255.
                tile = tile.astype(np.uint8)

                stitched[:, r1:r2, c1:c2] = tile

                ys = np.abs(f['CenterY_local_px'] - tile_size[-2])
                for cell_id, x, y in zip(f.index.to_list(), f['CenterX_local_px'], ys):
                    # cell_to_xy[cell_id] = ((x + c1) * scaler, (y + r1) * scaler)
                    cell_to_xy[cell_id] = (x + c1, y + r1)

            resized = TF.resize(
                torch.tensor(stitched),
                (int(stitched.shape[-2] * scaler), int(stitched.shape[-1] * scaler)),
                antialias=True
            ).numpy()

            img_dict = {'hires': resized[channel_idx]}
            img_dict.update({f'hires_{k}':v for k, v in zip(channels, resized)})

            adata = anndata.AnnData(X=cell_exp.values, obs=cell_meta)
            adata.var.index = cell_exp.columns
            adata.var.index.name = 'gene'

            adata.obsm['spatial'] = np.stack((
                np.asarray([cell_to_xy[cid][0] for cid in adata.obs.index.to_list()]),
                np.asarray([cell_to_xy[cid][1] for cid in adata.obs.index.to_list()]),
            )).swapaxes(1, 0)

            sfs = {'spot_diameter_fullres': 10.}
            sfs.update({f'tissue_{k}_scalef':scaler for k in img_dict})

            adata.uns['spatial'] = {
                sample: {
                    'images': img_dict,
                    'scalefactors': sfs
                }
            }

            adata.uns['ppm'] = img_ppm
        
            sample_to_adata[sample] = adata
    
    for sid, adata in sample_to_adata.items():
        # if sparse, then convert
        if 'sparse' in str(type(adata.X)).lower():
            adata.X = adata.X.toarray()

        if normalize:
            sc.pp.log1p(adata)

        sample_to_adata[sid] = adata
    
    if len(sample_to_adata) <= 1:
        return next(iter(sample_to_adata.values()))
    return sample_to_adata


def get_common_channels(filepaths, channel_mapping=None):
    channel_mapping = channel_mapping if channel_mapping is not None else {}
    pool = []
    for filepath in filepaths:
        adata = adata_from_cosmx(filepath)

        channels = adata.var.index.to_list()
        channels = [channel_mapping.get(c, c) for c in channels]
        pool += channels
    counts = Counter(pool)
    channels = sorted([c for c, count in counts.items() if count==len(filepaths)])

    return channels


def to_multiplex(adata, tiling_size=64, method='grid', radius_sf=1.):
    return visium.to_multiplex(adata, tiling_size=tiling_size, method=method, radius_sf=radius_sf)
