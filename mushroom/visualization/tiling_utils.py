
import pandas as pd
import numpy as np

import mushroom.data.multiplex as multiplex
import mushroom.data.xenium as xenium
import mushroom.data.cosmx as cosmx
import mushroom.data.visium as visium
import mushroom.utils as utils


from einops import rearrange

def tile_xenium(adata, target_size=None, tile_size=20):
    if target_size is None:
        target_size = xenium.get_fullres_size(adata)
    
    adata.obs['grid_name'] = [f'{x // tile_size}_{y // tile_size}' for x, y in adata.obsm['spatial']]
    df = pd.DataFrame(data=adata.X, columns=adata.var.index.to_list(), index=adata.obs.index.to_list())
    df['grid_name'] = adata.obs['grid_name'].to_list()
    df = df.groupby('grid_name').sum()
    
    img = np.zeros((target_size[0] // tile_size + 1, target_size[1] // tile_size + 1, df.shape[1]))
    for name, row in df.iterrows():
        x, y = [int(x) for x in name.split('_')]
        img[y, x] = row.values
    return img


def get_tiled_sections(config, dtype='multiplex', channel_names=None, tiling_size=20, drop=None, target_size=None):
    sections = [x for x in config['sections'] if x['data'][0]['dtype']==dtype]
    
    if drop is not None:
        sections = [x for x in sections if x['sid'] not in drop]
    
    fps = [x['data'][0]['filepath'] for x in sections]
    
    if dtype == 'multiplex':
        channels = multiplex.get_common_channels(fps)
    elif dtype == 'xenium':
        channels = xenium.get_common_channels(fps)
    elif dtype == 'visium':
        channels = visium.get_common_channels(fps)
    elif dtype == 'cosmx':
        channels = cosmx.get_common_channels(fps)
    else:
        raise ValueError(f'{dtype} is not valid dtype')
        
    if channel_names is not None:
        present = [channel for channel in channel_names if channel in channels]
        assert len(present) == len(channel_names), f'{set(channel_names) - set(present)} not in all images'
        channels = channel_names
    imgs = []
    for fp in fps:
        if dtype == 'multiplex':
            channel_to_img = multiplex.extract_ome_tiff(fp, channels=channels, as_dict=True)
            img = np.stack([channel_to_img[c] for c in channels])
        elif dtype == 'xenium':
            adata = xenium.adata_from_xenium(fp, normalize=True)
            adata = adata[:, channels]
#             img = xenium.to_multiplex(adata, tiling_size=tiling_size, method='grid')
            img = tile_xenium(adata, tile_size=tiling_size)
            img = rearrange(img, 'h w c -> c h w')
        elif dtype == 'visium':
            adata = visium.adata_from_visium(fp, normalize=True)
            adata = adata[:, channels]
            img = visium.to_multiplex(adata, tiling_size=tiling_size, method='radius')
            img = rearrange(img, 'h w c -> c h w')
        elif dtype == 'cosmx':
            adata = cosmx.adata_from_cosmx(fp, normalize=True)
            adata = adata[:, channels]
#             img = xenium.to_multiplex(adata, tiling_size=tiling_size, method='grid')
            img = tile_xenium(adata, tile_size=tiling_size)
            img = rearrange(img, 'h w c -> c h w')

        if target_size is not None and img.shape[-2:] != target_size:
            img = utils.rescale(img, size=target_size, target_dtype=img.dtype, dim_order='c h w')

        imgs.append(img)
    x = np.stack(imgs)
    
    if x.dtype != np.uint8:
        x = x - x.min((0, 2, 3), keepdims=True)
        x = x / x.max((0, 2, 3), keepdims=True)
        x = x * 255
        x = x.astype(np.uint8)
    
    return x