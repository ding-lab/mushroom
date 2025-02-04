import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
import tifffile
from einops import rearrange
from scipy import ndimage as ndi
from skimage.exposure import adjust_gamma

import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
import mushroom.data.xenium as xenium
import mushroom.data.cosmx as cosmx
import mushroom.data.he as he
import mushroom.utils as utils


def read_bigwarp_warp_field(fp, downsample_scaler):
    """
    Read bigwarp 
    """
    ddf = torch.tensor(tifffile.imread(fp))
    ddf = ddf[[1, 0]] # needs to be (h, w, c), bigwarp exports (w, h, c)

    # rescale to original size
    scale = 1 / downsample_scaler

    ddf = TF.resize(ddf, (int(ddf.shape[-2] * scale), int(ddf.shape[-1] * scale)), antialias=False)
    ddf *= scale

    return ddf


def warp_image(moving, ddf, fill_type='min'):
    """
    assumes 2d transform
    
    moving - (c h w)
    fixed - (c h w)
    ddf - (2 h w) # first channel is h displacment, second channel is w displacement
    """
    ref_grid_h, ref_grid_w = torch.meshgrid(
        torch.arange(ddf.shape[-2]),
        torch.arange(ddf.shape[-1]),
        indexing='ij',
    )

    h_idxs = torch.round(ref_grid_h + ddf[-2])
    w_idxs = torch.round(ref_grid_w + ddf[-1])

    mask = torch.zeros_like(ddf, dtype=torch.bool)
    mask[-2] = (h_idxs >= 0) & (h_idxs < moving.shape[-2])
    mask[-1] = (w_idxs >= 0) & (w_idxs < moving.shape[-1])

    masked_ddf = ddf * mask        
    h_idxs = torch.round(ref_grid_h + masked_ddf[-2]).to(torch.long)
    w_idxs = torch.round(ref_grid_w + masked_ddf[-1]).to(torch.long)

    h_idxs[h_idxs>= moving.shape[-2]] = 0 
    w_idxs[w_idxs>= moving.shape[-1]] = 0

    warped = moving[..., h_idxs, w_idxs]

    if fill_type == 'min':
        warped[..., mask.sum(0)<2] = 0
    else:
        warped[..., mask.sum(0)<2] = moving.max()
    
    return warped


def is_valid(pt, size):
    r, c = pt
    return (r >= 0) & (r < size[-2]) & (c >= 0) & (c < size[-1])


def warp_pts(pts, ddf, radius=1):
    """
    DEPRECIATED: use warp_pts_fast

    assumes 2d transform
    
    pts - (n, 2) # 2 is height, width
    ddf - (2 h w) # first channel is h displacment, second channel is w displacement
    """
    if not isinstance(pts, torch.Tensor):
        pts = torch.tensor(pts)
    pts = pts.to(torch.long)
    max_r, max_c = pts.max(dim=0).values
    img = torch.zeros((max_r + 1, max_c + 1), dtype=torch.long)
    for i, (r, c) in enumerate(pts):
        r1, r2 = max(0, r - radius), min(max_r + 1, r + radius)
        c1, c2 = max(0, c - radius), min(max_c + 1, c + radius)
        img[r1:r2, c1:c2] = i + 1

    img = warp_image(img, ddf)

    objects = ndi.find_objects(img.numpy())
    label_to_warped_pt = {} 
    for i, obj in enumerate(objects):
        if obj is None:
            continue

        r, c = obj[0].start, obj[1].start
        if r != max_r + 1 and r != 0: r += 1
        if c != max_c + 1 and c != 0: c += 1

        label_to_warped_pt[i] = (r, c)

    idxs = torch.arange(pts.shape[0], dtype=torch.long)
    size = (ddf.shape[-2], ddf.shape[-1])
    mask = torch.tensor(
        [True if i.item() in label_to_warped_pt and is_valid(label_to_warped_pt[i.item()], size) else False
        for i in idxs], dtype=torch.bool)
    idxs = idxs[mask]
    warped = torch.tensor([list(label_to_warped_pt[i.item()]) for i in idxs], dtype=torch.long)

    return warped, mask

def warp_pts_fast(pts, ddf, upscale_factor=1., bin_res=5000):
    """pts - (y, x)"""
    is_torch = False
    if isinstance(pts, torch.Tensor):
        pts = pts.numpy()
        is_torch = True
    if isinstance(ddf, torch.Tensor):
        ddf = ddf.numpy()
    
    ref_grid_h, ref_grid_w = np.meshgrid(
        np.arange(ddf.shape[-2]),
        np.arange(ddf.shape[-1]),
        indexing='ij',
    )

    h_idxs = ref_grid_h + ddf[-2]
    w_idxs = ref_grid_w + ddf[-1]

    h_idxs = utils.rescale(h_idxs, scale=upscale_factor, dim_order='h w', target_dtype=h_idxs.dtype)
    w_idxs = utils.rescale(w_idxs, scale=upscale_factor, dim_order='h w', target_dtype=h_idxs.dtype)
    
    h_min, h_max = h_idxs.min(), h_idxs.max()
    w_min, w_max = w_idxs.min(), w_idxs.max()

    h_bins = np.linspace(h_min, h_max, bin_res)
    w_bins = np.linspace(w_min, w_max, bin_res)

    h_binned = np.digitize(h_idxs, bins=h_bins) - 1
    w_binned = np.digitize(w_idxs, bins=w_bins) - 1

    meshes = np.stack(np.meshgrid(np.arange(h_binned.shape[0]), np.arange(h_binned.shape[1]), indexing='ij'))
    mapping = np.full((2, bin_res, bin_res), -1, dtype=int)
    mapping[:, h_binned.flatten(), w_binned.flatten()] = meshes[0].flatten(), meshes[1].flatten()

    y_coord_binned = np.digitize(pts[:, 0], bins=h_bins) - 1
    x_coord_binned = np.digitize(pts[:, 1], bins=w_bins) - 1

    warped = mapping[:, y_coord_binned, x_coord_binned]
    exclude = np.sum(warped==-1, axis=0) > 0
    exclude |= np.sum(warped==bin_res - 1, axis=0) > 0
    np.count_nonzero(exclude)

    warped = warped[:, ~exclude]
    warped = warped / upscale_factor

    if is_torch:
        warped = torch.tensor(warped)
        exclude = torch.tensor(exclude)

    return warped.T, ~exclude

# def register_visium(adata, ddf, target_pix_per_micron=1., moving_pix_per_micron=None):
def register_visium(to_transform, ddf, resolution=None):
    adata = to_transform.copy()

    d = next(iter(adata.uns['spatial'].values()))
    scalefactors = d['scalefactors']

    hires, lowres = torch.tensor(d['images']['hires']), torch.tensor(d['images']['lowres'])

    hires_scale = 1 / scalefactors['tissue_hires_scalef']
    hires = TF.resize(
        rearrange(hires, 'h w c -> c h w'), (int(hires_scale * hires.shape[0]), int(hires_scale * hires.shape[1])), antialias=True,
    )
    lowres_scale = 1 / scalefactors['tissue_lowres_scalef']
    lowres = TF.resize(
        rearrange(lowres, 'h w c -> c h w'), (int(lowres_scale * lowres.shape[0]), int(lowres_scale * lowres.shape[1])), antialias=True,
    )

    warped_hires = warp_image(hires, ddf, fill_type='max')
    scaled_warped_hires = TF.resize(warped_hires, (int(scalefactors['tissue_hires_scalef'] * warped_hires.shape[-2]), int(scalefactors['tissue_hires_scalef'] * warped_hires.shape[-1])), antialias=True)
    scaled_warped_hires = rearrange(scaled_warped_hires, 'c h w -> h w c').numpy()
    d['images']['hires'] = scaled_warped_hires / scaled_warped_hires.max() # numpy conversion has slight overflow issue

    warped_lowres = warp_image(lowres, ddf, fill_type='max')
    scaled_warped_lowres = TF.resize(warped_lowres, (int(scalefactors['tissue_lowres_scalef'] * warped_lowres.shape[-2]), int(scalefactors['tissue_lowres_scalef'] * warped_lowres.shape[-1])), antialias=True)
    scaled_warped_lowres = rearrange(scaled_warped_lowres, 'c h w -> h w c').numpy()
    d['images']['lowres'] = scaled_warped_lowres / scaled_warped_lowres.max() # numpy conversion has slight overflow issue

    # warped_lowres = rearrange(warp_image(lowres, ddf), 'c h w -> h w c').numpy()
    # d['images']['lowres'] = warped_lowres / warped_lowres.max() # numpy conversion has slight overflow issue

    adata.obsm['spatial_original'] = adata.obsm['spatial'].copy()
    # x = (torch.tensor(new.obsm['spatial']) * scale).to(torch.long)
    x = torch.tensor(adata.obsm['spatial']).to(torch.long)
    x = x[:, [1, 0]] # needs to be (h, w) instead of (w, h)
    transformed, mask = warp_pts_fast(x, ddf)
    adata = adata[mask.numpy()]
    adata.obsm['spatial'] = transformed[:, [1, 0]].numpy()

    if resolution is not None:
        adata.uns['ppm'] = resolution

    return adata

def register_cosmx(adata, ddf, resolution=None, radius=10):
    return register_xenium(adata, ddf, resolution=resolution, radius=radius)

def register_xenium(adata, ddf, resolution=None, radius=1, coordinate_sf=None):
    new = adata.copy()

    new.obsm['spatial_original'] = new.obsm['spatial'].copy()
    x = new.obsm['spatial'][:, [1, 0]]
    if coordinate_sf is not None:
        x *= coordinate_sf
    transformed, mask = warp_pts_fast(x, ddf)
    new = new[mask]
    new.obsm['spatial'] = transformed[:, [1, 0]]


    d = next(iter(new.uns['spatial'].values()))
    sf = d['scalefactors']['tissue_hires_scalef']
    orig_size = d['images']['hires'].shape
    hires = torch.tensor(rearrange(d['images']['hires'], 'h w -> 1 h w'))
    hires = TF.resize(hires, (int(hires.shape[-2] / sf), int(hires.shape[-1] / sf)), antialias=True).numpy()
    warped_hires = warp_image(hires, ddf)
    warped_hires = TF.resize(torch.tensor(warped_hires), (int(warped_hires.shape[-2] * sf), int(warped_hires.shape[-1] * sf)), antialias=True).numpy()[0]
    d['images']['hires'] = warped_hires / warped_hires.max() # numpy conversion has slight overflow issue


    if resolution is not None:
        new.uns['ppm'] = resolution

    return new

def register_he(img, ddf):
    return warp_image(img, ddf, fill_type='max')

def register_multiplex(data, ddf):
    if isinstance(data, dict):
        return {c:warp_image(img, ddf) for c, img in data.items()}
    return warp_image(data, ddf)


def display_adata(adata, method='both', key='hires', scale=1., ax=None, gamma=1., s=.01):
    img = next(iter(adata.uns['spatial'].values()))['images']['hires']
    sf = next(iter(adata.uns['spatial'].values()))['scalefactors']['tissue_hires_scalef']
    if len(img.shape) == 2:
        img = utils.rescale(img, scale=scale / sf, dim_order='h w')
    else:
        img = utils.rescale(img, scale=scale / sf)
    img = adjust_gamma(img, gamma)

    pts = adata.obsm['spatial'] * scale
    
    if ax is None:
        fig, ax = plt.subplots()

    if method in ['image', 'both']:
        ax.imshow(img)
    if method in ['points', 'both']:
        ax.scatter(pts[:, 0], pts[:, 1], s=s, c='orange')
        if method == 'points':
            ax.invert_yaxis()
 
 
def display_data_map(data_map, multiplex_channel='DAPI', vis_scale=.1, gamma=1., figsize=None, share_axis=False):
    # for sample, mapping in data_map.items():
        # order = mapping['order']
        # sid_to_dtypes = {}
        # for sid in order:
        #     sid_to_dtypes[sid] = [dtype for dtype, entries in mapping['data'].items() if sid in entries]
    target_section = data_map['target_sid']
    target_size = None
    nrows, ncols = len(data_map['sections']), max([len(item['data']) for item in data_map['sections']])
    if figsize is None:
        figsize = (ncols, nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if ncols == 1:
        axs = rearrange(axs, 'n -> n 1')

    for i, item in enumerate(data_map['sections']):
        sid = item['sid']
        for j, mapping in enumerate(item['data']):
            dtype, filepath = mapping['dtype'], mapping['filepath']
            print(sid, dtype)
            ax = axs[i, j]
            parsed_dtype = utils.parse_dtype(dtype)
            img, adata = None, None
            if parsed_dtype == 'visium':
                adata = visium.adata_from_visium(filepath)
                display_adata(adata, scale=vis_scale, method='points', ax=ax)
            elif parsed_dtype == 'xenium':
                adata = xenium.adata_from_xenium(filepath)
                display_adata(adata, scale=vis_scale, method='points', ax=ax, gamma=gamma)
            elif parsed_dtype == 'cosmx':
                adata = cosmx.adata_from_cosmx(filepath)
                display_adata(adata, scale=vis_scale, method='points', ax=ax, gamma=gamma)
            elif parsed_dtype == 'multiplex':
                img = multiplex.extract_ome_tiff(
                    filepath, channels=[multiplex_channel], scale=vis_scale
                )[multiplex_channel]
                img = adjust_gamma(img, gamma)
                ax.imshow(img)
            elif parsed_dtype == 'he':
                img = he.read_he(filepath, scale=vis_scale)
                if img.shape[0] == 3:
                    img = rearrange(img, 'c h w -> h w c')
                ax.imshow(img)
            ax.set_title(dtype)
            if j == 0:
                ax.set_ylabel(sid, rotation=90)
        
        if sid == target_section:
            if img is not None:
                target_size = img.shape[:2]
            else:
                target_size = visium.get_fullres_size(adata)

    for ax in axs.flatten():

        ax.axis('equal')
        if share_axis:
            ax.set_ylim(0, target_size[0])
            ax.set_xlim(0, target_size[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_fontsize(6)
        ax.yaxis.label.set_fontsize(6)


        