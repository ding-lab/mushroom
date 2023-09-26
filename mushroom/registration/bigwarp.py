import numpy as np
import torch
import torchvision.transforms.functional as TF
import tifffile
from einops import rearrange
from scipy import ndimage as ndi

from mushroom.data.visium import get_fullres_size


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


def warp_image(moving, ddf):
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
    warped[..., mask.sum(0)<2] = 0
    
    return warped


def is_valid(pt, size):
    r, c = pt
    return (r >= 0) & (r < size[-2]) & (c >= 0) & (c < size[-1])


def warp_pts(pts, ddf):
    """
    assumes 2d transform
    
    pts - (n, 2) # 2 is height, width
    ddf - (2 h w) # first channel is h displacment, second channel is w displacement
    """
    if not isinstance(pts, torch.Tensor):
        pts = torch.tensor(pts)

    max_r, max_c = pts.max(dim=0).values
    img = torch.zeros((max_r + 1, max_c + 1), dtype=torch.long)
    for i, (r, c) in enumerate(pts):
        r1, r2 = max(0, r - 1), min(max_r + 1, r + 1)
        c1, c2 = max(0, c - 1), min(max_c + 1, c + 1)
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


# def register_visium(he, adata, ddf, target_pix_per_micron=1., moving_pix_per_micron=None):
#     """
#     he - (3, h, w)
#     labeled - (1, h, w)
#     """
#     if not isinstance(he, torch.Tensor):
#         he = torch.tensor(he)
#     new = adata.copy()
#     if moving_pix_per_micron is None:
#         moving_pix_per_micron = next(iter(
#             adata.uns['spatial'].values()))['scalefactors']['spot_diameter_fullres'] / 65.
#     scale = target_pix_per_micron / moving_pix_per_micron # bring to target img resolution
#     he = register_he(he, ddf) # warped he is at target resolution

#     d = next(iter(new.uns['spatial'].values()))
#     scalefactors = d['scalefactors']

#     hires_size = (int(scalefactors['tissue_hires_scalef'] * he.shape[-2]),
#                   int(scalefactors['tissue_hires_scalef'] * he.shape[-1]))
#     lowres_size = (int(scalefactors['tissue_lowres_scalef'] * he.shape[-2]),
#                   int(scalefactors['tissue_lowres_scalef'] * he.shape[-1]))
    
#     hires = TF.resize(he, hires_size, antialias=True)
#     lowres = TF.resize(he, lowres_size, antialias=True)

#     d['images']['hires'] = rearrange(hires, 'c h w -> h w c').numpy()
#     d['images']['lowres'] = rearrange(lowres, 'c h w -> h w c').numpy()

#     scalefactors['spot_diameter_fullres'] *= scale
#     scalefactors['fiducial_diameter_fullres'] *= scale

#     new.obsm['spatial_original'] = new.obsm['spatial'].copy()
#     x = (torch.tensor(new.obsm['spatial']) * scale).to(torch.long)
#     x = x[:, [1, 0]] # needs to be (h, w) instead of (w, h)
#     transformed, mask = warp_pts(x, ddf)
#     new = new[mask.numpy()]
#     new.obsm['spatial'] = transformed[:, [1, 0]].numpy()
 
#     return new

def resize_and_warp(moving, ddf):
    """
    moving - (c, h, w)
    ddf - (h, w)
    """
    downsample =  moving.shape[-2] / ddf.shape[-2]
    downsample_ddf = TF.resize(ddf, (moving.shape[-2], moving.shape[-1]), antialias=False)
    downsample_ddf *= downsample
    downsample_ddf = downsample_ddf.to(torch.long)
    return register_he(moving, downsample_ddf)


def register_visium(adata, ddf, target_pix_per_micron=1., moving_pix_per_micron=None):
    new = adata.copy()
    if moving_pix_per_micron is None:
        moving_pix_per_micron = next(iter(
            adata.uns['spatial'].values()))['scalefactors']['spot_diameter_fullres'] / 65.
    scale = target_pix_per_micron / moving_pix_per_micron # bring to target img resolution

    d = next(iter(new.uns['spatial'].values()))
    scalefactors = d['scalefactors']

    hires, lowres = torch.tensor(d['images']['hires']), torch.tensor(d['images']['lowres'])
    hires = TF.resize(
        rearrange(hires, 'h w c -> c h w'), (int(scale * hires.shape[0]), int(scale * hires.shape[1])), antialias=True,
    )
    lowres = TF.resize(
        rearrange(lowres, 'h w c -> c h w'), (int(scale * lowres.shape[0]), int(scale * lowres.shape[1])), antialias=True,
    )

    warped_hires = rearrange(resize_and_warp(hires, ddf), 'c h w -> h w c').numpy()
    d['images']['hires'] = warped_hires / warped_hires.max() # numpy conversion has slight overflow issue

    warped_lowres = rearrange(resize_and_warp(lowres, ddf), 'c h w -> h w c').numpy()
    d['images']['lowres'] = warped_lowres / warped_lowres.max() # numpy conversion has slight overflow issue

    new.obsm['spatial_original'] = new.obsm['spatial'].copy()
    x = (torch.tensor(new.obsm['spatial']) * scale).to(torch.long)
    x = x[:, [1, 0]] # needs to be (h, w) instead of (w, h)
    transformed, mask = warp_pts(x, ddf)
    new = new[mask.numpy()]
    new.obsm['spatial'] = transformed[:, [1, 0]].numpy()

    return new

def register_he(he, ddf):
    return warp_image(he, ddf)

def register_multiplex(data, ddf):
    if isinstance(data, dict):
        return {c:warp_image(img, ddf) for c, img in data.items()}
    return warp_image(data, ddf)