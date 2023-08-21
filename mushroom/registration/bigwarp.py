import numpy as np
import torch
import torchvision.transforms.functional as TF
import tifffile
from einops import rearrange
from scipy import ndimage as ndi


def read_bigwarp_warp_field(fp, downsample_scaler):
    """
    Read bigwarp 
    """
    ddf = torch.tensor(tifffile.imread(fp))
    ddf = ddf[[1, 0]] # needs to be (h, w) in first dim, bigwarp exports (w, h, c)

    # rescale to original size
    scale = 1 / downsample_scaler
    ddf = TF.resize(ddf, (int(ddf.shape[-2] * scale), int(ddf.shape[-1] * scale)))
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
    
    pts - (n, 2) # 2 is h, w
    ddf - (2 h w) # first channel is h displacment, second channel is w displacement
    """
    if not isinstance(pts, torch.Tensor):
        pts = torch.tensor(pts)

    max_r, max_c = pts.max(dim=0).values
    img = torch.zeros((max_r + 1, max_c + 1), dtype=torch.long)
    for i, (r, c) in enumerate(pts):
        img[r, c] = i + 1

    img = warp_image(img, ddf)

    objects = ndi.find_objects(img.numpy())
    label_to_warped_pt = {}
    for i, obj in enumerate(objects):
        if obj is None:
            continue

        r, c = obj[0].start, obj[1].start
        label_to_warped_pt[i] = (r, c)

    idxs = torch.arange(pts.shape[0], dtype=torch.long)
    size = (ddf.shape[-2], ddf.shape[-1])
    mask = torch.tensor(
        [True if i.item() in label_to_warped_pt and is_valid(label_to_warped_pt[i.item()], size) else False
        for i in idxs], dtype=torch.bool)
    idxs = idxs[mask]
    warped = torch.tensor([list(label_to_warped_pt[i.item()]) for i in idxs], dtype=torch.long)

    return warped, mask


def regenerate_adata(he, adata, ddf, phenocycler_pixels_per_micron=1.9604911906033102):
    """
    he - (3, h, w)
    labeled - (1, h, w)
    """
    new = adata.copy()
    pix_per_micron = next(iter(
        adata.uns['spatial'].values()))['scalefactors']['spot_diameter_fullres'] / 65.
    scale = 1 / pix_per_micron * phenocycler_pixels_per_micron # bring to codex resolution

    d = next(iter(new.uns['spatial'].values()))
    scalefactors = d['scalefactors']

    hires_size = (int(scalefactors['tissue_hires_scalef'] * he.shape[-2]),
                  int(scalefactors['tissue_hires_scalef'] * he.shape[-1]))
    lowres_size = (int(scalefactors['tissue_lowres_scalef'] * he.shape[-2]),
                  int(scalefactors['tissue_lowres_scalef'] * he.shape[-1]))

    hires = TF.resize(he, hires_size)
    lowres = TF.resize(he, lowres_size)

    d['images']['hires'] = rearrange(hires, 'c h w -> h w c').numpy()
    d['images']['lowres'] = rearrange(lowres, 'c h w -> h w c').numpy()

    scalefactors['spot_diameter_fullres'] *= scale
    scalefactors['fiducial_diameter_fullres'] *= scale

    new.obsm['spatial_original'] = new.obsm['spatial'].copy()
    x = (torch.tensor(new.obsm['spatial']) * scale).to(torch.long)
    x = x[:, [1, 0]]
    transformed, mask = warp_pts(x, ddf)
    new = new[mask.numpy()]
    new.obsm['spatial'] = transformed[:, [1, 0]].numpy()
 
    return new


def register_visium(he, adata, ddf, phenocycler_pixels_per_micron=1.9604911906033102):
    pix_per_micron = next(iter(
        adata.uns['spatial'].values()))['scalefactors']['spot_diameter_fullres'] / 65.
    scale = 1 / pix_per_micron * phenocycler_pixels_per_micron # bring to codex resolution
    target = (int(he.shape[-2] * scale), int(he.shape[-1] * scale))

    he = TF.resize(he, target)

    adata.uns['he_rescaled'] = rearrange(he, 'c h w -> h w c').numpy()
    adata.obsm['he_rescaled_spatial'] = (adata.obsm['spatial'] * scale).astype(np.int32)

    warped_he = warp_image(he, ddf)
    adata.uns['he_rescaled_warped'] = rearrange(warped_he, 'c h w -> h w c').numpy()

    new = regenerate_adata(warped_he, adata, ddf)

    return new


def register_multiplex(channel_to_img, ddf):
    warped_channel_to_img = {c:warp_image(img, ddf) for c, img in channel_to_img.items()}
    
    return warped_channel_to_img