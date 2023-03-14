import torch
import torchvision.transforms.functional as TF
import numpy as np
from einops import rearrange


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_means_and_stds(adatas):
    means, stds = None, None
    for a in adatas:
        x = rearrange(next(iter(a.uns['spatial'].values()))['images']['lowres'], 'h w c -> c h w')

        if means is None:
            means = x.mean((1, 2))
            stds = x.std((1, 2))
        else:
            means = (means + x.mean((1, 2))) / 2
            stds = (stds + x.std((1, 2))) / 2
    return means, stds


def rescale_img(img, scale=.5, shape=None):
    if shape is None:
        h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    else:
        h, w = shape
    scaled = TF.resize(rearrange(torch.Tensor(img), 'h w c -> c h w'), size=(h, w))
    return rearrange(scaled, 'c h w -> h w c').numpy().astype(np.uint8)


def rescale_with_pad(img, scale=.5, shape=None, padding_mode='reflect'):
    if shape is None:
        h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    else:
        h, w = shape
    r, c = img.shape[0], img.shape[1]
    
    right_pad = r-c if r>c else 0
    bottom_pad = c-r if c>r else 0
    padded = TF.pad(rearrange(torch.Tensor(img), 'h w c -> c h w'),
                    padding=[0, 0, right_pad, bottom_pad], padding_mode=padding_mode)
    
    scaled = TF.resize(padded, size=(h, w))
    return rearrange(scaled, 'c h w -> h w c').numpy().astype(np.uint8)


def project_expression(labeled, exp, voxel_idxs):

    new = torch.zeros((labeled.shape[-2], labeled.shape[-1], exp.shape[1]), dtype=exp.dtype)
    for i, idx in enumerate(voxel_idxs):
        new[labeled.squeeze()==idx] = exp[i]
    return new


def construct_tile_expression(padded_exp, masks, n_voxels, normalize=True):
    tile = torch.zeros((masks.shape[0], masks.shape[-2], masks.shape[-1], padded_exp.shape[-1]),
                       device=padded_exp.device)
    for b in range(tile.shape[0]):
        for exp, m in zip(padded_exp[b], masks[b]):
            tile[b, :, :][m==1] = exp.to(torch.float32)
            
    tile = rearrange(tile, 'b h w c -> b c h w')
    tile = tile.detach().cpu().numpy()
    
    tile /= np.expand_dims(tile.max(axis=(0, -2, -1)), (0, -2, -1))

    return rearrange(tile, 'b c h w -> b h w c')