import os

import numpy as np
import scanpy as sc
import torch
import torchvision.transforms.functinoal as TF
from einops import rearrange

import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
import mushroom.data.he as he

def resize(img, scale=None, size=None):
    """
    img = (..., h, w)
    scale = float
    """
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
        is_tensor = False
    else:
        is_tensor = True
    
    if scale is not None:
        size = (int(img.shape[-2] * scale), int(img.shape[-1] * scale))
    
    img = TF.resize(img, size)

    if not is_tensor:
        img = img.numpy()

    return img


def get_fullres_size(config):
    fullres_size = None
    for section in config['sections']:
        for entry in section['data']:
            if entry['dtype'] == 'visium':
                return visium.get_fullres_size(visium.adata_from_visium(entry['filepath']))
            if entry['dtype'] == 'multiplex':
                c, h, w = multiplex.get_size(entry['filepath'])
                return (h, w)
            if entry['dtype'] == 'he':
                h, w, c = he.get_size(entry['filepath'])
                return (h, w)
    

def calculate_target_visualization_shape(config, output_data, pixels_per_micron=None, microns_per_section=5):
    scaled_size = output_data['true_imgs'].shape
    sections = config['sections']
    if pixels_per_micron is None:
        for s in sections:
            for entry in s['data']:
                if entry['dtype'] == 'visium':
                    try:
                        pixels_per_micron = visium.pixels_per_micron(entry['filepath'])
                    except:
                        pass
                if entry['dtype'] == 'multiplex':
                    try:
                        pixels_per_micron = multiplex.pixels_per_micron(entry['filepath'])
                    except:
                        pass
                
                if pixels_per_micron is not None: break
            if pixels_per_micron is not None: break

    step_size = pixels_per_micron * microns_per_section * config['learner_kwargs']['scale']
    z_max = int(step_size * np.max([entry['position'] for entry in sections]))
    
    target_shape = (scaled_size[-1], scaled_size[-2], z_max) # (x, y z)

    return target_shape

class Planes(object):
    def __init__(self):
        pass

    def get_image(self, section_id):
        pass

class VisiumPlanes(Planes):
    def __init__(self, section_to_adata, fullres_size, scaled_size, scale, downsample=.1):
        super().__init__()
        self.image_key = 'vizres'
        self.fullres_size = fullres_size
        self.scaled_size = scaled_size
        self.scale = scale
        self.downsample = downsample
        self.downsample_size = (int(x * self.downsample) for x in self.scaled_size)

        self.scaled_size_nopad = (int(x * self.scale) for x in self.fullres_size)
        pad_h, pad_w = [x - y for x, y in zip(self.scaled_size, self.scaled_size_nopad)]
        self.border = np.asarray([
            pad_w // 2, pad_h // 2, pad_w // 2 + pad_w % 2, pad_h // 2 + pad_h % 2 # left, top, right, bottom
        ])

        self.section_to_adata = section_to_adata
        if isinstance(next(iter(section_to_adata.values())), str):
            self.section_to_adata = {
                k:visium.adata_from_visium(v) for k, v in section_to_adata.items()
            }
        
        for sid, adata in self.section_to_adata.items():
            d = next(iter(adata.uns['spatial'].values()))
            hires = d['images']['hires']
            sf = d['scalefactors']['tissue_hires_scalef']

            # calculate new spot coords and image
            left, top = self.border[:2]
            left, top = left / self.scale, top / self.scale
            adata.obsm['spatial_viz'] = adata.obsm['spatial_viz'] + np.asarray([left, top]).astype(int)

            scaler = self.scale / sf
            vizimg = torch.tensor(rearrange(hires, 'h w c -> c h w'))
            vizimg = resize(vizimg, scale=scaler)
            vizimg = TF.pad(vizimg, self.border)
            vizimg = resize(vizimg, self.downsample)
            vizimg = rearrange(vizimg, 'c h w -> h w c').numpy()
            d = next(iter(adata.uns['spatial'].values()))
            d['images']['vizres'] = vizimg
            d['scalefactors']['tissue_vizres_scalef'] = scaler

class MushroomViewer(object):
    def __init__(self, config, output_data):
        self.fullres_size = get_fullres_size(config)
