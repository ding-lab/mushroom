import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import tifffile
import torch
import torchio
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from scipy.interpolate import interp1d

import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
import mushroom.data.he as he
from mushroom.visualization.napari import NapariImageArgs

def resize(img, scale=None, size=None, antialias=True, interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
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
    
    img = TF.resize(img, size, antialias=antialias, interpolation=interpolation)

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
    def __init__(self, section_to_data, fullres_size, scaled_size, scale):
        self.section_to_data = section_to_data
        self.fullres_size = fullres_size
        self.scaled_size = scaled_size
        self.scale = scale

        self.scaled_size_nopad = tuple([int(x * self.scale) for x in self.fullres_size])
        pad_h, pad_w = [x - y for x, y in zip(self.scaled_size, self.scaled_size_nopad)]
        self.border = [
            pad_w // 2, pad_h // 2, pad_w // 2 + pad_w % 2, pad_h // 2 + pad_h % 2 # left, top, right, bottom
        ]

    def get_image(self, section_id):
        pass

class VisiumPlanes(Planes):
    def __init__(self, section_to_data, fullres_size, scaled_size, scale, normalize=True):
        super().__init__(section_to_data, fullres_size, scaled_size, scale)

        self.section_to_adata = self.section_to_data
        if isinstance(next(iter(self.section_to_adata.values())), str):
            self.section_to_adata = {
                k:visium.adata_from_visium(v, normalize=normalize) for k, v in self.section_to_adata.items()
            }
        
        for sid, adata in self.section_to_adata.items():
            d = next(iter(adata.uns['spatial'].values()))
            hires = d['images']['hires']
            sf = d['scalefactors']['tissue_hires_scalef']

            # calculate new spot coords and image
            left, top = self.border[:2]
            adata.obsm['spatial_viz'] = ((adata.obsm['spatial'] * self.scale + np.asarray([left, top])) / self.scale).astype(int)
            # left, top = left / self.scale, top / self.scale
            # adata.obsm['spatial_viz'] = adata.obsm['spatial'] + np.asarray([left, top]).astype(int)

            scaler = self.scale / sf
            vizimg = torch.tensor(rearrange(hires, 'h w c -> c h w'))
            vizimg = resize(vizimg, scale=scaler)
            vizimg = TF.pad(vizimg, padding=self.border)
            # vizimg = resize(vizimg, self.downsample)
            vizimg = rearrange(vizimg, 'c h w -> h w c').numpy()
            d = next(iter(adata.uns['spatial'].values()))
            d['images']['vizres'] = vizimg
            d['scalefactors']['tissue_vizres_scalef'] = self.scale
    
    def get_image(self, section_id, marker=None):
        fp = 'tmp.png'
        adata = self.section_to_adata[section_id]
        _, ax = plt.subplots()
        left, top, right, bottom = [int(x / self.scale) for x in self.border]
        crop_coords = [0, self.fullres_size[-2] + top + bottom, 0, self.fullres_size[-1] + left + right]
        _ = sc.pl.spatial(adata, img_key='vizres', basis='spatial_viz', color=marker,
                            crop_coord=crop_coords, show=False, colorbar_loc=None, ax=ax)
        ax.axis('off')
        ax.set_title('')
        plt.savefig(fp, bbox_inches='tight', pad_inches=0., dpi=300)
        plt.close()
        img = np.asarray(Image.open(fp))
        img = img[..., :-1] # (h, w, c)
        img = rearrange(img, 'h w c -> c h w')

        os.remove(fp)

        return img
    
    def get_spots(self, section_id, marker=None):
        adata = self.section_to_adata[section_id]
        if marker is None:
            marker = adata.var.index.to_list()[0]

        X = adata[:, marker].X
        if 'sparse' in str(type(X)).lower():
            X = X.toarray()
        spots = [{'x': int(x), 'y':int(y), 'value': val}
                 for (x, y), val in zip(adata.obsm['spatial_viz'] * self.scale, X[:, 0])]
        
        return spots
    
    def get_spots_img_shape(self, section_id):
        adata = self.section_to_adata[section_id]
        d = next(iter(adata.uns['spatial'].values()))
        return d['images']['vizres'].shape
    

class MultiplexPlanes(Planes):
    def __init__(self, section_to_data, fullres_size, scaled_size, scale, channel_mapping=None):
        super().__init__(section_to_data, fullres_size, scaled_size, scale)

        if channel_mapping is None:
            channel_mapping = {}

        self.section_to_img = {}
        self.section_to_channels = {}
        for sid, filepath in self.section_to_data.items():
            channels = multiplex.get_ome_tiff_channels(filepath)
            channels = [channel_mapping.get(c, c) for c in channels]
            img = next(iter(multiplex.get_section_to_image(
                {sid:filepath}, channels, channel_mapping=channel_mapping, scale=self.scale
            ).values()))
            img = TF.pad(img, padding=self.border).numpy()
            self.section_to_img[sid] = img
            self.section_to_channels[sid] = [channel_mapping.get(c, c) for c in channels]
    
    def get_image(self, section_id, marker=None):
        if marker is None:
            return self.section_to_img[section_id]
        channels = self.section_to_channels[section_id]
        return np.expand_dims(self.section_to_img[section_id][channels.index(marker)], 0)
    
    def get_channels(self, section_id):
        return self.section_to_channels[section_id]
    

class HEPlanes(Planes):
    def __init__(self, section_to_data, fullres_size, scaled_size, scale):
        super().__init__(section_to_data, fullres_size, scaled_size, scale)

        self.section_to_img = {}
        for sid, filepath in self.section_to_data.items():
            img = he.read_he(filepath)
            img = torch.tensor(rearrange(img, 'h w c -> c h w'))
            img_scaled_nopad = resize(img, scale=self.scale)
            img_scaled = TF.pad(img_scaled_nopad, padding=self.border)
            self.section_to_img[sid] = img_scaled.numpy()
    
    def get_image(self, section_id):
        return self.section_to_img[section_id]


class MushroomViewer(object):
    def __init__(self, config, output_data, downsample=1., pixels_per_micron=None, microns_per_section=5):
        self.pixels_per_micron = pixels_per_micron
        self.microns_per_section = microns_per_section
        self.scale = config['learner_kwargs']['scale']
        self.downsample = downsample

        self.fullres_size = get_fullres_size(config) # (h, w)
        self.scaled_shape = calculate_target_visualization_shape( # (w, h, z)
            config, output_data,
            pixels_per_micron=self.pixels_per_micron, microns_per_section=self.microns_per_section
        )
        self.downsampled_shape = tuple([int(x * self.downsample) for x in self.scaled_shape])
        self.scaled_size = (self.scaled_shape[1], self.scaled_shape[0]) # (h, w)

        self.section_ids = [entry['id'] for entry in config['sections']]
        self.initial_positions = np.asarray([entry['position'] for entry in config['sections']])
        m = interp1d([0, self.initial_positions.max()], [0, self.downsampled_shape[-1]])
        self.positions = np.asarray([m(x) for x in self.initial_positions]).astype(int)

        self.volume, self.cluster_ids = self.parse_output_data(output_data, self.downsampled_shape) # d, z, y, x

        self.dtype_mapping = {}
        for entry in config['sections']:
            for d in entry['data']:
                if d['dtype'] not in self.dtype_mapping:
                    self.dtype_mapping[d['dtype']] = {}
                self.dtype_mapping[d['dtype']][entry['id']] = d['filepath']
        
        self.dtype_to_planes = {}
        for dtype, section_to_data in self.dtype_mapping.items():
            if dtype == 'visium':
                self.dtype_to_planes[dtype] = VisiumPlanes(
                    section_to_data, self.fullres_size, self.scaled_size, self.scale
                )
            if dtype == 'multiplex':
                self.dtype_to_planes[dtype] = MultiplexPlanes(
                    section_to_data, self.fullres_size, self.scaled_size, self.scale,
                    channel_mapping=config['learner_kwargs']['channel_mapping']
                )
            if dtype == 'he':
                self.dtype_to_planes[dtype] = HEPlanes(
                    section_to_data, self.fullres_size, self.scaled_size, self.scale
                )
    
    def parse_output_data(self, output_data, downsampled_shape):
        target_size = (downsampled_shape[-1], downsampled_shape[-2])
        transform = torchio.transforms.Resize(downsampled_shape)

        if isinstance(output_data, str):
            output_data = torch.load(output_data)

        volume = rearrange(output_data['cluster_distance_volume'], 'z y x d -> d x y z')
        volume = transform(volume)
        volume = rearrange(volume, 'd x y z -> d z y x')
        
        cluster_ids = output_data['cluster_ids']

        return volume, cluster_ids
    
    def get_sections(self, dtype=None, dtype_to_marker=None):
        if dtype_to_marker is None:
            dtype_to_marker = {}
        napari_args_list = []
        for dtype, planes in self.dtype_to_planes.items():
            if dtype is None or dtype==dtype:
                for section_id in planes.section_to_data.keys():
                    markers = dtype_to_marker.get(dtype, [None])
                    if markers[0] is None and dtype == 'multiplex':
                            markers = planes.get_channels(section_id)
                    for marker in markers:
                        img, channels, spots = None, None, None
                        if dtype == 'visium':
                            img = planes.get_image(section_id, marker=marker)
                            img = np.asarray(TF.rgb_to_grayscale(torch.tensor(img)))
                            spots = planes.get_spots(section_id, marker)
                            spot_img_shape = planes.get_spots_img_shape(section_id)
                            sf = self.volume.shape[-1] / spot_img_shape[1]
                            channels = [marker]
                        if dtype == 'multiplex':
                            img = planes.get_image(section_id, marker=marker)
                            channels = [marker]
                            sf = self.volume.shape[-1] / img.shape[-1]
                        if dtype == 'he':
                            img = planes.get_image(section_id)
                            img = np.asarray(TF.rgb_to_grayscale(torch.tensor(img)))
                            sf = self.volume.shape[-1] / img.shape[-1]
                        
                        napari_args_list.append(
                            NapariImageArgs(
                                name=f'{dtype}_{section_id}',
                                position=self.positions[self.section_ids.index(section_id)],
                                dtype=dtype,
                                scale_factor=sf,
                                img=img,
                                channels=channels,
                                spots=spots,
                            )
                        )

        return napari_args_list
            
        
