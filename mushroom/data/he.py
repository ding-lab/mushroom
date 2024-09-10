import logging
import re

import numpy as np
import torch
import torchvision.transforms.functional as TF
import tifffile
from tifffile import TiffFile
from einops import rearrange
from torchvision.transforms import Normalize
from ome_types import from_xml

import mushroom.data.multiplex as multiplex
from mushroom.data.inference import InferenceTransform, InferenceSectionDataset
from mushroom.data.utils import LearnerData

def pixels_per_micron(filepath):
    tf = tifffile.TiffFile(filepath)
    p = next(iter(tf.pages))

    # test for .svs metadata
    matches = re.findall(r'MPP = [0-9]+.[0-9]+', p.description)
    if len(matches):
        return 1 / float(matches[0].split()[-1])
    
    # test for OME-TIF
    if p.is_ome:
        ome = from_xml(tf.ome_metadata)
        im = ome.images[0]
        return im.pixels.physical_size_x
    
    # just give back resolution of page
    if p.resolution is not None:
        return p.resolution[0]

def get_size(filepath):
    tif = TiffFile(filepath)
    p = next(iter(tif.pages))
    return p.shape

def read_he(filepath, scale=None):
    ext = filepath.split('.')[-1]
    if ext in ['tif', 'svs', 'tiff']:
        img = tifffile.imread(filepath)
    else:
        logging.warning(f'File extension {ext} is not .tif or .svs, attempting to open but may cause errors')
        # raise RuntimeError(f'File extension {ext} not supported for H&E. Supported extensions are .tif or .svs')
        img = tifffile.imread(filepath)
    
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
        dim_order = 'h w c'
    elif img.shape[0] == 3:
        dim_order = 'c h w'
    else:
        raise RuntimeError(f'Dimensions of {img.shape} are not valid HE dimensions. Must be (3, height, width) or (height, width, 3)')

    if scale is not None:
        img = torch.tensor(img)
        target_size = [int(x * scale) for x in img.shape[-2:]]
        img = TF.resize(img, size=target_size, antialias=True)
        img = img.numpy()

    if dim_order == 'h w c':
        img = rearrange(img, 'c h w -> h w c')
    
    return img

def get_section_to_image(sid_to_filepaths, scale=.1):
    section_to_img = {}
    for sid, filepath in sid_to_filepaths.items():
        logging.info(f'generating image data for section {sid}')
        img = read_he(filepath)
        img = torch.tensor(img)
        if img.shape[-1] == 3:
            img = rearrange(img, 'h w c -> c h w')
        img = TF.resize(img, (int(scale * img.shape[-2]), int(scale * img.shape[-1])), antialias=True).to(torch.float32)
        img /= img.max()
        img = img.to(torch.float32)
        
        section_to_img[sid] = img
    return section_to_img
