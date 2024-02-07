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
    if ext == 'tif':
        img = tifffile.imread(filepath)
    elif ext == 'svs':
        raise RuntimeError('reading .svs not implemented yet')
    else:
        raise RuntimeError(f'Extension {ext} not supported for H&E')
    
    if scale is not None:
        img = rearrange(torch.tensor(img), 'h w c -> c h w')
        target_size = [int(x * scale) for x in img.shape[-2:]]
        img = TF.resize(img, size=target_size, antialias=True)
        img = rearrange(img, 'c h w -> h w c').numpy()
    
    return img

def get_section_to_image(sid_to_filepaths, scale=.1):
    section_to_img = {}
    for sid, filepath in sid_to_filepaths.items():
        logging.info(f'generating image data for section {sid}')
        img = read_he(filepath)
        img = torch.tensor(rearrange(img, 'h w c -> c h w'))
        img = TF.resize(img, (int(scale * img.shape[-2]), int(scale * img.shape[-1])), antialias=True).to(torch.float32)
        img /= img.max()
        img = img.to(torch.float32)
        
        section_to_img[sid] = img
    return section_to_img
