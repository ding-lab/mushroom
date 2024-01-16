import logging

import numpy as np
import torch
import torchvision.transforms.functional as TF
import tifffile
from tifffile import TiffFile
from einops import rearrange
from torchvision.transforms import Normalize

import mushroom.data.multiplex as multiplex
from mushroom.data.inference import InferenceTransform, InferenceSectionDataset
from mushroom.data.utils import LearnerData

def get_size(filepath):
    tif = TiffFile(filepath)
    p = next(iter(tif.pages))
    return p.shape

def read_he(filepath):
    ext = filepath.split('.')[-1]
    if ext == 'tif':
        return tifffile.imread(filepath)
    elif ext == 'svs':
        raise RuntimeError('reading .svs not implemented yet')
    else:
        raise RuntimeError(f'Extension {ext} not supported for H&E')

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
