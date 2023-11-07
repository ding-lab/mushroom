import os
import re

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchio.transforms import Resize
from einops import rearrange

def listfiles(folder, regex=None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)


def get_interpolated_volume(stacked, section_positions, method='label_gaussian'):
    """
    section_positions - slide indices
    stacked - (n h w) or (c n h w)
    """
    section_positions = np.asarray(section_positions)
    section_range = (section_positions.min(), section_positions.max())

    squeeze = False
    if len(stacked.shape) == 3:
        stacked = rearrange(stacked, 'n h w -> 1 n h w')
        squeeze = True

    interp_volume = np.zeros((stacked.shape[0], section_range[-1], stacked.shape[-2], stacked.shape[-1]), dtype=stacked.dtype)
    for i in range(stacked.shape[1] - 1):
        l1, l2 = section_positions[i], section_positions[i+1]

        stack = stacked[:, i:i+2]
        transform = Resize((l2 - l1, stack.shape[-2], stack.shape[-1]), image_interpolation=method)
        resized = transform(stack)

        interp_volume[:, l1:l2] = resized
    
    if squeeze:
        interp_volume = rearrange(interp_volume, '1 n h w -> n h w')

    return interp_volume


def display_thresholds(cuts, cluster_ids, intensity_df, channel):
    nrows, ncols = len(cuts), cluster_ids.shape[0]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
    for cut_idx in range(nrows):
        cut = cuts[cut_idx]
        clusters = intensity_df[intensity_df[channel] >= cut].index.to_list()
        for section_idx in range(ncols):
            ax = axs[cut_idx, section_idx]
            masks = np.stack(cluster_ids[section_idx]==cluster for cluster in clusters)
            mask = masks.sum(0) > 0
            ax.imshow(mask)
            ax.set_xticks([])
            ax.set_yticks([])
        
        axs[cut_idx, 0].set_ylabel("%.2f" % cut, rotation=90)
    return axs
        
