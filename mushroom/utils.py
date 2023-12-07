import os
import re

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from torchio.transforms import Resize
from einops import rearrange
from sklearn.cluster import AgglomerativeClustering

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


def relabel(labels):
    new = torch.zeros_like(labels, dtype=labels.dtype)
    ids = labels.unique()
    for i in range(len(ids)):
        new[labels==ids[i]] = i
        
    return new


def aggregate_clusters(df, cluster_ids, n_clusters=10, distance_threshold=None):
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, distance_threshold=distance_threshold
    ).fit(df.values)
    cluster_to_label = {c:l for c, l in zip(df.index.to_list(), clustering.labels_)}
    agg_ids = np.vectorize(cluster_to_label.get)(cluster_ids)
    return cluster_to_label, agg_ids


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


def rescale(x, scale=.1, size=None, dim_order='h w c', target_dtype=torch.uint8, antialias=True, interpolation=TF.InterpolationMode.BILINEAR):
    is_tensor = isinstance(x, torch.Tensor)
    if not is_tensor:
        x = torch.tensor(x)

    if dim_order == 'h w c':
        x = rearrange(x, 'h w c -> c h w')
    elif dim_order == 'h w':
        x = rearrange(x, 'h w -> 1 h w')

    if size is None:
        size = (int(x.shape[-2] * scale), int(x.shape[-1] * scale))

    x = TF.resize(x, size, antialias=antialias, interpolation=interpolation)
    x = TF.convert_image_dtype(x, target_dtype)

    if dim_order == 'h w c':
        x = rearrange(x, 'c h w -> h w c')
    elif dim_order == 'h w':
        x = rearrange(x, '1 h w -> h w')

    if not is_tensor:
        x = x.numpy()
    
    return x
            
