import collections.abc
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchio.transforms import Resize
from einops import rearrange
from sklearn.cluster import AgglomerativeClustering

CHARS = 'abcdefghijklmnopqrstuvwxyz'

DEFAULT_KERNEL_3 = torch.full((3,3,3), .25)
DEFAULT_KERNEL_3[2, 2, 2] = 1.

DEFAULT_KERNEL_5 = torch.full((5,5,5), .1)
DEFAULT_KERNEL_5[1:-1, 1:-1, 1:-1] = .25
DEFAULT_KERNEL_5[2, 2, 2] = 1.

DEFAULT_KERNEL_7 = torch.full((7,7,7), .05)
DEFAULT_KERNEL_7[1:-1, 1:-1, 1:-1] = .1
DEFAULT_KERNEL_7[2:-2, 2:-2, 2:-2] = .25
DEFAULT_KERNEL_7[3, 3, 3] = 1.

DEFAULT_KERNELS = {
    3: DEFAULT_KERNEL_3,
    5: DEFAULT_KERNEL_5,
    7: DEFAULT_KERNEL_7
}

DTYPES = ('multiplex', 'xenium', 'visium', 'he', 'cosmx', 'points',)

def listfiles(folder, regex=None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)

# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_dtype(dtype_identifier):
    if '_' not in dtype_identifier:
        parsed = dtype_identifier
    else:
        parsed = dtype_identifier.split('_')[-1]

    if parsed not in DTYPES:
        raise RuntimeError(f'{dtype_identifier} was supplied and its parsed form {parsed} is not a valid dtype string. valid data identifiers are either {DTYPES} or "[string]_[dtype]" where [string] can be any filepath-safe string and [dtype] must in {DTYPES}')
    
    return parsed

def smooth_probabilities(probs, kernel=None, kernel_size=5):
    """
    probs - (n h w labels)
    kernel - (k, k, k) where k is kernel size
    """
    if kernel is None and kernel_size is None:
        return probs
    
    if kernel is None:
        kernel = DEFAULT_KERNELS[5]

    is_numpy = isinstance(probs, np.ndarray)
    if is_numpy:
        probs = torch.tensor(probs)

    stamp = rearrange(kernel, '... -> 1 1 1 1 ...')
    convs = []
    for prob in probs:
        # pad so we end up with the right shape
        kernel_size = kernel.shape[0]
        pad = tuple([kernel_size // 2 for i in range(6)])
        prob = rearrange(
            F.pad(rearrange(prob, 'n h w c -> c n h w'), pad=pad, mode='replicate'),
            'c n h w -> n h w c'
        )

        prob = prob.unfold(0, kernel_size, 1)
        prob = prob.unfold(1, kernel_size, 1)
        prob = prob.unfold(2, kernel_size, 1)
        out = (prob * stamp).sum(dim=(-3, -2, -1))
        out /= out.max()

        if is_numpy:
            out = out.numpy()

        convs.append(out)
    return convs

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

    interp_volume = np.zeros((stacked.shape[0], section_range[-1] + 1, stacked.shape[-2], stacked.shape[-1]), dtype=stacked.dtype)
    for i in range(stacked.shape[1] - 1):
        l1, l2 = section_positions[i], section_positions[i+1]

        stack = stacked[:, i:i+2]

        transform = Resize((l2 - l1, stack.shape[-2], stack.shape[-1]), image_interpolation=method)
        resized = transform(stack)
        interp_volume[:, l1:l2] = resized
    
    # add last section
    interp_volume[:, -1] = stacked[:, -1]
    
    if squeeze:
        interp_volume = rearrange(interp_volume, '1 n h w -> n h w')

    return interp_volume

def smoosh(*args):
    new = 0
    for i, val in enumerate(args):
        new += val * 10**i
    return new

def relabel(labels):
    new = torch.zeros_like(labels, dtype=labels.dtype)
    ids = labels.unique()
    for i in range(len(ids)):
        new[labels==ids[i]] = i
        
    return new

# def label_agg_clusters(clusters):
#     smooshed = np.vectorize(smoosh)(*clusters)
#     relabeled = relabel(torch.tensor(smooshed)).numpy()
#     mapping = {relabeled[s, r, c]:tuple([x[s, r, c].item() for x in clusters]) for s in range(relabeled.shape[0]) for r in range(relabeled.shape[1]) for c in range(relabeled.shape[2])}
#     return relabeled, mapping
def label_agg_clusters(agg_clusters):
    aggs = np.stack(agg_clusters)
    aggs = rearrange(aggs, 'z n h w -> (n h w) z')
    aggs = np.unique(aggs, axis=0)
    agg_to_label = {tuple(agg):i for i, agg in enumerate(aggs)}
    label_to_agg = {v:k for k, v in agg_to_label.items()}

    def assign_labels(*args):
        return agg_to_label[tuple(args)]

    relabeled = np.vectorize(assign_labels)(*agg_clusters)
    
    return relabeled, label_to_agg


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

    
    dtype_map = { # quick and dirty dtype mapping for common np dtypes
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.int64): torch.int64,
        np.dtype(bool): torch.bool,
        np.uint8: torch.uint8,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int64: torch.int64,
        bool: torch.bool,
    }

    target_dtype = dtype_map.get(np.dtype(target_dtype) if isinstance(target_dtype, np.dtype) else target_dtype, target_dtype)

    x = TF.resize(x, size, antialias=antialias, interpolation=interpolation)
    
    # really need to rewrite this in a sane way
    if x.dtype not in [torch.long, torch.int64, torch.int32, torch.bool, torch.float32, torch.float64] and x.dtype != target_dtype: # if its a labeled image this wont work
        x = TF.convert_image_dtype(x, target_dtype)

    if dim_order == 'h w c':
        x = rearrange(x, 'c h w -> h w c')
    elif dim_order == 'h w':
        x = rearrange(x, '1 h w -> h w')

    if not is_tensor:
        x = x.numpy()
    
    return x

def read_mask(mask):
    if isinstance(mask, str):
        ext = mask.split('/')[-1].split('.')[-1]
        if ext == 'tif':
            mask = tifffile.imread(mask)
        elif ext in ['npy', 'npz']:
            mask = np.load(mask)
        else:
            raise RuntimeError(f'Extension {ext} is not supported for masks')

    if mask is not None:
        return mask > 0
    