import logging
import os

import igraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat

import mushroom.utils as utils

def merge_labels_by_contact(volume, fraction=1e-4):
    min_count = int(np.product(volume.shape) * fraction)
    new = volume.copy()
    labels, counts = np.unique(new, return_counts=True)
    for label, count in zip(labels, counts):
        if count < min_count:
            mask = volume == label
            expanded = skimage.morphology.binary_dilation(mask)
            boundary = expanded & (mask==False)
            boundry_labels = volume[boundary]
            bls, bcs = np.unique(boundry_labels, return_counts=True)
            new_label = bls[bcs.argmax()]
            new[mask] = new_label
            
    new = utils.relabel(torch.tensor(new)).numpy()
    return new

def relabel_merged_volume(merged):
    clusters, counts = rearrange(merged, 'n h w z -> (n h w) z').unique(dim=0, return_counts=True)
    label_to_cluster = [tuple(x.numpy()) for i, x in enumerate(clusters)]
    
    new = torch.zeros(merged.shape[:3], dtype=torch.long)
    for i, cluster in enumerate(clusters):
        mask = (merged == cluster).sum(-1) == len(cluster)
        new[mask] = i
    
    return new, label_to_cluster

def merge_volumes(volumes, are_probs=False, kernel=None, kernel_size=5, block_size=10):
    if not are_probs:
        probs = [F.one_hot(torch.tensor(v)).to(torch.float32) for v in volumes]
    else:
        probs = [torch.tensor(v) for v in volumes]

    smoothed = utils.smooth_probabilities(probs, kernel=kernel, kernel_size=kernel_size) # (n h w nclusters)

    chars = utils.CHARS[:len(smoothed)]
    ein_exp = ','.join([f'nhw{x}' for x in chars])
    ein_exp += f'->nhw{chars}'

    orig_shape = smoothed[0].shape
    n, h, w = orig_shape[:3]
    h_blocks, w_blocks = (h // block_size + 1), (w // block_size + 1)
    h_pad = h_blocks * block_size - h
    w_pad = w_blocks * block_size - w

    smoothed_blocks = []
    for x in smoothed:
        x = F.pad(x, (0, 0, 0, w_pad, 0, h_pad))
        x = rearrange(x, 'n (hb h) (wb w) c -> hb wb n h w c', hb=h_blocks, wb=w_blocks)
        smoothed_blocks.append(x)

    cluster_blocks = torch.empty((h_blocks, w_blocks, n, block_size, block_size, len(smoothed_blocks)), dtype=torch.long)
    for hb in range(h_blocks):
        for wb in range(w_blocks):
            blocks = [x[hb, wb] for x in smoothed_blocks]
            x = torch.einsum(ein_exp, *blocks)
            flat_x = rearrange(x, 'n h w ... -> n h w (...)')
            idxs, values = flat_x.argmax(-1), flat_x.max(-1).values
            meshes = torch.meshgrid(*[torch.arange(s) for s in x.shape[3:]], indexing='ij')
            flat_meshes = torch.stack([mesh.flatten() for mesh in meshes])
            objs = flat_meshes[:, idxs.flatten()]
            cluster_block = rearrange(objs, 'c (n h w) -> n h w c', n=n, h=block_size, w=block_size)
            cluster_blocks[hb, wb] = cluster_block
    clusters = rearrange(cluster_blocks, 'hb wb n h w c -> n (hb h) (wb w) c', hb=h_blocks, wb=w_blocks)
    clusters = clusters[:, :h, :w] # back to original size
    
    relabeled, label_to_cluster = relabel_merged_volume(clusters)
    
    return relabeled.numpy(), values.numpy(), label_to_cluster

def integrate_volumes(dtype_to_volume, dtype_to_cluster_intensities, are_probs=False, dist_thresh=.5, n_iterations=10, resolution=1., dtype_to_weight=None, kernel=None, kernel_size=None, min_fraction=1e-4):
    dtypes, volumes = zip(*dtype_to_volume.items())

    logging.info('merging cluster volumes')
    labeled, _, label_to_cluster = merge_volumes(volumes, are_probs=are_probs, kernel=kernel, kernel_size=kernel_size)
    label_to_cluster = torch.tensor(np.stack(label_to_cluster))

    n_dtypes = len(dtypes)
    n_labels = labeled.max()
    max_c = label_to_cluster.max()

    if dtype_to_weight is not None:
        # make sure dist_thresh will still work
        maximum = np.max(list(dtype_to_weight.values()))
        dtype_to_weight = {k:v / maximum for k, v in dtype_to_weight.items()}
        total = np.sum(list(dtype_to_weight.values()))
        scaler = len(dtype_to_weight) / total
        dtype_to_weight = {k:v * scaler for k, v in dtype_to_weight.items()}

    # dtype_to_cluster_dists = {}
    cluster_dists = torch.zeros(n_dtypes, max_c + 1, max_c + 1)
    for i, dtype in enumerate(dtypes):
        df = dtype_to_cluster_intensities[dtype]
        data = torch.cdist(torch.tensor(df.values), torch.tensor(df.values)).numpy()
        data /= df.shape[1]
        data /= data.std()
        data *= dtype_to_weight[dtype] if dtype_to_weight is not None else 1.
        
        cluster_dists[i, :data.shape[0], :data.shape[1]] = torch.tensor(data)

    n_edges = n_labels * n_labels
    logging.info(f'constructing graph with {n_edges} edges')
    grid = torch.stack(torch.meshgrid(torch.arange(n_labels), torch.arange(n_labels), indexing='ij'))
    edges = rearrange(grid, 'n h w -> (h w) n')
    cluster_edges_a = label_to_cluster[edges[:, 0]]
    cluster_edges_b = label_to_cluster[edges[:, 1]]

    vals = torch.stack(
        [cluster_dists[i, cluster_edges_a[:, i], cluster_edges_b[:, i]] for i in range(cluster_dists.shape[0])]
    )
    weights = vals.mean(0)

    mask = weights < dist_thresh
    edges = edges[mask]
    weights = weights[mask]
    logging.info(str(edges.shape[0]) + ' edges remaining after filtering')
    logging.info(f'{n_labels} of {n_labels} nodes processed')

    weights = weights.numpy()
    edges = edges.numpy()
    
    logging.info('starting leiden clustering')
    g = igraph.Graph()
    g.add_vertices(labeled.max() + 1)
    g.add_edges(edges)
    results = g.community_leiden(weights=weights, resolution=resolution, n_iterations=1, objective_function='modularity')

    to_integrated = {i:c for i, c in enumerate(results.membership)}
    integrated = np.vectorize(to_integrated.get)(labeled)

    integrated = merge_labels_by_contact(integrated, fraction=min_fraction)

    return integrated