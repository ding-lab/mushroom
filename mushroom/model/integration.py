import os

import igraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat

import mushroom.utils as utils


def relabel_merged_volume(merged):
    clusters, counts = rearrange(merged, 'n h w z -> (n h w) z').unique(dim=0, return_counts=True)
    label_to_cluster = [tuple(x.numpy()) for i, x in enumerate(clusters)]
    
    new = torch.zeros(merged.shape[:3], dtype=torch.long)
    for i, cluster in enumerate(clusters):
        mask = (merged == cluster).sum(-1) == len(cluster)
        new[mask] = i
    
    return new, label_to_cluster

def merge_volumes(volumes, are_probs=False, kernel=None):
    if kernel is None:
        kernel = torch.full((3,3,3), .2)
        kernel[1,1,1] = 1.
        stamp = rearrange(kernel, 'n h w -> 1 1 1 1 n h w')
    
    if not are_probs:
        probs = [F.one_hot(torch.tensor(v)).to(torch.float32) for v in volumes]
        # print(probs[0].shape)
    else:
        probs = [torch.tensor(v) for v in volumes]

    convs = []
    for prob in probs:
        # pad so we end up with the right shape
        prob = rearrange(
            F.pad(rearrange(prob, 'n h w c -> c n h w'), pad=(1,1,1,1,1,1), mode='replicate'),
            'c n h w -> n h w c'
        )

        prob = prob.unfold(0, 3, 1)
        prob = prob.unfold(1, 3, 1)
        prob = prob.unfold(2, 3, 1)
        out = (prob * stamp).sum(dim=(-3, -2, -1))
        out /= out.max()
        convs.append(out)

    chars = utils.CHARS[:len(convs)]
    ein_exp = ','.join([f'nhw{x}' for x in chars])
    ein_exp += f'->nhw{chars}'

    x = torch.einsum(ein_exp, *convs)

    flat_x = rearrange(x, 'n h w ... -> n h w (...)')
    idxs, values = flat_x.argmax(-1), flat_x.max(-1).values
    meshes = torch.meshgrid(
        torch.arange(x.shape[-3]),
        torch.arange(x.shape[-2]),
        torch.arange(x.shape[-1])
    )
    flat_meshes = torch.stack([mesh.flatten() for mesh in meshes])

    objs = flat_meshes[:, idxs.flatten()]
    
    n, h, w = x.shape[:3]
    clusters = rearrange(objs, 'c (n h w) -> n h w c', n=n, h=h, w=w)
    
    relabeled, label_to_cluster = relabel_merged_volume(clusters)
    
    return relabeled.numpy(), values.numpy(), label_to_cluster

def integrate_volumes(dtype_to_volume, dtype_to_cluster_intensities, are_probs=False, dist_thresh=.5, n_iterations=10, resolution=1., dtype_to_weight=None):
    dtypes, volumes = zip(*dtype_to_volume.items())
    labeled, _, label_to_cluster = merge_volumes(volumes, are_probs=are_probs)

    if dtype_to_weight is not None:
        # make sure dist_thresh will still work
        maximum = np.max(list(dtype_to_weight.values()))
        dtype_to_weight = {k:v / maximum for k, v in dtype_to_weight.items()}
        total = np.sum(list(dtype_to_weight.values()))
        scaler = len(dtype_to_weight) / total
        dtype_to_weight = {k:v * scaler for k, v in dtype_to_weight.items()}

    dtype_to_cluster_dists = {}
    for dtype, df in dtype_to_cluster_intensities.items():
        data = torch.cdist(torch.tensor(df.values), torch.tensor(df.values)).numpy()
        data /= data.std()
        data *= dtype_to_weight[dtype] if dtype_to_weight is not None else 1.
        dtype_to_cluster_dists[dtype] = data

    edges, weights = [], []
    for label_a in range(labeled.max()):
        cluster_a = label_to_cluster[label_a]
        node_edges = []
        for label_b in range(labeled.max()):
            cluster_b = label_to_cluster[label_b]
            
            dist = 0.
            if label_a != label_b:
                for dtype, ca, cb in zip(dtypes, cluster_a, cluster_b):
                    dist += dtype_to_cluster_dists[dtype][ca, cb]
            dist /= len(dtypes)
                    
            if dist > 0.:
                node_edges.append([label_a, label_b, dist])
        
        es = sorted(node_edges, key=lambda x:x[-1], reverse=False)
        weights += [val for _, _, val in es if val<=dist_thresh]
        edges += [[a, b] for a, b, val in es if val<=dist_thresh]
    weights = np.asarray(weights)
    edges = np.asarray(edges)
        
    g = igraph.Graph()
    g.add_vertices(labeled.max() + 1)
    g.add_edges(edges)
    results = g.community_leiden(weights=weights, resolution=resolution, n_iterations=n_iterations, objective_function='modularity')

    to_integrated = {i:c for i, c in enumerate(results.membership)}
    integrated = np.vectorize(to_integrated.get)(labeled)

    return integrated