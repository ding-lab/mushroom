import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat


def relabel_merged_volume(merged):
    clusters, counts = rearrange(merged, 'n h w z -> (n h w) z').unique(dim=0, return_counts=True)
    int_to_cluster = [tuple(x.numpy()) for i, x in enumerate(clusters)]
    cluster_to_int = {x:i for i, x in enumerate(int_to_cluster)}
    
    new = torch.zeros(merged.shape[:3], dtype=torch.long)
    for i, cluster in enumerate(clusters):
        mask = (merged == cluster).sum(-1) == len(cluster)
        new[mask] = i
    
    return new, int_to_cluster

def merge_volumes(volumes, are_probs=False, kernel=None):
    if kernel is None:
        kernel = torch.full((3,3,3), .2)
        kernel[1,1,1] = 1.
        stamp = rearrange(kernel, 'n h w -> 1 1 1 1 n h w')
    
    if not are_probs:
        hots = [F.one_hot(torch.tensor(v)) for v in volumes]

        probs = []
        for hot in hots:
            hot = hot.unfold(0, 3, 1)
            hot = hot.unfold(1, 3, 1)
            hot = hot.unfold(2, 3, 1)
            out = (hot * stamp).sum(dim=(-3, -2, -1))
            out /= out.max()
            probs.append(out)
    else:
        probs = [torch.tensor(v) for v in volumes]
        
    x = torch.einsum('nhwa,nhwb,nhwc->nhwabc', *probs)
    flat_x = rearrange(x, 'n h w ... -> n h w (...)')
    idxs = flat_x.argmax(-1)
    values = flat_x.max(-1).values
    
    meshes = torch.meshgrid(torch.arange(x.shape[-3]), torch.arange(x.shape[-2]), torch.arange(x.shape[-1]))
    flat_meshes = torch.stack([mesh.flatten() for mesh in meshes])
    objs = flat_meshes[:, idxs.flatten()]
    
    n, h, w = x.shape[:3]
    final = rearrange(objs, 'c (n h w) -> n h w c', n=n, h=h, w=w)
    
    new, int_to_cluster = relabel_merged_volume(final)
    
    new = F.pad(new, (1,1,1,1,1,1), value=-1)
    values = F.pad(values, (1,1,1,1,1,1), value=0.)
    
    return new.numpy(), values.numpy(), int_to_cluster