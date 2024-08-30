import contextlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import seaborn as sns
import skimage
import torch
import torchvision.transforms.functional as TF
import tifffile
from einops import rearrange, repeat
from matplotlib.colors import CSS4_COLORS, LinearSegmentedColormap
from PIL import Image
from skimage.exposure import adjust_gamma
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation

SEQUENTIAL_CMAPS = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

COLORS = [
    np.asarray([52, 40, 184]) / 255., # blue
    np.asarray([204, 137, 4]) / 255., # orange
    np.asarray([25, 117, 5]) / 255., # green
    np.asarray([161, 19, 14]) / 255., # red
    np.asarray([109, 12, 148]) / 255., # purple
    np.asarray([92, 60, 1]) / 255., # brown
    np.asarray([224, 54, 185]) / 255., # pink
    np.asarray([2, 194, 191]) / 255., # cyan
]
COLORS += CSS4_COLORS


def get_cmap(n):
    if n < 10:
        return sns.color_palette()
    if n < 20:
        return sns.color_palette('tab20')
    
    iters = (n // 20) + 2
    cmap = []
    for i in range(iters):
        cmap += sns.color_palette('tab20')
    
    return cmap

def get_hierarchical_cmap(label_to_hierarchy):
    aggs = np.stack(list(label_to_hierarchy.values()))
    n_clusts = aggs.max(0) + 1

    n_maps = n_clusts[0]

    color_endpoints = COLORS[:n_maps]
    
    label_to_color = {}
    for label, agg in label_to_hierarchy.items():
        if len(agg) == 1:
            label_to_color[label] = LinearSegmentedColormap.from_list('a', ['white', color_endpoints[label]], N=100)(.8)
        elif len(agg) == 2:
            n_colors = n_clusts[1]
            val = (agg[-1] + 1) / n_colors
            label_to_color[label] = LinearSegmentedColormap.from_list('a', ['white', color_endpoints[agg[0]]], N=100)(val)
        else:
            n_colors = np.product(n_clusts[1:])
            arr = np.arange(n_colors)
            for i in range(1, len(agg) - 1):
                x, max_c = agg[i], n_clusts[i + 1]
                arr = arr[x * max_c:(x + 1) * max_c]
            idx = arr[agg[-1]]
            val = (idx + 1) / n_colors
            label_to_color[label] = LinearSegmentedColormap.from_list('a', ['white', color_endpoints[agg[0]]], N=100)(val)
    label_to_color = {k:v[:3] for k, v in label_to_color.items()}
    return label_to_color

def display_labeled_as_rgb(labeled, cmap=None, preserve_indices=True, label_to_hierarchy=None, discard_max=False):
    if isinstance(labeled, torch.Tensor):
        labeled = labeled.numpy()
    
    if label_to_hierarchy is not None:
        cmap = get_hierarchical_cmap(label_to_hierarchy)
    elif preserve_indices:
        cmap = get_cmap(labeled.max() + 1) if cmap is None else cmap
    else:
        cmap = get_cmap(len(np.unique(labeled))) if cmap is None else cmap
    
    ids, counts = np.unique(labeled, return_counts=True, )
    max_label = ids[counts.argmax()]

    labels = sorted(np.unique(labeled))
    if len(cmap) < len(labels):
        raise RuntimeError('cmap is too small')
    new = np.zeros((labeled.shape[0], labeled.shape[1], 3))
    for i, l in enumerate(labels):
        if preserve_indices:
            c = cmap[l]
        else:
            c = cmap[i]
        
        if discard_max:
            if l == max_label:
                c = [0., 0., 0.]
        new[labeled==l] = c
    return new


def display_clusters(clusters, cmap=None, figsize=None, horizontal=True, preserve_indices=False, return_axs=False, label_to_hierarchy=None, discard_max=False):
    if figsize is None:
        figsize = (clusters.shape[0] * 2, 5)
        if not horizontal:
            figsize = (figsize[1], figsize[0])

    if horizontal:
        fig, axs = plt.subplots(ncols=clusters.shape[0] + 1, figsize=figsize)
    else:
        fig, axs = plt.subplots(nrows=clusters.shape[0] + 1, figsize=figsize)

    if cmap is None:
        if label_to_hierarchy is not None:
            cmap = get_hierarchical_cmap(label_to_hierarchy)
        else:
            cmap = get_cmap(len(np.unique(clusters)))
    elif isinstance(cmap, str):
        cmap = sns.color_palette(cmap)
    
    # if discard_max:
    #     ids, counts = np.unique(clusters, return_counts=True, )
    #     l = ids[counts.argmax()]
    #     cmap[l] = [1., 1., 1.]

    for i, labeled in enumerate(clusters):
        axs[i].imshow(display_labeled_as_rgb(labeled, cmap=cmap, preserve_indices=preserve_indices, label_to_hierarchy=label_to_hierarchy, discard_max=discard_max))
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    display_legend(np.unique(clusters), cmap, ax=axs[-1], label_to_hierarchy=label_to_hierarchy)
    axs[-1].axis('off')

    if return_axs:
        return axs


def display_legend(labels, cmap, ax=None, label_to_hierarchy=None):
    if ax is None:
        fig, ax = plt.subplots()

    if label_to_hierarchy is not None:
        xs = [agg + (l,) for l, agg in label_to_hierarchy.items()]
        order = [x[-1] for x in sorted(xs)]
        ax.legend(
            handles=[mpatches.Patch(color=cmap[label], label=label)
                    for label in order],
            loc='center'
        )
    else:
        ax.legend(
            handles=[mpatches.Patch(color=color, label=label)
                    for label, color in zip(labels, cmap)],
            loc='center'
        )
    # return ax


def show_groups(clusters, groups):
    mapping = {i:i for i in np.unique(clusters)}
    mapping.update({i:-1 for i in np.unique(clusters) if i not in groups})
    neigh_ids = np.vectorize(mapping.get)(clusters)
    display_clusters(neigh_ids)




def volume_to_gif(volume, is_probs, filepath, axis=0, duration=200):
    if is_probs:
        imgs = [repeat(img, 'h w -> h w c', c=3)
                    for img in volume.swapaxes(0, axis)]
    else:
        imgs = [display_labeled_as_rgb(labeled, preserve_indices=True)
                for labeled in volume.swapaxes(0, axis)]
    imgs = [((img / img.max()) * 255.).astype(np.uint8) for img in imgs]

    ims = [Image.fromarray(x, mode="RGB")
                for x in imgs]
    ims[0].save(fp=filepath, format='GIF', append_images=ims,
             save_all=True, duration=duration, loop=0)
    
def save_reference_gif(rgb, filepath, axis=0, duration=200, color=(1., 0., 0.), thickness=1):
    if not isinstance(color, np.ndarray):
        color = np.asarray(color)
    if rgb.max() <= 1.:
        rgb /= rgb.max()
        rgb *= 255.
        rgb = rgb.astype(np.uint8)
        color = (color * 255.).astype(np.uint8)
    
    n_frames = rgb.shape[axis]
    imgs = []
    for frame in range(n_frames):
        selections = tuple([slice(None) if i!=axis else slice(i, i+thickness) for i in range(n_frames)])
        selections = tuple([slice(None) if i!=axis else slice(frame, frame + thickness) for i in range(len(rgb.shape))])
        new = rgb.copy()
        new[selections] = color
        imgs.append(new)
        
    ims = [Image.fromarray(x, mode="RGB")
                for x in imgs]
    ims[0].save(fp=filepath, format='GIF', append_images=ims,
             save_all=True, duration=duration, loop=0)
    



def to_cmapped_rgb_continuous(mask, tiled, idx=0, thresh=.5,
                              boundary_dist=1, external_dist=4, method='region',
                              cmap=None, min_area=4, vmax=1., as_border=False):
    img = tiled[idx]
    
    if cmap is None:
        cmap = sns.color_palette('viridis', n_colors=int(100 * vmax) + 1)
    else:
        cmap = sns.color_palette(cmap, n_colors=int(100 * vmax) + 1)
    
    labeled = skimage.morphology.label(mask)
    print(labeled.shape, img.shape)
    props = regionprops(labeled, img)
    props = [p for p in props if p.area >= min_area]
    
    blank = np.ones((labeled.shape[0], labeled.shape[1], 4), dtype=np.float32)
    blank[..., -1] = 0.
    meta = {}
    for p in props:
        r1, c1, r2, c2 = p.bbox
        r1 = max(0, r1 - external_dist)
        c1 = max(0, c1 - external_dist)
        r2 = min(labeled.shape[0] - 1, r2 + external_dist)
        c2 = min(labeled.shape[1] - 1, c2 + external_dist)
        
        initial = labeled[r1:r2, c1:c2] == p.label
        
        eroded = initial.copy()
        for i in range(int(boundary_dist)):
            eroded = binary_erosion(eroded)
        expanded = initial.copy()
        for i in range(int(boundary_dist)):
            expanded = skimage.morphology.binary_dilation(expanded)
        boundary_mask = expanded ^ eroded
        
        expanded = initial.copy()
        for i in range(int(external_dist)):
            expanded = skimage.morphology.binary_dilation(expanded)
        external_mask = expanded ^ initial
        
        region_img = tiled[:, r1:r2, c1:c2]
        
        boundary_means = region_img[:, boundary_mask].mean(-1)
        internal_means = region_img[:, initial].mean(-1)
        external_means = region_img[:, external_mask].mean(-1)

        if method == 'region':
            m = initial
        elif method == 'external':
            m = external_mask
        elif method == 'boundary':
            m = boundary_mask
        else:
            raise RuntimeError(f'invalid method {method}')

        pixels = region_img[idx, m]
        frac = np.count_nonzero(pixels > thresh) / len(pixels)
        
        color = cmap[min(int(vmax * 100), int(frac * 100))]
        color = np.asarray([*color, 1.])

        tile = blank[r1:r2, c1:c2]

        if not as_border:
            tile[initial] = color
        else:
            tile[boundary_mask] = color
        
        blank[r1:r2, c1:c2] = tile
        
        meta[p.label] = {
            'fraction': frac,
            'area': p.area,
            'centroid': p.centroid,
            'bbox_actual': (r1, c1, r2, c2),
            'prop': p,
            'boundary_means': boundary_means,
            'internal_means': internal_means,
            'external_means': external_means
        }

    return blank, meta, labeled

def to_stacked_rgb_continuous(
        config, dtype_to_masks, dtype_to_tiled, dtype_to_channel_idxs, dtype_to_thresholds,
        boundary_dist=1, external_dist=4,
        spacing_scaler=10, squish_scaler=2,
        cmap=None, vmax=1., method='region', min_area=4.,
        border_kwargs=None
    ):
    target_size = next(iter(dtype_to_masks.values()))[0].shape[:2]
    dtypes = dtype_to_masks.keys()
    
    positions, sids, dts = zip(*[(entry['position'], entry['sid'], entry['data'][0]['dtype'])
                                    for entry in config['sections']
                                    if entry['data'][0]['dtype'] in dtypes])
    
    sid_to_data = {}
    for dtype in dtypes:
        intensities = dtype_to_tiled[dtype]
        masks = dtype_to_masks[dtype]
        ps, ids = zip(*[(p, sid) for sid, p, dt in zip(sids, positions, dts) if dt == dtype])
        thresh = dtype_to_thresholds[dtype]
        for sid, mask, intensity, position in zip(ids, masks, intensities, ps):
            idx = dtype_to_channel_idxs[dtype]
            rgb, meta, labeled = to_cmapped_rgb_continuous(
                mask, intensity, idx=idx, thresh=thresh,
                boundary_dist=boundary_dist, external_dist=external_dist,
                cmap=cmap, vmax=vmax, min_area=min_area, method=method
            )

            if border_kwargs is not None:
                border_rgb, _, _ = to_cmapped_rgb_continuous(
                    mask, intensity,
                    boundary_dist=boundary_dist, external_dist=external_dist,
                    min_area=min_area, as_border=True, **border_kwargs
                )
                m = border_rgb[..., -1] == 1.
                rgb[m] = border_rgb[m]
            
            sid_to_data[sid] = {
                'dtype': dtype,
                'rgb': rgb,
                'position': position,
                'meta': meta,
            }

    blank = np.zeros((target_size[-2] // squish_scaler, target_size[-1], 4))

    blank = np.concatenate(
        (blank, np.zeros((max(positions) // spacing_scaler, blank.shape[1], blank.shape[2])))
    )

    for sid in sids:
        data = sid_to_data[sid]
        rgb = data['rgb']
        position = data['position'] // spacing_scaler
        size = (rgb.shape[0] // squish_scaler, rgb.shape[1])
        rgb = utils.rescale(rgb, size=size, dim_order='h w c')

        # pad to size of blank with a translation
        top = np.zeros((position, blank.shape[1], blank.shape[2]))
        bottom = np.zeros((blank.shape[0] - (position + rgb.shape[0]), blank.shape[1], blank.shape[2]))
        padded = np.concatenate((top, rgb, bottom))

        m = blank[..., -1] == 0
        blank[m] = padded[m]

    return blank, sid_to_data


def to_cmapped_rgb_category(mask, tiled, myoepi_idx=0, immune_idx=1,
                            myoepi_thresh=100, immune_thresh=100, area_thresh=100,
                            boundary_dist=1, external_dist=4, cat_to_color=None, min_area=4):
    
    immune_img = tiled[immune_idx]
    myoepi_img = tiled[myoepi_idx]
    
    cmap = sns.color_palette('tab10')
    possibles = [f'my{x}_im{y}_a{z}' for x in [True, False] for y in [True, False] for z in [True, False]]
    if cat_to_color is None:
        cat_to_color = {x:c for x, c in zip(possibles, cmap)}
    
    labeled = skimage.morphology.label(mask)
    props = regionprops(labeled, myoepi_img)
    props = [p for p in props if p.area >= min_area]
    
    blank = np.ones((labeled.shape[0], labeled.shape[1], 4), dtype=np.float32)
    blank[..., -1] = 0.
    meta = {}
    for p in props:
        r1, c1, r2, c2 = p.bbox
        r1 = max(0, r1 - external_dist)
        c1 = max(0, c1 - external_dist)
        r2 = min(labeled.shape[0] - 1, r2 + external_dist)
        c2 = min(labeled.shape[1] - 1, c2 + external_dist)
        
        initial = labeled[r1:r2, c1:c2] == p.label
        
        eroded = initial.copy()
        for i in range(int(boundary_dist)):
            eroded = binary_erosion(eroded)
        expanded = initial.copy()
        for i in range(int(boundary_dist)):
            expanded = skimage.morphology.binary_dilation(expanded)
        boundary_mask = expanded ^ eroded
        
        expanded = initial.copy()
        for i in range(int(external_dist)):
            expanded = skimage.morphology.binary_dilation(expanded)
        external_mask = expanded ^ initial
        
        region_img = tiled[:, r1:r2, c1:c2]
        
        boundary_means = region_img[:, boundary_mask].mean(-1)
        internal_means = region_img[:, initial].mean(-1)
        external_means = region_img[:, external_mask].mean(-1)
        
#         print(myoepi_idx, boundary_means)

        pixels = region_img[myoepi_idx, boundary_mask]
        myoepi_mean = np.count_nonzero(pixels > myoepi_thresh) / len(pixels)
        
        pixels = region_img[immune_idx, external_mask]
        immune_mean = np.count_nonzero(pixels > immune_thresh) / len(pixels)
        
        is_myoepi = myoepi_mean > .25
        is_immune = immune_mean > .20
        
        
#         myoepi_mean = boundary_means[myoepi_idx]
#         immune_mean = external_means[immune_idx]
        
#         is_myoepi = myoepi_mean > myoepi_thresh
#         is_immune = immune_mean > immune_thresh
        is_area = p.area > area_thresh
        
        cat = f'my{is_myoepi}_im{is_immune}_a{is_area}'
        color = cat_to_color[cat]
        color = np.asarray([*color, 1.])

        tile = blank[r1:r2, c1:c2]
        tile[initial] = color
        blank[r1:r2, c1:c2] = tile
        
        meta[p.label] = {
            'category': cat,
            'is_myoepi': is_myoepi,
            'is_immune': is_immune,
            'is_area': is_area,
            'myoepi_mean': myoepi_mean,
            'immune_mean': immune_mean,
            'area': p.area,
            'centroid': p.centroid,
            'bbox_actual': (r1, c1, r2, c2),
            'prop': p,
            'boundary_means': boundary_means,
            'internal_means': internal_means,
            'external_means': external_means
        }

    return blank, meta, labeled, cat_to_color

def to_stacked_rgb_category(
        config, dtype_to_masks, dtype_to_tiled, dtype_to_channel_idxs, dtype_to_thresholds,
        boundary_dist=1, external_dist=4,
        spacing_scaler=10, squish_scaler=2,
        cat_to_color=None
    ):
    target_size = next(iter(dtype_to_masks.values()))[0].shape[:2]
    dtypes = dtype_to_masks.keys()
    
    positions, sids, dts = zip(*[(entry['position'], entry['sid'], entry['data'][0]['dtype'])
                                    for entry in config['sections']
                                    if entry['data'][0]['dtype'] in dtypes])
    
    sid_to_data = {}
    for dtype in dtypes:
        intensities = dtype_to_tiled[dtype]
        masks = dtype_to_masks[dtype]
        ps, ids = zip(*[(p, sid) for sid, p, dt in zip(sids, positions, dts) if dt == dtype])
        thresholds = dtype_to_thresholds[dtype]
        for sid, mask, intensity, position in zip(ids, masks, intensities, ps):
            myoepi_idx = dtype_to_channel_idxs[dtype]['myoepi']
            immune_idx = dtype_to_channel_idxs[dtype]['immune']
            rgb, meta, labeled, _ = to_cmapped_rgb_category(
                mask, intensity, myoepi_idx=myoepi_idx, immune_idx=immune_idx,
                myoepi_thresh=thresholds['myoepi_thresh'],
                immune_thresh=thresholds['immune_thresh'],
                area_thresh=thresholds['area_thresh'],
                boundary_dist=boundary_dist, external_dist=external_dist,
                cat_to_color=cat_to_color
            )
            sid_to_data[sid] = {
                'dtype': dtype,
                'rgb': rgb,
                'position': position,
                'meta': meta,
                'cat_to_color': cat_to_color
            }

    blank = np.zeros((target_size[-2] // squish_scaler, target_size[-1], 4))

    blank = np.concatenate(
        (blank, np.zeros((max(positions) // spacing_scaler, blank.shape[1], blank.shape[2])))
    )

    for sid in sids:
        data = sid_to_data[sid]
        rgb = data['rgb']
        position = data['position'] // spacing_scaler
        size = (rgb.shape[0] // squish_scaler, rgb.shape[1])
        rgb = utils.rescale(rgb, size=size, dim_order='h w c')

        # pad to size of blank with a translation
        top = np.zeros((position, blank.shape[1], blank.shape[2]))
        bottom = np.zeros((blank.shape[0] - (position + rgb.shape[0]), blank.shape[1], blank.shape[2]))
        padded = np.concatenate((top, rgb, bottom))

        m = blank[..., -1] == 0
        blank[m] = padded[m]

    return blank, sid_to_data

      