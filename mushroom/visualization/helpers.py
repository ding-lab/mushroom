

import numpy as np
import pandas as pd
import seaborn as sns
import skimage
from einops import rearrange
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
import skimage.measure

from mushroom.utils import rescale



def to_masks(imgs, thresh, erosion=None):
    masks = []
    for img in imgs:
        mask = img > thresh
        mask = binary_fill_holes(mask)
        
        if erosion is not None:
            for i in range(erosion):
                mask = binary_erosion(mask)
        
        masks.append(mask)
    masks = np.stack(masks)
    return masks

def quantify_labeled(labeled, img, channels, boundary_dist=1, external_dist=4, channel_info=None):
    """
    labeled - (h, w)
    img - (c, h, w)
    channels - (c,)
    channel_info - {
        channel_name: {
            'idx': 
            'thresh': 
        }
    }
    """
    props = skimage.measure.regionprops(labeled, rearrange(img, 'c h w -> h w c'))
    data = []
    for p in props:
        r1, c1, r2, c2 = p.bbox
        r1 = max(0, r1 - external_dist)
        c1 = max(0, c1 - external_dist)
        r2 = min(labeled.shape[0] - 1, r2 + external_dist)
        c2 = min(labeled.shape[1] - 1, c2 + external_dist)
        columns = ['area', 'r1', 'r2', 'c1', 'c2', 'centroid_r', 'centroid_c']
        row = [p.area, r1, r2, c1, c2, p.centroid[0], p.centroid[1]]
        
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
        
        region_img = img[:, r1:r2, c1:c2]

        mask_dict = {
            'boundary': boundary_mask,
            'region': initial,
            'external': external_mask
        }
        
        boundary_means = region_img[:, mask_dict['boundary']].mean(-1)
        internal_means = region_img[:, mask_dict['region']].mean(-1)
        external_means = region_img[:, mask_dict['external']].mean(-1)

        columns += [f'{channel}_boundary' for channel in channels]
        row += boundary_means.tolist()

        columns += [f'{channel}_region' for channel in channels]
        row += internal_means.tolist()

        columns += [f'{channel}_external' for channel in channels]
        row += external_means.tolist()

        if channel_info is not None:
            for name, d in channel_info.items():
                idx, thresh = channels.index(d['channel']), d['thresh']
                for method, mask in mask_dict.items():
                    pixels = region_img[idx, mask]
                    if len(pixels):
                        frac = np.count_nonzero(pixels > thresh) / len(pixels)
                    else:
                        frac = np.nan
                    row.append(frac)
                    columns.append(f'{name}_{method}_fraction')
        data.append(row)
    df = pd.DataFrame(data=data, columns=columns)
    return df, props


def get_cmap(n, cmap=None):
    if n < 10:
        default = sns.color_palette()
    elif n < 60:
        default = sns.color_palette('tab20') + sns.color_palette('tab20b') + sns.color_palette('tab20c')
    else:
        default = sns.color_palette('hsv', n_colors=n)
    cmap = sns.color_palette(cmap, n_colors=n) if cmap is not None else default
    return cmap

def to_cmapped_rgb(labeled, region_df, hue_labels=None, hue=None, label_col='label', cmap=None, vmax=None, borders=False, n=100, force_categorical=False):
    if hue is None:
        region_df['hue'] = 'all'
        hue = 'hue'

    value = region_df[hue][0]
    if isinstance(value, (int, float, complex)) and not force_categorical:
        is_categorical = False
        cmap = sns.color_palette(cmap, n_colors=n + 1) if cmap is not None else 'viridis'
        bins = np.linspace(region_df[hue].min(), region_df[hue].max() if vmax is None else vmax, n)
        values = np.digitize(region_df[hue], bins)
        if vmax is not None:
            max_value = np.digitize(vmax, bins)
            print(max_value)
    else:
        is_categorical = True
        pool = sorted(set(region_df[hue])) if hue_labels is None else hue_labels

        if isinstance(cmap, str) or cmap is None:
            cmap = get_cmap(len(pool), cmap=cmap)
    
        values = region_df[hue].to_list()
    
    # props = skimage.measure.regionprops(labeled)
    rgba = np.ones((labeled.shape[0], labeled.shape[1], 4), dtype=np.float32)
    rgba[..., -1] = 0.

    # assert len(props) == region_df.shape[0]
    for (i, row), value in zip(region_df.iterrows(), values):
        if pd.isnull(value):
            continue

        r1, c1, r2, c2 = row['r1'], row['c1'], row['r2'], row['c2']

        if is_categorical:
            value = pool.index(value)
        else:
            if vmax is not None:
                value = min(max_value, value)
        


        color = cmap[value]
        color = np.asarray([*color, 1.])

        tile = rgba[r1:r2, c1:c2]
        initial = labeled[r1:r2, c1:c2] == row[label_col]

        tile[initial] = color
        if borders:
            eroded = binary_erosion(initial)
            expanded = binary_dilation(initial)
            boundary = expanded ^ eroded
            tile[boundary] = [.8, .8, .8, 1.]
        
        rgba[r1:r2, c1:c2] = tile
    
    return rgba


def to_stacked_rgb(
        sids, to_position, to_labeled, region_df, hue=None,
        spacing_scaler=10, squish_scaler=2, force_categorical=False,
    ):
    positions = [to_position[k] for k in sids]
    target_size = next(iter(to_labeled.values())).shape
    blank = np.zeros((target_size[-2] // squish_scaler, target_size[-1], 4))
    blank = np.concatenate(
        (blank, np.zeros((max(positions) // spacing_scaler, blank.shape[1], blank.shape[2])))
    )

    value = region_df[hue][0]
    if isinstance(value, str) or force_categorical:
        pool = sorted(set(region_df[hue]))
    else:
        pool = None

    for k in sids:
        labeled = to_labeled[k]
        position = to_position[k]
        df = region_df[region_df['sid']==k]

        if df.shape[0]:
            rgba = to_cmapped_rgb(
                labeled, df, hue=hue, hue_labels=pool , label_col='label',
                force_categorical=force_categorical, borders=False
            )
        else:
            rgba = np.zeros((labeled.shape[0], labeled.shape[1], 4))

        # pad to size of blank with a translation
        position //= spacing_scaler
        size = (rgba.shape[0] // squish_scaler, rgba.shape[1])
        rgba = rescale(rgba, size=size, dim_order='h w c')

        top = np.zeros((position, blank.shape[1], blank.shape[2]))
        bottom = np.zeros((blank.shape[0] - (position + rgba.shape[0]), blank.shape[1], blank.shape[2]))
        padded = np.concatenate((top, rgba, bottom))

        m = blank[..., -1] == 0
        blank[m] = padded[m]

    return blank