from dataclasses import dataclass
from typing import Iterable

import napari
import numpy as np
import pandas as pd
import seaborn as sns
import torch

@dataclass
class NapariImageArgs:
    name: str
    position: int
    dtype: str
    scale_factor: float
    channels: Iterable = None
    img: np.ndarray = None
    spots: Iterable = None


def to_uint8(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    x = x.astype(np.float32)
    # x -= x.min()
    
    x /= x.max()
    x *= 255.
    x = x.astype(np.uint8)
    return x

def display_napari(napari_args_list, volume):
    viewer = napari.Viewer(ndisplay=3)
    
    # volume = volume.max() - volume
    volume = torch.nn.functional.softmax(volume, dim=0)
    volume = to_uint8(volume)
    for cluster in range(volume.shape[0]):
        x = volume[cluster]
        viewer.add_image(
            x,
            depiction='volume',
            rendering='iso',
            iso_threshold=10.,
            name=f'cluster {cluster}',
            visible=False
        )
    
    for args in napari_args_list:
        if args.dtype == 'multiplex':
            x = to_uint8(args.img)
            channel = args.channels[0]
            name = args.name
            viewer.add_image(
                x,
                name=f'{name}_{channel}',
                visible=False,
                opacity=.3,
                translate=[args.position, 0, 0],
                scale=[1, args.scale_factor, args.scale_factor]
            )
        if args.dtype == 'visium':
            # x = to_uint8(args.img)
            name = args.name
            point_properties = {'value': [item['value'] if not pd.isnull(item['value']) else 0. for item in args.spots]}
            pts = np.asarray([[args.position, item['y'], item['x']] for item in args.spots])
            channel = args.channels[0]
            viewer.add_points(
                pts,
                properties=point_properties,
                face_color='value',
                face_colormap='viridis',
                edge_color='value',
                edge_colormap='viridis',
                name=f'{name}_{channel}',
                size=10,
                blending='opaque',
                # editable=False,
                visible=False,
                # translate=[args.position, 0, 0],
                scale=[1, args.scale_factor, args.scale_factor],
            )
            # viewer.add_image(
            #     x,
            #     name=name,
            #     visible=False,
            #     opacity=.3,
            #     translate=[args.position, 0, 0],
            #     scale=[1, args.scale_factor, args.scale_factor]
            # )
        if args.dtype == 'he':
            x = to_uint8(args.img)
            name = args.name
            viewer.add_image(
                x,
                name=name,
                visible=False,
                opacity=.3,
                translate=[args.position, 0, 0],
                scale=[1, args.scale_factor, args.scale_factor]
            )

    
    