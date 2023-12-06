import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Normalize, RandomCrop, Compose

import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
import mushroom.data.xenium as xenium

DTYPES = ('multiplex', 'xenium', 'visium',)


def get_multiplex_section_to_img(config, scale, channels=None, channel_mapping=None, contrast_pct=None):
    sid_to_filepaths = {
        entry['id']:d['filepath'] for entry in config for d in entry['data']
        if d['dtype']=='multiplex'
    }
    section_ids = [entry['id'] for entry in config
                   if 'multiplex' in [d['dtype'] for d in entry['data']]]

    if channels is None:
        fps = [d['filepath'] for entry in config for d in entry['data']
                if d['dtype']=='multiplex']
        channels = multiplex.get_common_channels(fps, channel_mapping=channel_mapping)
    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info('processing sections')
    section_to_img = multiplex.get_section_to_image(
        sid_to_filepaths, channels, channel_mapping=channel_mapping, scale=scale, contrast_pct=contrast_pct)
    
    means = torch.cat(
        [x.mean(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    stds = torch.cat(
        [x.std(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    normalize = Normalize(means, stds)
    
    return section_to_img, normalize

def get_xenium_section_to_img(
        config, scale, size, patch_size,
        channels=None, channel_mapping=None, fullres_size=None,
    ):
    sid_to_filepaths = {
        entry['id']:d['filepath'] for entry in config for d in entry['data']
        if d['dtype']=='xenium'
    }
    section_ids = [entry['id'] for entry in config
                   if 'xenium' in [d['dtype'] for d in entry['data']]]

    if channels is None:
        fps = [d['filepath'] for entry in config for d in entry['data']
                if d['dtype']=='xenium']
        channels = xenium.get_common_channels(
            fps, channel_mapping=channel_mapping
        )
    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info(f'processing sections')
    section_to_img = {
        sid:xenium.to_multiplex(xenium.adata_from_xenium(fp, normalize=True), tiling_size=tiling_size)
        for sid, fp in sid_to_filepaths.items()
    }

    means = torch.cat(
        [x.mean(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    stds = torch.cat(
        [x.std(dim=(-2, -1)).unsqueeze(0) for x in section_to_img.values()]
    ).mean(0)
    normalize = Normalize(means, stds)

    return section_to_img, normalize


def get_visium_section_to_img(
        config, scale, size, patch_size,
        channels=None, channel_mapping=None, fullres_size=None, pct_expression=.02
    ):
    sid_to_filepaths = {
        entry['id']:d['filepath'] for entry in config for d in entry['data']
        if d['dtype']=='visium'
    }
    section_ids = [entry['id'] for entry in config
                   if 'visium' in [d['dtype'] for d in entry['data']]]

    if channels is None:
        fps = [d['filepath'] for entry in config for d in entry['data']
                if d['dtype']=='visium']
        channels = visium.get_common_channels(
            fps, channel_mapping=channel_mapping, pct_expression=pct_expression
        )
    logging.info(f'using {len(channels)} channels')
    logging.info(f'{len(section_ids)} sections detected: {section_ids}')

    logging.info(f'processing sections')
    section_to_adata = {sid:visium.adata_from_visium(fp, normalize=True) for sid, fp in sid_to_filepaths.items()}
    section_to_img, section_to_adata = visium.get_section_to_image( # labeled image where pixels represent location of barcodes, is converted by transform to actual exp image
        section_to_adata, channels, patch_size=patch_size, channel_mapping=channel_mapping, scale=scale, fullres_size=fullres_size
    )

     # TODO: find a cleaner way to do this, is long because trying to avoid explicit sparse matrix conversion of .X
    means = np.asarray(np.vstack(
        [a.X.mean(0) for a in section_to_adata.values()]
    ).mean(0)).squeeze()
    stds = np.asarray(np.vstack(
        [a.X.std(0) for a in section_to_adata.values()]
    ).mean(0)).squeeze()
    normalize = Normalize(means, stds)

    return section_to_img, section_to_adata, normalize



def get_learner_data(config, scale, size, patch_size, channels=None, channel_mapping=None, contrast_pct=None, fullres_size=None, pct_expression=.02):


    train_transform = MultiplexTrainingTransform(size=size, patch_size=patch_size, normalize=normalize)
    inference_transform = InferenceTransform(normalize)

    logging.info('generating training dataset')
    train_ds = MultiplexSectionDataset(
        section_ids, section_to_img, transform=train_transform
    )
    logging.info('generating inference dataset')
    inference_ds = InferenceSectionDataset(
        section_ids, section_to_img, transform=inference_transform, size=size
    )

    learner_data = LearnerData(
        dtype_to_section_to_img=dtype_to_section_to_img,
        train_transform=train_transform,
        inference_transform=inference_transform,
        train_ds=train_ds,
        inference_ds=inference_ds,
        channels=channels
    )

    return learner_data