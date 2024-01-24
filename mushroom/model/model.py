import logging
import os
from typing import Any, Optional
import warnings
from pytorch_lightning.utilities.types import STEP_OUTPUT

import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from vit_pytorch import ViT
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback

from mushroom.model.sae import SAE, SAEargs
from mushroom.visualization.utils import display_labeled_as_rgb
import mushroom.utils as utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
warnings.simplefilter(action='ignore', category=FutureWarning)


class WandbImageCallback(Callback):
    def __init__(self, wandb_logger, learner_data, inference_dl, channel=None):
        self.logger = wandb_logger
        self.inference_dl = inference_dl
        self.learner_data = learner_data
        # self.channel = channel if channel is not None else self.learner_data.channels[0]
        self.channel = 0


    def on_train_epoch_end(self, trainer, pl_module):
        outputs = []

        with torch.no_grad():
            for batch in self.inference_dl:
                tiles, slides, dtypes = batch['tiles'], batch['slides'], batch['dtypes']
                tiles = [x.to(pl_module.device) for x in tiles]
                slides = [x.to(pl_module.device) for x in slides]
                dtypes = [x.to(pl_module.device) for x in dtypes]
                outs = pl_module.forward(tiles, slides, dtypes)

                # do this properly eventually
                outs['outputs']['dtype_to_true_pixels'] = {k:v.cpu() for k, v in outs['outputs']['dtype_to_true_pixels'].items()}
                outs['outputs']['dtype_to_pred_pixels'] = {k:v.cpu() for k, v in outs['outputs']['dtype_to_pred_pixels'].items()}

                outputs.append(outs)

        formatted = pl_module.format_prediction_outputs(outputs)
        predicted_pixels = [[z.cpu().clone().detach().numpy() for z in x] for x in formatted['predicted_pixels']]
        true_pixels = [x.cpu().clone().detach().numpy() for x in formatted['true_pixels']]
        clusters = [x for x in formatted['clusters']]

        for level, imgs in enumerate(predicted_pixels):
            self.logger.log_image(
                key=f'predicted pixels {level} {self.channel}',
                images=[img[..., 0] for img in imgs],
                caption=[str(i) for i in range(len(imgs))]
            )
        
        self.logger.log_image(
            key=f'true pixels {self.channel}',
            images=[img[..., 0] for img in true_pixels],
            caption=[str(i) for i in range(len(true_pixels))]
        )
        for level, cs in enumerate(clusters):
            # print([np.unique(c) for c in cs])
            self.logger.log_image(
                key=f'clusters {level}',
                images=[display_labeled_as_rgb(labeled, preserve_indices=True) for labeled in cs],
                caption=[str(i) for i in range(len(cs))]
            )

class VariableTrainingCallback(Callback):
    def __init__(self, end_pretraining_at=5):
        self.end_pretraining_at = end_pretraining_at

    def on_train_epoch_end(self, trainer, pl_module):
        if self.end_pretraining_at == pl_module.current_epoch:
            pass
            # logging.info(f'stopping pretraining at {self.end_pretraining_at}')
            # pl_module.sae.end_pretraining()
                # pass
                # pl_module.sae.freeze_
                # print(f'stoppint pretraining level {self.pretrain_for}')
                # pl_module.sae.end_pretraining()


class LitMushroom(LightningModule):
    def __init__(
            self,
            sae_args,
            learner_data,
            lr=1e-4,
            total_steps=1,
            ):
        super().__init__()
        self.image_size = sae_args.size
        self.patch_size = sae_args.patch_size
        self.lr = lr
        self.learner_data = learner_data
        self.sae_args = sae_args

        logging.info('creating ViT')
        encoder = ViT(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_classes=sae_args.num_classes,
            dim=sae_args.encoder_dim,
            depth=sae_args.encoder_depth,
            heads=sae_args.heads,
            mlp_dim=sae_args.mlp_dim,
        )

        self.sae = SAE(
            encoder=encoder,
            n_slides=len(self.learner_data.train_ds.section_ids),
            dtypes=self.learner_data.dtypes,
            dtype_to_n_channels=self.learner_data.dtype_to_n_channels,
            codebook_dim=self.sae_args.codebook_dim,
            dtype_to_decoder_dims=self.sae_args.dtype_to_decoder_dims,
            recon_scaler=sae_args.recon_scaler,
            neigh_scaler=sae_args.neigh_scaler,
            total_steps=total_steps
        )

        self.outputs = None

    def _flatten_outputs(self, outputs):
        ds = self.learner_data.inference_ds
        flat_dtypes = [ds.section_ids[sid][1] for sid, *_ in ds.idx_to_coord]
        n_levels = len(outputs[0]['outputs']['level_to_encoded'])
        batch_size = len(outputs[0]['outputs']['level_to_encoded'])
        
        flat = {}
        for k in ['encoded_tokens_prequant']:
            flat[k] = torch.concat([x['outputs'][k][:, 2:] for x in outputs])# skip slide and dtype token
        
        for k in ['level_to_encoded', 'cluster_probs', 'clusters']:
            for level in range(n_levels):
                flat[f'{k}_{level}'] = torch.concat([x['outputs'][k][level] for x in outputs])
                
        k = 'dtype_to_true_pixels'
        pool = [v for v in flat_dtypes]
        flat['true_pixels'] = []
        spot = 0
        for i, x in enumerate(outputs):
            batch_size = len(x['outputs']['encoded_tokens_prequant'])
            dtypes = pool[spot:spot + batch_size]
            spot += batch_size

            dtype_to_idx = {dtype:0 for dtype in sorted(set(dtypes))}
            for dtype in dtypes:
                idx = dtype_to_idx[dtype]
                obj = x['outputs'][k][dtype][idx]
                flat['true_pixels'].append(obj)
                dtype_to_idx[dtype] += 1
                    
        k = 'dtype_to_pred_pixels' # refactor this
        for level in range(n_levels):
            pool = [v for v in flat_dtypes]
            flat[f'pred_pixels_{level}'] = []
            spot = 0
            for i, x in enumerate(outputs):
                batch_size = len(x['outputs']['encoded_tokens_prequant'])
                dtypes = pool[spot:spot + batch_size]
                spot += batch_size

                dtype_to_idx = {dtype:0 for dtype in sorted(set(dtypes))}
                for dtype in dtypes:
                    idx = dtype_to_idx[dtype]
                    obj = x['outputs'][k][dtype][level, idx]
                    flat[f'pred_pixels_{level}'].append(obj)
                    dtype_to_idx[dtype] += 1

        return flat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        tiles, slides, dtypes = batch['tiles'], batch['slides'], batch['dtypes']
        pairs, is_anchor = batch['pairs'], batch['is_anchor']
        outs = self.forward(tiles, slides, dtypes, pairs=pairs, is_anchor=is_anchor)
        outs['neigh_scaler'] = self.sae.variable_neigh_scaler.get_scaler()
        batch_size = sum([v.shape[1] for k, v in outs['outputs']['dtype_to_true_pixels'].items()]) // 2
        self.log_dict({f'{k}_step':v for k, v in outs.items() if k!='outputs'}, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log_dict({f'{k}_epoch':v for k, v in outs.items() if k!='outputs'}, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return outs
    
    def predict_step(self, batch):
        tiles, slides, dtypes = batch['tiles'], batch['slides'], batch['dtypes']
        return self.forward(tiles, slides, dtypes)
    
    def format_prediction_outputs(self, outputs):
        ds = self.learner_data.inference_ds
        n_levels = len(outputs[0]['outputs']['level_to_encoded'])

        flat = self._flatten_outputs(outputs)
        clusters = [torch.stack(
            [ds.section_from_tiles(
                flat[f'clusters_{level}'].unsqueeze(-1), i
            ).squeeze(-1) for i in range(len(ds.section_ids))]
        ).to(torch.long) for level in range(n_levels)]
        cluster_probs = [torch.stack(
            [ds.section_from_tiles(
                flat[f'cluster_probs_{level}'], i
            ) for i in range(len(ds.section_ids))]
        ) for level in range(n_levels)]
        pred_pixels = [
            [ds.section_from_tiles(
                flat[f'pred_pixels_{level}'], i
            ) for i in range(len(ds.section_ids))]
        for level in range(n_levels)]
        true_pixels = [ds.section_from_tiles(
                flat['true_pixels'], i
            ) for i in range(len(ds.section_ids))]
        
        relabeled_clusters = [utils.label_agg_clusters(clusters[:i + 1]) for i in range(len(clusters))]

        return {
            'predicted_pixels': pred_pixels, # nested list of (h, w, c), length num levels, length num sections
            'true_pixels': true_pixels, # list of (h, w, c), length num sections
            'clusters': relabeled_clusters, # list of (n, h, w), length num levels
            'cluster_probs': cluster_probs, # list of (n, h, w, n_clusters), length num levels
            'agg_clusters': clusters, # list of (n h w), length num levels
        }

    def forward(self, tiles, slides, dtypes, pairs=None, is_anchor=None):
        losses, outputs = self.sae(tiles, slides, dtypes, pairs=pairs, is_anchor=is_anchor)

        if 'overall_loss' in losses:
            losses['loss'] = losses['overall_loss']
        losses['outputs'] = outputs

        return losses
