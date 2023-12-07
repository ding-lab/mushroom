import logging
import os
from typing import Any, Optional
import warnings
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from vit_pytorch import ViT
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback

from mushroom.model.sae import SAE, SAEargs
from mushroom.visualization.utils import display_labeled_as_rgb

logger = logging.getLogger()
logger.setLevel(logging.INFO)
warnings.simplefilter(action='ignore', category=FutureWarning)


class WandbImageCallback(Callback):
    def __init__(self, wandb_logger, learner_data, inference_dl, channel=None):
        self.logger = wandb_logger
        self.inference_dl = inference_dl
        self.learner_data = learner_data
        self.channel = channel if channel is not None else self.learner_data.channels[0]


    def on_train_epoch_end(self, trainer, pl_module):
        outputs = []

        with torch.no_grad():
            for batch in self.inference_dl:
                tiles, slides, dtypes = batch['tiles'], batch['slides'], batch['dtypes']
                tiles, slides, dtypes = tiles.to(pl_module.device), slides.to(pl_module.device), dtypes.to(pl_module.device)
                outs = pl_module.forward(tiles, slides, dtypes)
                outputs.append(outs)
    
        formatted = pl_module.format_prediction_outputs(outputs)
        predicted_pixels = formatted['predicted_pixels'].cpu().clone().detach().numpy()
        true_pixels = formatted['true_pixels'].cpu().clone().detach().numpy()
        clusters = formatted['clusters'].cpu().clone().detach().numpy().astype(int)

        self.logger.log_image(
            key=f'predicted pixels {self.channel}',
            images=[img[self.learner_data.channels.index(self.channel)] for img in predicted_pixels],
            caption=[str(i) for i in range(len(predicted_pixels))]
        )
        self.logger.log_image(
            key=f'true pixels {self.channel}',
            images=[img[self.learner_data.channels.index(self.channel)] for img in true_pixels],
            caption=[str(i) for i in range(len(true_pixels))]
        )
        self.logger.log_image(
            key='predicted pixels first section',
            images=[img for img in predicted_pixels[0]],
            caption=self.learner_data.channels
        )
        self.logger.log_image(
            key='clusters',
            images=[display_labeled_as_rgb(labeled, preserve_indices=True) for labeled in clusters],
            caption=[str(i) for i in range(len(clusters))]
        )


class LitMushroom(LightningModule):
    def __init__(
            self,
            sae_args,
            learner_data,
            lr=1e-4,
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
            neigh_scaler=sae_args.neigh_scaler
        )

        self.outputs = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        tiles, slides, dtypes = batch['tiles'], batch['slides'], batch['dtypes']
        pairs, is_anchor = batch['pairs'], batch['is_anchor']
        outs = self.forward(tiles, slides, dtypes, pairs=pairs, is_anchor=is_anchor)
        self.log_dict({k:v for k, v in outs.items() if k!='outputs'}, on_step=True, on_epoch=False, prog_bar=True)
        return outs
    
    def predict_step(self, batch):
        tiles, slides, dtypes = batch['tiles'], batch['slides'], batch['dtypes']
        return self.forward(tiles, slides, dtypes)
    
    def format_prediction_outputs(self, outputs):
        n = len(self.learner_data.inference_ds)
        num_patches = self.sae_args.size // self.sae_args.patch_size

        cluster_ids = torch.zeros(n, num_patches, num_patches)
        cluster_probs = torch.zeros(n, num_patches, num_patches, self.sae_args.codebook_size)

        dtype_to_pred_patches = {dtype: for dtype, n_channels in self.learner_data.dtype_to_n_channels.items()}
        dtype_to_true_patches = {dtype:torch.zeros(n, n_channels, num_patches, num_patches) for dtype, n_channels in self.learner_data.dtype_to_n_channels.items()}

        bs = outputs[0]['outputs']['clusters'].shape[0]

        for i, output in enumerate(outputs):
            dtype_to_pred_pixel_values = {dtype:rearrange(
                pred_pixels, 'b (h w) c -> b c h w',
                h=num_patches, w=num_patches,
                c=self.learner_data.dtype_to_n_channels[dtype])
                for dtype, pred_pixels in output['outputs']['dtype_to_pred_pixels'].items()}
            dtype_to_true_pixel_values = {dtype:rearrange(
                true_pixels, 'b (h w) c -> b c h w',
                h=num_patches, w=num_patches,
                c=self.learner_data.dtype_to_n_channels[dtype])
                for dtype, true_pixels in output['outputs']['dtype_to_true_pixels'].items()}
            clusters = rearrange(
                output['outputs']['clusters'], 'b (h w) -> b h w',
                h=num_patches, w=num_patches
            )
            probs = rearrange(
                output['outputs']['cluster_probs'], 'b (h w) d -> b h w d',
                h=num_patches, w=num_patches
            )
            start, stop = i * bs, (i * bs) + output['outputs']['clusters'].shape[0]
            for dtype in dtype_to_pred_pixel_values.keys():
                dtype_to_pred_patches[dtype][start:stop] = dtype_to_pred_pixel_values[dtype].cpu().detach()
                true_patches[start:stop] = true_pixel_values.cpu().detach()
            cluster_ids[start:stop] = clusters.cpu().detach()
            cluster_probs[start:stop] = probs.cpu().detach()

        predicted_pixels = torch.stack(
            [self.learner_data.inference_ds.section_from_tiles(
                pred_patches, i, size=(num_patches, num_patches)
            ) for i in range(len(self.learner_data.inference_ds.section_ids))]
        )
        true_pixels = torch.stack(
            [self.learner_data.inference_ds.section_from_tiles(
                true_patches, i, size=(num_patches, num_patches)
            ) for i in range(len(self.learner_data.inference_ds.section_ids))]
        )
        recon_cluster_ids = torch.stack(
            [self.learner_data.inference_ds.section_from_tiles(
                rearrange(cluster_ids, 'n h w -> n 1 h w'),
                i, size=(num_patches, num_patches))
                for i in range(len(self.learner_data.inference_ds.section_ids))] 
        ).squeeze(1)
        recon_cluster_probs = torch.stack(
            [self.learner_data.inference_ds.section_from_tiles(
                rearrange(cluster_probs, 'n h w c -> n c h w'),
                i, size=(num_patches, num_patches))
                for i in range(len(self.learner_data.inference_ds.section_ids))] 
        )

        return {
            'predicted_pixels': predicted_pixels,
            'true_pixels': true_pixels,
            'clusters': recon_cluster_ids,
            'cluster_probs': recon_cluster_probs,
        }
    
    def forward(self, tiles, slides, dtypes, pairs=None, is_anchor=None):
        # if pos_x is not None:
        #     x = torch.concat((anchor_x, pos_x))
        #     slide = torch.concat((anchor_slide, pos_slide))
        #     dtype = torch.concat((anchor_dtype, pos_dtype))
        # else:
        #     x = anchor_x
        #     slide = anchor_slide
        #     dtype = anchor_dtype

        # pairs, is_anchor 
        # your dataloader wont work due to different sizes

        losses, outputs = self.sae(tiles, slides, dtypes, pairs=pairs, is_anchor=is_anchor)
        losses['loss'] = losses['overall_loss']
        losses['outputs'] = outputs

        return losses
