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
    def __init__(self, wandb_logger, learner_data, inference_dl, true_pixels, channel=None):
        # super.__init__()
        self.logger = wandb_logger
        self.inference_dl = inference_dl
        self.learner_data = learner_data
        self.true_pixels = true_pixels
        self.channel = channel if channel is not None else self.learner_data.channels[0]


    def on_train_epoch_end(self, trainer, pl_module):
        # return None
        # outputs = trainer.predict(pl_module, self.inference_dl)
        # return None
        outputs = []

        with torch.no_grad():
            for batch in self.inference_dl:
                x, slide = batch['tile'], batch['idx']
                x, slide = x.to(pl_module.device), slide.to(pl_module.device)
                outs = pl_module.forward(x, slide)
                outputs.append(outs)
    
        formatted = pl_module.format_prediction_outputs(outputs)
        predicted_pixels = formatted['predicted_pixels'].cpu().clone().detach().numpy()
        clusters = formatted['clusters'].cpu().clone().detach().numpy().astype(int)

        self.logger.log_image(
            key=f'predicted pixels {self.channel}',
            images=[img[self.learner_data.channels.index(self.channel)] for img in predicted_pixels],
            caption=[str(i) for i in range(len(predicted_pixels))]
        )
        self.logger.log_image(
            key=f'true pixels {self.channel}',
            images=[img[self.learner_data.channels.index(self.channel)] for img in self.true_pixels],
            caption=[str(i) for i in range(len(self.true_pixels))]
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
        self.n_slides = len(learner_data.section_to_img)
        self.n_channels = len(learner_data.channels)
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
            channels=len(learner_data.channels),
        )

        self.sae = SAE(
            encoder=encoder,
            n_slides=len(learner_data.section_to_img),
            n_channels=len(learner_data.channels),
            decoder_dims=sae_args.decoder_dims,
            codebook_size=sae_args.codebook_size,
            kl_scaler=sae_args.kl_scaler,
            recon_scaler=sae_args.recon_scaler,
            neigh_scaler=sae_args.neigh_scaler
        )

        self.outputs = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        anchor_x, anchor_slide = batch['anchor_tile'], batch['anchor_idx']
        pos_x, pos_slide = batch['pos_tile'], batch['pos_idx']
        outs = self.forward(anchor_x, anchor_slide, pos_x=pos_x, pos_slide=pos_slide)
        self.log_dict({k:v for k, v in outs.items() if k!='outputs'}, on_step=True, on_epoch=False, prog_bar=True)
        return outs
    
    def predict_step(self, batch):
        x, slide = batch['tile'], batch['idx']
        return self.forward(x, slide)
    
    def format_prediction_outputs(self, outputs):
        n = len(self.learner_data.inference_ds)
        num_patches = self.sae_args.size // self.sae_args.patch_size

        cluster_ids = torch.zeros(n, num_patches, num_patches)
        cluster_probs = torch.zeros(n, num_patches, num_patches, self.sae_args.codebook_size)
        pred_patches = torch.zeros(n, self.n_channels, num_patches, num_patches)

        bs = outputs[0]['outputs']['clusters'].shape[0]

        for i, output in enumerate(outputs):
            pred_pixel_values = rearrange(
                output['outputs']['pred_pixel_values'], 'b (h w) c -> b c h w',
                h=num_patches, w=num_patches,
                c=self.n_channels)
            clusters = rearrange(
                output['outputs']['clusters'], 'b (h w) -> b h w',
                h=num_patches, w=num_patches
            )
            probs = rearrange(
                output['outputs']['cluster_probs'], 'b (h w) d -> b h w d',
                h=num_patches, w=num_patches
            )
            start, stop = i * bs, (i * bs) + output['outputs']['clusters'].shape[0]
            pred_patches[start:stop] = pred_pixel_values.cpu().detach()
            cluster_ids[start:stop] = clusters.cpu().detach()
            cluster_probs[start:stop] = probs.cpu().detach()

        recon_imgs = torch.stack(
            [self.learner_data.inference_ds.section_from_tiles(
                pred_patches, i, size=(num_patches, num_patches)
            ) for i in range(len(self.learner_data.inference_ds.sections))]
        )
        recon_cluster_ids = torch.stack(
            [self.learner_data.inference_ds.section_from_tiles(
                rearrange(cluster_ids, 'n h w -> n 1 h w'),
                i, size=(num_patches, num_patches))
                for i in range(len(self.learner_data.inference_ds.sections))] 
        ).squeeze(1)
        recon_cluster_probs = torch.stack(
            [self.learner_data.inference_ds.section_from_tiles(
                rearrange(cluster_probs, 'n h w c -> n c h w'),
                i, size=(num_patches, num_patches))
                for i in range(len(self.learner_data.inference_ds.sections))] 
        )

        return {
            'predicted_pixels': recon_imgs,
            'clusters': recon_cluster_ids,
            'cluster_probs': recon_cluster_probs
        }
    
    def forward(self, anchor_x, anchor_slide, pos_x=None, pos_slide=None):
        if pos_x is not None:
            x = torch.concat((anchor_x, pos_x))
            slide = torch.concat((anchor_slide, pos_slide))
        else:
            x = anchor_x
            slide = anchor_slide

        losses, outputs = self.sae(x, slide)
        losses['loss'] = losses['overall_loss']
        losses['outputs'] = outputs

        return losses
