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

import mushroom.data.xenium as xenium
import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
from mushroom.model.sae import SAE, SAEargs

logger = logging.getLogger()
logger.setLevel(logging.INFO)
warnings.simplefilter(action='ignore', category=FutureWarning)


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
        self.log_dict({k:v for k, v in outs.items() if k!='outputs'}, on_step=False, on_epoch=True, prog_bar=True)
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



    # def embed_sections(self, device=None):
    #     device = device if device is not None else self.device
    #     self.sae = self.sae.to(device)

    #     if self.sae is None:
    #         raise RuntimeError('Must train model prior to embedding sections.')

    #     n = len(self.inference_ds)
    #     num_patches = self.size[0] // self.sae_args.patch_size
    #     embs = torch.zeros(
    #         n, num_patches, num_patches, self.sae.encoder.pos_embedding.shape[-1])
    #     pred_patches = torch.zeros(n, len(self.channels), num_patches, num_patches)
    #     cluster_ids = torch.zeros(n, num_patches, num_patches)
    #     cluster_probs = torch.zeros(n, num_patches, num_patches, self.sae_args.codebook_size)

    #     bs = self.inference_dl.batch_size
    #     self.sae.eval()
    #     with torch.no_grad():
    #         for i, b in enumerate(self.inference_dl):
    #             x, slide = b['tile'], b['idx']
    #             x, slide = x.to(device), slide.to(device)
    #             prequant_tokens, mu, std = self.sae.encode(x, slide, use_means=True)
    #             encoded_tokens, clusters, probs = self.sae.quantize(prequant_tokens)

    #             pred_pixel_values = self.sae.decode(encoded_tokens, slide)

    #             encoded_tokens = rearrange(encoded_tokens[:, 1:], 'b (h w) d -> b h w d',
    #                                     h=num_patches, w=num_patches)
    #             pred_pixel_values = rearrange(
    #                 pred_pixel_values, 'b (h w) c -> b c h w',
    #                 h=num_patches, w=num_patches,
    #                 c=len(self.channels))
    #             clusters = rearrange(clusters, 'b (h w) -> b h w',
    #                                     h=num_patches, w=num_patches)
    #             probs = rearrange(probs, 'b (h w) d -> b h w d',
    #                                     h=num_patches, w=num_patches)

    #             embs[i * bs:(i + 1) * bs] = encoded_tokens.cpu().detach()
    #             pred_patches[i * bs:(i + 1) * bs] = pred_pixel_values.cpu().detach()
    #             cluster_ids[i * bs:(i + 1) * bs] = clusters.cpu().detach()
    #             cluster_probs[i * bs:(i + 1) * bs] = probs.cpu().detach()

    #     recon_imgs = torch.stack(
    #         [self.inference_ds.section_from_tiles(
    #             pred_patches, i, size=(num_patches, num_patches)
    #         ) for i in range(len(self.inference_ds.sections))]
    #     )
    #     recon_embs = torch.stack(
    #         [self.inference_ds.section_from_tiles(
    #             rearrange(embs, 'n h w c -> n c h w'),
    #             i, size=(num_patches, num_patches))
    #             for i in range(len(self.inference_ds.sections))] 
    #     )
    #     recon_cluster_ids = torch.stack(
    #         [self.inference_ds.section_from_tiles(
    #             rearrange(cluster_ids, 'n h w -> n 1 h w'),
    #             i, size=(num_patches, num_patches))
    #             for i in range(len(self.inference_ds.sections))] 
    #     ).squeeze(1)
    #     recon_cluster_probs = torch.stack(
    #         [self.inference_ds.section_from_tiles(
    #             rearrange(cluster_probs, 'n h w c -> n c h w'),
    #             i, size=(num_patches, num_patches))
    #             for i in range(len(self.inference_ds.sections))] 
    #     )

    #     return recon_imgs, recon_embs, recon_cluster_ids, recon_cluster_probs