from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import repeat, rearrange
from vector_quantize_pytorch import VectorQuantize


@dataclass
class SAEargs:
    size: int = 256
    patch_size: int = 32
    encoder_dim: int = 1024
    decoder_dims: Iterable = (1024, 1024, 1024 * 4,)
    encoder_depth: int = 6
    heads: int = 8
    mlp_dim: int = 2048
    num_classes: int = 1000
    codebook_size: int = 30
    kl_scaler: float = .001
    recon_scaler: float = 1.
    neigh_scaler: float = 1.


class SAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        n_slides,
        n_channels,
        codebook_size = 100,
        decoder_dims = (1024, 1024, 1024 * 4,),
        kl_scaler = .001,
        recon_scaler = 1.,
        neigh_scaler = .1,
    ):
        super().__init__()
        self.kl_scaler = kl_scaler
        self.recon_scaler = recon_scaler
        self.neigh_scaler = neigh_scaler

        self.encoder = encoder
        self.decoder_dims = decoder_dims
        self.n_channels = n_channels
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.num_patches = num_patches

        self.n_slides = n_slides
        self.slide_embedding = nn.Embedding(self.n_slides, encoder_dim)

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        # pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        blocks = []
        for i, dim in enumerate(decoder_dims):
            if i == 0:
                block = nn.Sequential(
                    nn.Linear(encoder_dim, dim),
                    nn.ReLU()
                )
            else:
                block = nn.Sequential(
                    nn.Linear(decoder_dims[i - 1], dim),
                    nn.ReLU()
                )
            blocks.append(block)
        blocks.append(nn.Sequential(
            nn.Sequential(
                nn.Linear(decoder_dims[-1], n_channels),
            )
        ))
        self.decoder = nn.Sequential(*blocks)

        self.latent_mu = nn.Linear(encoder_dim, encoder_dim)
        self.latent_var = nn.Linear(encoder_dim, encoder_dim)
        self.latent_norm = nn.BatchNorm1d(num_patches)

        self.to_logits = nn.Linear(encoder_dim, codebook_size)

        self.codebook = nn.Parameter(torch.rand(codebook_size, encoder_dim))


    def _kl_divergence(self, z, mu, std):
        # lightning imp.
        # Monte carlo KL divergence
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)

        return kl


    def encode(self, img, slides):
        device = img.device

        # get patches
        patches = self.to_patch(img) # b (h w) (p1 p2 c)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)

        # add slide emb
        slide_tokens = self.slide_embedding(slides) # b d

        slide_tokens = rearrange(slide_tokens, 'b d -> b 1 d')
        slide_tokens += self.encoder.pos_embedding[:, :1]

        tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # add slide token
        tokens = torch.cat((slide_tokens, tokens), dim=1)

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        return encoded_tokens

    
    def quantize(self, encoded_tokens):
        # vector quantize
        slide_token, patch_tokens = encoded_tokens[:, :1], encoded_tokens[:, 1:]

        logits = self.to_logits(patch_tokens)
        hots = F.gumbel_softmax(logits, dim=-1, hard=True)
        probs = F.softmax(logits, dim=-1)
        encoded = hots @ self.codebook

        encoded_tokens = torch.cat((slide_token, encoded), dim=1)

        return encoded_tokens, probs.argmax(dim=-1), probs
    
    def decode(self, encoded_tokens, slide):
        
        pred_pixel_values = self.decoder(encoded_tokens[:, 1:])

        return pred_pixel_values


    def forward(self, img, slide, use_means=False):
        """
        """
        outputs = []

        encoded_tokens_prequant  = self.encode(img, slide)

        encoded_tokens, clusters, probs = self.quantize(encoded_tokens_prequant)

        pred_pixel_values = self.decode(encoded_tokens, slide) # (b, n, channels)

        # (b c h w)
        patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                            p1=self.to_patch.axes_lengths['p1'], p2=self.to_patch.axes_lengths['p2'])
        patches = patches.mean(-2)

        recon_loss = F.mse_loss(pred_pixel_values, patches) # (b, n, channels)

        outputs = {
            'encoded_tokens_prequant': encoded_tokens_prequant,
            'encoded_tokens': encoded_tokens,
            'pred_pixel_values': pred_pixel_values,
            'cluster_probs': probs,
            'clusters': clusters,
        }

        anchor_probs, pos_probs = probs.chunk(2)
        anchor_clusters, pos_clusters = clusters.chunk(2)
        token_idxs = torch.randint(anchor_probs.shape[1], (anchor_probs.shape[0],))

        pred = anchor_probs[torch.arange(anchor_probs.shape[0]), token_idxs]
        target = pos_clusters[torch.arange(pos_clusters.shape[0]), token_idxs]
        neigh_loss = F.cross_entropy(pred, target)

        # pred = anchor_embs[torch.arange(anchor_embs.shape[0]), token_idxs]
        # target = pos_embs[torch.arange(pos_embs.shape[0]), token_idxs]
        # neigh_loss = F.cosine_embedding_loss(pred, target, torch.ones(anchor_embs.shape[0], device=anchor_embs.device))

        overall_loss = recon_loss * self.recon_scaler + neigh_loss * self.neigh_scaler

        losses = {
            'overall_loss': overall_loss,
            'recon_loss': recon_loss,
            'neigh_loss': neigh_loss,
        }

        return losses, outputs