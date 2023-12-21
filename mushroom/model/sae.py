from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import repeat, rearrange
from scipy.interpolate import interp1d
from vector_quantize_pytorch import VectorQuantize


@dataclass
class SAEargs:
    size: int = 8
    patch_size: int = 1
    encoder_dim: int = 256
    codebook_dim: int = 64
    dtype_to_decoder_dims: Mapping = MappingProxyType({'multiplex': (256, 128, 64,), 'visium': (256, 512, 1024 * 2,), 'xenium': (256, 256, 256,)})
    encoder_depth: int = 6
    heads: int = 8
    mlp_dim: int = 2048
    num_classes: int = 1000
    num_clusters: Iterable = (8, 4, 2,)
    recon_scaler: float = 1.
    neigh_scaler: float = 1.
    level_scalers: Iterable = (.8, .4, .2,)



def get_decoder(in_dim, decoder_dims, n_channels):
    blocks = []
    for i, dim in enumerate(decoder_dims):
        if i == 0:
            block = nn.Sequential(
                nn.Linear(in_dim, dim),
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
    return nn.Sequential(*blocks)

class VariableScaler(object):
    def __init__(self, max_value, total_steps=1, min_value=0.0, function='log'):

        if function == 'linear':
            self.values = np.linspace(min_value, max_value, total_steps)
        elif function == 'log':
            x = np.linspace(0, 50, total_steps)
            x = np.log1p(x + 1)
            self.values = interp1d([x.min(), x.max()],[min_value, max_value])(x)
        elif function == 'ramp_up':
            self.values = np.linspace(min_value, max_value, total_steps // 2)
            self.values = np.concatenate([self.values, np.full((total_steps // 2 + 1,), max_value)])
        elif function == 'constant':
            self.values = np.full((total_steps,), max_value)

        self.idx = 0

    def get_scaler(self):
        if self.idx < len(self.values):
            return self.values[self.idx]
        return self.values[-1]

    def step(self):
        self.idx += 1


class SAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        n_slides,
        dtypes,
        dtype_to_n_channels, # mapping, keys are dtypes, values are n_channels
        codebook_dim = 64,
        num_clusters = (8, 4, 2,),
        dtype_to_decoder_dims = {'multiplex': (256, 128, 64,), 'visium': (256, 512, 1024 * 2,), 'xenium': (256, 256, 256,)},
        recon_scaler = 1.,
        neigh_scaler = .1,
        level_scalers = (1., 1., 1.,),
        total_steps = 1,
    ):
        super().__init__()
        self.recon_scaler = recon_scaler
        # self.neigh_scaler = 0.
        self.neigh_scaler = neigh_scaler
        self.variable_neigh_scaler = VariableScaler(self.neigh_scaler, total_steps=total_steps, min_value=0.)

        self.num_clusters = num_clusters
        self.codebook_dim = codebook_dim
        self.level_scalers = [x if i == 0 else 0. for i, x in enumerate(level_scalers)]

        self.dtypes = dtypes
        self.encoders = torch.nn.ModuleList([deepcopy(encoder) for i in range(len(num_clusters))])
        self.num_patches, self.encoder_dim = encoder.pos_embedding.shape[-2:]
        self.num_patches -= 1 # don't count slide embedding
        for encoder in self.encoders:
            encoder.pos_embedding  = nn.Parameter(torch.randn(1, self.num_patches + 2, self.encoder_dim)) # need to add token for slide and dtype embedding  

        self.dtype_to_decoder_dims = dtype_to_decoder_dims
        self.dtype_to_n_channels = dtype_to_n_channels

        self.n_slides = n_slides
        self.slide_embedding = nn.Embedding(self.n_slides, self.encoder_dim)
        self.dtype_embedding = nn.Embedding(len(self.dtypes), self.encoder_dim)

        # self.to_patch = encoder.to_patch_embedding[0]
        self.patch_dim = next(iter(encoder.to_patch_embedding[1].parameters())).shape[0]
        p1, p2 = encoder.to_patch_embedding[0].axes_lengths['p1'], encoder.to_patch_embedding[0].axes_lengths['p2']
        self.dtype_to_patch = nn.ModuleDict({
            dtype: nn.Sequential(
                encoder.to_patch_embedding[0],
                nn.Linear(n * p1 * p2, self.encoder_dim),
            )
            for dtype, n in dtype_to_n_channels.items()
        })
        # self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        self.dtype_to_decoder = nn.ModuleDict({
            dtype:get_decoder(codebook_dim + self.encoder_dim, dims, dtype_to_n_channels[dtype])
            for dtype, dims in self.dtype_to_decoder_dims.items()
            if dtype in dtype_to_n_channels
        })

        self.level_to_logits = nn.ModuleList([nn.Linear(self.encoder_dim, n) for n in self.num_clusters])

        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(self.num_clusters[0], codebook_dim))])
        if len(self.num_clusters) > 1:
            total = 1
            for n in self.num_clusters[1:]:
                total *= n
                dim = codebook_dim * total
                self.codebooks.append(nn.Parameter(torch.randn(self.num_clusters[0], dim)))

    def _to_quantized(self, hots):
        b, n, d = hots[0].shape[0], hots[0].shape[1], self.codebook_dim
        results = []
        for i, c in enumerate(self.num_clusters):
            if i == 0:
                encoded = hots[i] @ self.codebooks[i]
            elif i == 1:
                z = hots[0] @ self.codebooks[i]
                z = rearrange(z, 'b n (c d) -> b n c d', c=c, d=d)
                encoded = torch.einsum('bncd,bnc->bnd', z, hots[i])
            else:
                z = hots[0] @ self.codebooks[i]

                for j in range(2, i+1):
                    a = np.product(self.num_clusters[j:i+1])
                    z = rearrange(z, 'b n (c a d) -> b n c (a d)', c=self.num_clusters[j-1], a=a, d=d)
                    z = torch.einsum('bncd,bnc->bnd', z, hots[j-1])

                z = rearrange(z, 'b n (c d) -> b n c d', c=c, d=d)
                encoded = torch.einsum('bncd,bnc->bnd', z, hots[i])
            results.append(encoded)
        return results
    
    def freeze_all_except_codebooks(self):
        modules = [
            self.encoder,
            self.dtype_to_patch,
            self.dtype_to_decoder,
            self.level_to_logits

        ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        params = [
            self.slide_embedding,
            self.dtype_embedding
        ]
        for param in params:
                param.requires_grad = False

    def freeze_cluster_level(self, level):
        modules = [
            self.encoders[level],
            self.level_to_logits[level],
            self.dtype_to_patch,
        ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        params = [
            self.slide_embedding,
            self.dtype_embedding,
            self.codebooks[level]
        ]
        for param in params:
                param.requires_grad = False
       
    def encode(self, imgs, slides, dtypes):
        level_to_encoded_tokens = []
        for encoder in self.encoders:
            patches = []
            for img, dtype in zip(imgs, dtypes):
                # get patches
                name = self.dtypes[dtype[0]]
                patches.append(self.dtype_to_patch[name](img)) # b (h w) (p1 p2 c)

            all_tokens = torch.concat(patches, dim=0)
            all_slides = torch.concat(slides, dim=0)
            all_dtypes = torch.concat(dtypes, dim=0)

            # patches = self.patch_to_emb(patches)

            batch, num_patches, *_ = all_tokens.shape

            # # patch to encoder tokens
            # tokens = self.patch_to_emb(patches)

            # add slide emb
            slide_tokens = self.slide_embedding(all_slides) # b d
            dtype_tokens = self.dtype_embedding(all_dtypes) # b d

            slide_tokens = rearrange(slide_tokens, 'b d -> b 1 d')
            dtype_tokens = rearrange(dtype_tokens, 'b d -> b 1 d')

            slide_tokens += encoder.pos_embedding[:, :1]
            dtype_tokens += encoder.pos_embedding[:, 1:2]
            all_tokens += encoder.pos_embedding[:, 2:(num_patches + 2)]

            # add slide and dtype token
            tokens = torch.cat((slide_tokens, dtype_tokens, all_tokens), dim=1)

            # attend with vision transformer
            encoded_tokens = encoder.transformer(tokens)

            level_to_encoded_tokens.append(encoded_tokens)
        
        return level_to_encoded_tokens

    
    def quantize(self, level_to_encoded_tokens):
        level_to_logits, level_to_hots, level_to_probs = [], [], []
        for encoded_tokens, to_logits in zip(level_to_encoded_tokens, self.level_to_logits):
            slide_token, dtype_token, patch_tokens = encoded_tokens[:, :1], encoded_tokens[:, 1:2], encoded_tokens[:, 2:]

            logits = to_logits(patch_tokens)
            hots = F.gumbel_softmax(logits, dim=-1, hard=True)
            probs = F.softmax(logits, dim=-1)

            level_to_logits.append(logits)
            level_to_hots.append(hots)
            level_to_probs.append(probs)
        
        level_to_encoded = self._to_quantized(level_to_hots)
        level_to_encoded = torch.stack(level_to_encoded)

        return level_to_encoded, level_to_hots, [probs.argmax(dim=-1) for probs in level_to_probs], level_to_probs
    
    def decode(self, level_to_encoded, slide, dtype):
        slide_embs = self.slide_embedding(slide) # b d
        slide_embs = rearrange(slide_embs, 'b d -> 1 b 1 d')
        slide_embs = repeat(slide_embs, 'a b c d -> (a a1) b (c c1) d',
                            a1=level_to_encoded.shape[0], c1=level_to_encoded.shape[2])
        level_to_encoded = torch.concat((level_to_encoded, slide_embs), dim=-1)

        dtype_to_pred_pixels = {}
        for label in dtype.unique():
            name = self.dtypes[label]
            mask = dtype==label
            dtype_to_pred_pixels[name] = self.dtype_to_decoder[name](level_to_encoded[:, mask])

        return dtype_to_pred_pixels


    def forward(self, imgs, slides, dtypes, pairs=None, is_anchor=None):
        """
        imgs - (dtypes, b, c, h, w) where dtypes is number of different dtypes in batch
        slides - (dtypes, b,)
        pairs - (dtypes, b,)
        dtypes - (dtypes, b,), all entries are same value for each dtype
        """
        outputs = []

        level_to_encoded_tokens_prequant  = self.encode(imgs, slides, dtypes)

        level_to_encoded, level_to_hots, level_to_clusters, level_to_probs = self.quantize(level_to_encoded_tokens_prequant)

        flat_slides = torch.concat(slides, dim=0)
        flat_dtypes = torch.concat(dtypes, dim=0)
        dtype_to_pred_pixels = self.decode(level_to_encoded, flat_slides, flat_dtypes) # returns dtype:(level, b, n, channels) where b and channels vary based on dtype

        # (b c h w)
        dtype_to_true_pixels = {}
        for img, dtype in zip(imgs, dtypes):
            name = self.dtypes[dtype[0]]
            p1 = self.dtype_to_patch[name][0].axes_lengths['p1']
            p2 = self.dtype_to_patch[name][0].axes_lengths['p2']
            patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                      p1=p1, p2=p2)
            dtype_to_true_pixels[name] = patches.mean(-2)

        outputs = {
            'level_to_encoded': level_to_encoded,
            'cluster_probs': level_to_probs,
            'clusters': level_to_clusters,
            'dtype_to_true_pixels': dtype_to_true_pixels,
            'dtype_to_pred_pixels': dtype_to_pred_pixels,
        }

        if pairs is not None and is_anchor is not None:
            dtype_to_recon_loss = {}
            for name in self.dtypes:
                for level, pixel_level in enumerate(dtype_to_pred_pixels[name]):
                    dtype_to_recon_loss[f'{level}_{name}'] = F.mse_loss(pixel_level, dtype_to_true_pixels[name])

            flat_pairs = torch.concat(pairs, dim=0)
            flat_is_anchor = torch.concat(is_anchor, dim=0)
            level_to_neigh_loss = []
            level_to_frac_loss = []
            for level, (clusters, probs, hots) in enumerate(zip(level_to_clusters, level_to_probs, level_to_hots)):

                anchor_mask = flat_is_anchor
                pos_mask = ~flat_is_anchor

                anchor_clusters, anchor_probs, anchor_hots = clusters[anchor_mask], probs[anchor_mask], hots[anchor_mask]
                anchor_order = torch.argsort(flat_pairs[anchor_mask])
                pos_clusters, pos_probs, pos_hots = clusters[pos_mask], probs[pos_mask], hots[pos_mask]
                pos_order = torch.argsort(flat_pairs[pos_mask])

                anchor_clusters, pos_clusters = anchor_clusters[anchor_order], pos_clusters[pos_order]
                anchor_probs, pos_probs = anchor_probs[anchor_order], pos_probs[pos_order]
                anchor_hots, pos_hots = anchor_hots[anchor_order], pos_hots[pos_order]

                # anchor_sums, pos_sums = anchor_hots.sum(1), pos_hots.sum(1)
                # anchor_sums /= anchor_sums.sum(-1).unsqueeze(-1)
                # pos_sums /= pos_sums.sum(-1).unsqueeze(-1)
                # # level_to_frac_loss.append(F.mse_loss(anchor_sums, pos_sums))
                # level_to_neigh_loss.append(F.mse_loss(anchor_sums, pos_sums))


                # level_to_neigh_loss.append(F.cross_entropy(anchor_hots, pos_hots))

                # level_to_neigh_loss.append(F.cross_entropy(anchor_probs, pos_probs))

                token_idxs = torch.randint(anchor_probs.shape[1], (anchor_probs.shape[0],))

                pred = anchor_probs[torch.arange(anchor_probs.shape[0]), token_idxs]
                target = pos_clusters[torch.arange(pos_clusters.shape[0]), token_idxs]
                level_to_neigh_loss.append(F.cross_entropy(pred, target))


            # count all dtypes the same for now
            losses = {}
            neigh_total, recon_total = 0, 0
            # neigh_scaler = self.neigh_scaler
            neigh_scaler = self.variable_neigh_scaler.get_scaler()
            self.variable_neigh_scaler.step()
            dtype_to_scaler = {
                'multiplex': 2.,
                'xenium': 1.
            }
            for level, scaler in enumerate(self.level_scalers):
                losses[f'neigh_loss_level_{level}'] = level_to_neigh_loss[level]
                neigh_total += level_to_neigh_loss[level] * scaler
                for name in self.dtypes:
                    losses[f'recon_loss_{level}_{name}'] = dtype_to_recon_loss[f'{level}_{name}']
                    recon_total += dtype_to_recon_loss[f'{level}_{name}'] * scaler * dtype_to_scaler[name]
            losses['recon_loss'] = recon_total
            losses['neigh_loss'] = neigh_total

            overall = recon_total * self.recon_scaler + neigh_total * neigh_scaler
            losses['overall_loss'] = overall
        else:
            losses = {}

        return losses, outputs
