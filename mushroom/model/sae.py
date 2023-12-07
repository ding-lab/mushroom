from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import repeat, rearrange
from vector_quantize_pytorch import VectorQuantize


@dataclass
class SAEargs:
    size: int = 8
    patch_size: int = 1
    encoder_dim: int = 256
    codebook_dim: int = 64
    dtypes: Iterable = ('multiplex', 'visium', 'xenium',)
    dtype_to_n_channels: dict = {'multiplex':50, 'visium':10000, 'xenium':512}
    dtype_to_decoder_dims: dict = {'multiplex': (256, 128, 64,), 'visium': (256, 512, 1024 * 2,), 'xenium': (256, 256, 256,)}
    encoder_depth: int = 6
    heads: int = 8
    mlp_dim: int = 2048
    num_classes: int = 1000
    num_clusters: Iterable = (8, 4, 2,)
    recon_scaler: float = 1.
    neigh_scaler: float = 1.



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

def intermediate_op(hots, z, b, n, d, c=None):
    if c is not None:
        zz = hots @ z
        zz = rearrange(zz, 'b1 n1 (c b2 n2 d) -> b1 b2 n1 n2 c d', c=c, b2=b, n2=n, d=d)
        zz = torch.diagonal(torch.diagonal(zz))
        zz = rearrange(zz, 'c d b n -> c (b n d)')
    else:
        zz = hots @ z
        zz = rearrange(zz, 'b1 n1 (b2 n2 d) -> b1 b2 n1 n2 d', b2=b, n2=n, d=d)
        zz = torch.diagonal(torch.diagonal(zz))
        zz = rearrange(zz, 'd b n -> b n d')
    return zz

def to_quantized(num_clusters, logits, hots, codebooks):
    results = []
    for i, c in enumerate(cs):
        print(i, c)
        if i == 0:
            encoded = hots[i] @ codebooks[i]
        elif i == 1:
            z = hots[0] @ codebooks[i]
            z = rearrange(z, 'b n (c d) -> c (b n d)', c=c, d=d)
            encoded = intermediate_op(hots[i], z, b, n, d)
        else:
            z = hots[0] @ codebooks[i]

            for j in range(2, i+1):
                a = np.product(cs[j:i+1])
                print(z.shape)
                z = rearrange(z, 'b n (c a d) -> c (a b n d)', c=cs[j-1], a=a, d=d)
                z = intermediate_op(hots[j-1], z, b, n, d, c=a)

                if a != c:
                    z = rearrange(z, 'a (b n d) -> b n (a d)', b=b, n=n, d=d)

            encoded = intermediate_op(hots[i], z, b, n, d)
        results.append(encoded)
    return results




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
    ):
        super().__init__()
        self.recon_scaler = recon_scaler
        self.neigh_scaler = neigh_scaler
        self.num_clusters = num_clusters

        self.num_clusters = num_clusters
        self.dtypes = dtypes
        self.encoder = encoder
        self.num_patches, self.encoder_dim = encoder.pos_embedding.shape[-2:]
        self.num_patches -= 1 # don't count slide embedding
        self.encoder.pos_embedding  = nn.Parameter(torch.randn(1, self.num_patches + 2, self.encoder_dim)) # need to add token for slide and dtype embedding  

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
                nn.Linear(n * p1 * p2, self.patch_dim),
            )
            for dtype, n in dtype_to_n_channels.items()
        })
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        self.dtype_to_decoder = nn.ModuleDict({
            dtype:get_decoder(codebook_dim + self.slide_embedding.shape[-1], dims, dtype_to_n_channels[dtype])
            for dtype, dims in self.dtype_to_decoder_dims.items()
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
        b, n, d = hots.shape[0], hots.shape[1], self.codebook_dim
        results = []
        for i, c in enumerate(self.num_clusters):
            if i == 0:
                encoded = hots[i] @ self.codebooks[i]
            elif i == 1:
                z = hots[0] @ self.codebooks[i]
                z = rearrange(z, 'b n (c d) -> c (b n d)', c=c, d=d)
                encoded = intermediate_op(hots[i], z, b, n, d)
            else:
                z = hots[0] @ self.codebooks[i]

                for j in range(2, i+1):
                    a = np.product(self.num_clusters[j:i+1])
                    z = rearrange(z, 'b n (c a d) -> c (a b n d)', c=self.num_clusters[j-1], a=a, d=d)
                    z = intermediate_op(hots[j-1], z, b, n, d, c=a)

                    if a != c:
                        z = rearrange(z, 'a (b n d) -> b n (a d)', b=b, n=n, d=d)

                encoded = intermediate_op(hots[i], z, b, n, d)
            results.append(encoded)
        return results

       
    def encode(self, imgs, slides, dtypes):
        patches = []
        for img, dtype in zip(imgs, dtypes):
            # get patches
            name = self.dtypes[dtype[0]]
            patches.append(self.dtype_to_patch[name](img)) # b (h w) (p1 p2 c)

        patches = torch.concat(patches, dim=0)
        slides = torch.concat(slides, dim=0)
        dtypes = torch.concat(dtypes, dim=0)

        patches = self.patch_to_emb(patches)

        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens
        tokens = self.patch_to_emb(patches)

        # add slide emb
        slide_tokens = self.slide_embedding(slides) # b d
        dtype_tokens = self.dtype_embedding(dtypes) # b d

        slide_tokens = rearrange(slide_tokens, 'b d -> b 1 d')
        dtype_tokens = rearrange(dtype_tokens, 'b d -> b 1 d')

        slide_tokens += self.encoder.pos_embedding[:, :1]
        dtype_tokens += self.encoder.pos_embedding[:, 1:2]
        tokens += self.encoder.pos_embedding[:, 2:(num_patches + 2)]

        # add slide and dtype token
        tokens = torch.cat((slide_tokens, dtype_tokens, tokens), dim=1)

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        return encoded_tokens

    
    def quantize(self, encoded_tokens):
        slide_token, dtype_token, patch_tokens = encoded_tokens[:, :1], encoded_tokens[:, 1:2], encoded_tokens[:, 2:]

        level_to_logits, level_to_hots, level_to_probs = [], [], []
        for to_logits in self.level_to_logits:
            logits = to_logits(patch_tokens)
            hots = F.gumbel_softmax(logits, dim=-1, hard=True)
            probs = F.softmax(logits, dim=-1)

            level_to_logits.append(logits)
            level_to_hots.append(hots)
            level_to_probs.append(probs)
        
        level_to_encoded = self._to_quantized(level_to_hots)

        level_to_logits = torch.stack(level_to_logits) # l, b, n, d
        level_to_hots = torch.stack(level_to_hots)
        level_to_probs = torch.stack(level_to_probs)
        level_to_encoded = torch.stack(level_to_encoded)

        return level_to_encoded, level_to_probs.argmax(dim=-1), level_to_probs
    
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
            dtype_to_pred_pixels[name] = self.dtype_to_decoder[name](level_to_encoded[mask])

        return dtype_to_pred_pixels


    def forward(self, imgs, slides, dtypes, pairs, is_anchor):
        """
        imgs - list of (b, c, h, w)
        slides - list of (b,)
        pairs - list of (b,)
        dtypes - list of (b,), all entries for each list should have same value
        """
        outputs = []

        encoded_tokens_prequant  = self.encode(imgs, slides, dtypes)

        level_to_encoded, level_to_clusters, level_to_probs = self.quantize(encoded_tokens_prequant)

        flat_slides = torch.concat(slides, dim=0)
        flat_dtypes = torch.concat(dtypes, dim=0)
        dtype_to_pred_pixels = self.decode(level_to_encoded, flat_slides, flat_dtypes) # returns dtype:(level, b, n, channels) where b and channels vary based on dtype

        # (b c h w)
        dtype_to_true_pixels = {}
        for img, dtype in zip(imgs, dtypes):
            name = self.dtypes.index(dtype[0])
            p1 = self.dtype_to_patch[name][0].axes_lengths['p1']
            p2 = self.dtype_to_patch[name][0].axes_lengths['p2']
            patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                      p1=p1, p2=p2)
            dtype_to_true_pixels[name] = patches.mean(-2)

        outputs = {
            'encoded_tokens_prequant': encoded_tokens_prequant,
            'level_to_encoded': level_to_encoded,
            'cluster_probs': level_to_probs,
            'clusters': level_to_clusters,
            'dtype_to_true_pixels': dtype_to_true_pixels,
            'dtype_to_pred_pixels': dtype_to_pred_pixels,
        }

        dtype_to_recon_loss = {
            name:F.mse_loss(dtype_to_pred_pixels[name], dtype_to_true_pixels[name])
            for name in self.dtypes
        }

        flat_pairs = torch.concat(pairs, dim=0)
        flat_is_anchor = torch.concat(is_anchor, dim=0)
        dtype_to_neigh_loss = {}
        for i, name in enumerate(self.dtypes):
            dtype_mask = flat_dtypes==i

            clusters = level_to_clusters[:, dtype_mask]
            probs = level_to_probs[:, dtype_mask]

            anchor_mask = flat_is_anchor[dtype_mask]
            pos_mask = ~flat_is_anchor[dtype_mask]

            anchor_clusters, anchor_probs = clusters[:, anchor_mask], probs[:, anchor_mask]
            anchor_order = torch.argsort(flat_pairs[dtype_mask][anchor_mask])
            pos_clusters, pos_probs = clusters[:, pos_mask], probs[:, pos_mask]
            pos_order = torch.argsort(flat_pairs[dtype_mask][pos_mask])

            anchor_clusters, pos_clusters = anchor_clusters[:, anchor_order], pos_clusters[:, pos_order]
            anchor_probs, pos_probs = anchor_probs[:, anchor_order], pos_probs[:, pos_order]

            token_idxs = torch.randint(anchor_probs.shape[2], (anchor_probs.shape[1],))

            pred = anchor_probs[:, torch.arange(anchor_probs.shape[1]), token_idxs]
            target = pos_clusters[:, torch.arange(pos_clusters.shape[1]), token_idxs]
            dtype_to_neigh_loss[name] = F.cross_entropy(pred, target)

        # count all dtypes the same for now
        losses = {}
        total = 0
        for name in self.dtypes:
            losses[f'recon_loss_{name}'] = dtype_to_recon_loss[name]
            losses[f'neigh_loss_{name}'] = dtype_to_neigh_loss[name]
            overall = dtype_to_recon_loss[name] * self.recon_scaler + dtype_to_neigh_loss[name] * self.neigh_scaler
            losses[f'overall_{name}'] = overall
            total += overall
        losses['overall_loss'] = total

        return losses, outputs