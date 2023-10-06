from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from vit_pytorch.vit import Transformer
from vector_quantize_pytorch import VectorQuantize
from timm import create_model

from mushroom.model.expression_reconstruction import ZinbReconstructor


@dataclass
class SAEargs:
    size: int = 256
    patch_size: int = 32
    encoder_dim: int = 1024
    encoder_depth: int = 6
    heads: int = 8
    mlp_dim: int = 2048
    num_classes: int = 1000
    decoder_dim: int = 512
    decoder_depth: int = 6
    decoder_dim_head: int = 64
    triplet_scaler: float = 1.
    recon_scaler: float = 1.


class SAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        n_slides,
        decoder_type = 'pixels',
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        triplet_scaler = 1.,
        recon_scaler = 1.,
    ):
        super().__init__()
        self.decoder_type = decoder_type

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.triplet_scaler = triplet_scaler
        self.recon_scaler = recon_scaler

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.n_slides = n_slides
        self.slide_embedding = nn.Embedding(self.n_slides, encoder_dim)

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches + 1, decoder_dim)

        if self.decoder_type == 'zinb':
            self.to_pixels = ZinbReconstructor(decoder_dim, pixel_values_per_patch, n_metagenes=20)
        else:
            self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        
        n = int((num_patches - 1)**.5)
        self.repatch = Rearrange('b (h w) d -> b h w d', h=n, w=n)

        codebook_size = 100
        self.vq = VectorQuantize(
            dim = encoder_dim,
            codebook_size = codebook_size,     # codebook size
            # use_cosine_sim = True,
            orthogonal_reg_weight = 5,
        )


    def encode(self, img, slides):
        device = img.device

        # get patches
        # print('img', img.shape)
        patches = self.to_patch(img) # b (h w) (p1 p2 c)
        # print('patches', patches.shape)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        # print('tokens', tokens.shape)

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
        patch_tokens, indices, commit_loss = self.vq(patch_tokens)
        encoded_tokens = torch.cat((slide_token, patch_tokens), dim=1)

        return encoded_tokens, indices, commit_loss
    
    def decode(self, encoded_tokens):
        device = encoded_tokens.device
        
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        decoder_tokens += self.decoder_pos_emb(torch.arange(decoder_tokens.shape[1], device=device))

        decoded_tokens = self.decoder(decoder_tokens)

        return decoded_tokens


    def forward(self, imgs, slides, imgs_raw=None):
        """
        imgs - (n b c h w)
          + n=3, first image is anchor tile, second image is positive tile, third image is negative tile
        """
        outputs = []
        recon_loss = 0
        vq_loss = 0
        for i, (img, slide) in enumerate(zip(imgs, slides)):
            encoded_tokens = self.encode(img, slide)

            # quantize
            encoded_tokens, indices, commit_loss = self.quantize(encoded_tokens)
            # print(encoded_tokens.shape)
            
            decoded_tokens = self.decode(encoded_tokens)

            pred_pixel_values = self.to_pixels(decoded_tokens[:, 1:])
            if self.decoder_type == 'zinb':
                patches = self.to_patch(imgs_raw[i]) # (b n genes)
                loss = torch.mean(-pred_pixel_values['nb'].log_prob(patches))
                pred_pixel_values = pred_pixel_values['exp']
            else:
                loss = F.mse_loss(pred_pixel_values, self.to_patch(img))

            recon_loss += loss
            # vq_loss += commit_loss

            outputs.append({
                'encoded_tokens': encoded_tokens,
                'decoded_tokens': decoded_tokens,
                'pred_pixel_values': pred_pixel_values,
                # 'indices': indices,
                'recon_loss': loss
            })
        recon_loss /= len(imgs)
        # vq_loss /= len(imgs)

        idxs = torch.arange(1, outputs[0]['encoded_tokens'].shape[1])
        # idxs = idxs[torch.randperm(idxs.shape[0])][:10]
        idxs = idxs[torch.randperm(idxs.shape[0])]
        neg_idxs = idxs[torch.randperm(idxs.shape[0])]
        
        anchor_embs = outputs[0]['encoded_tokens'][:, idxs]
        pos_embs = outputs[1]['encoded_tokens'][:, idxs]
        neg_embs = outputs[2]['encoded_tokens'][:, neg_idxs]
        anchor_embs = rearrange(anchor_embs, 'b n d -> (b n) d')
        pos_embs = rearrange(pos_embs, 'b n d -> (b n) d')
        neg_embs = rearrange(neg_embs, 'b n d -> (b n) d')




        # anchor_repatched = self.repatch(outputs[0]['encoded_tokens'][:, 1:])
        # pos_repatched = self.repatch(outputs[1]['encoded_tokens'][:, 1:])
        # n_tokens = outputs[0]['encoded_tokens'].shape[1] - 1
        # anchor_row_idxs = torch.randint(1, int(n_tokens**.5) - 1, (n_tokens,))
        # anchor_col_idxs = torch.randint(1, int(n_tokens**.5) - 1, (n_tokens,))
        # pos_row_idxs = anchor_row_idxs + torch.randint(-1, 2, (n_tokens,))
        # pos_col_idxs = anchor_col_idxs + torch.randint(-1, 2, (n_tokens,))
        # anchor_embs = anchor_repatched[:, anchor_row_idxs, anchor_col_idxs] # (b, n, d)
        # pos_embs = pos_repatched[:, pos_row_idxs, pos_col_idxs]

        # neg_idxs = torch.randperm(pos_embs.shape[1]) + 1
        # neg_embs = outputs[2]['encoded_tokens'][:, neg_idxs]


        # anchor_embs = rearrange(anchor_embs, 'b n d -> (b n) d')
        # pos_embs = rearrange(pos_embs, 'b n d -> (b n) d')
        # neg_embs = rearrange(neg_embs, 'b n d -> (b n) d')






        # neighbor_loss = F.cosine_embedding_loss(anchor_embs, anchor_embs[torch.randperm(anchor_embs.shape[0])],
        #                                    -torch.ones(anchor_embs.shape[0], device=anchor_embs.device))



        pos = F.cosine_embedding_loss(anchor_embs, pos_embs, torch.ones(anchor_embs.shape[0], device=anchor_embs.device))
        neg = F.cosine_embedding_loss(anchor_embs, neg_embs, -torch.ones(anchor_embs.shape[0], device=anchor_embs.device))
        triplet_loss = pos + neg

        overall_loss = triplet_loss * self.triplet_scaler + recon_loss * self.recon_scaler
        # overall_loss = recon_loss + (vq_loss[0] * .2)

        losses = {
            'overall_loss': overall_loss,
            'recon_loss': recon_loss,
            'triplet_loss': triplet_loss,
            # 'commit_loss': vq_loss[0],
            # 'neighbor_loss': neighbor_loss
        }
        

        return losses, outputs