import logging
import os
import warnings

import torch
from einops import rearrange, repeat
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from vit_pytorch import ViT

import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
from mushroom.model.sae_v3 import SAE, SAEargs

logger = logging.getLogger()
logger.setLevel(logging.INFO)
warnings.simplefilter(action='ignore', category=FutureWarning)


class SAELearner(object):
    def __init__(
            self,
            config, # mushroom input section configuration file
            dtype, # data type to use for training
            channel_mapping=None, # channel mapping if channels have different names accross sections
            channels=None, # channels to use in model
            scale=.1, # how much to downsample image from full resolution
            batch_size=32, # batch size during training and inference
            num_workers=1, # number of workers to use in data loader
            sae_args=SAEargs(), # args for ViT
            device=None,
            pct_expression=.02, # channels with % of spots expressing < will be removed
            ):
        self.config = config
        self.dtype = dtype
        self.channel_mapping = channel_mapping
        self.scale = scale
        self.sae_args = sae_args
        self.size = (sae_args.size, sae_args.size)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        logging.info(f'using device: {self.device}')

        logging.info(f'generating inputs for {self.dtype} tissue sections')
        if self.dtype == 'multiplex':
            learner_data = multiplex.get_learner_data(
                self.config, self.scale, self.size, self.sae_args.patch_size,
                channels=channels, channel_mapping=self.channel_mapping
            )
        elif self.dtype == 'he':
            pass
        elif self.dtype == 'visium':
            learner_data = visium.get_learner_data(
                self.config, self.scale, self.size, self.sae_args.patch_size,
                channels=channels, channel_mapping=self.channel_mapping, pct_expression=pct_expression,
            )
        else:
            raise RuntimeError(f'dtype must be one of the following: \
["multiplex", "he", "visium"], got {self.dtype}')

        self.section_to_img = learner_data.section_to_img
        self.train_transform = learner_data.train_transform
        self.inference_transform = learner_data.inference_transform
        self.train_ds = learner_data.train_ds
        self.inference_ds = learner_data.inference_ds
        self.channels = learner_data.channels

        logging.info('creating data loaders')
        self.train_dl = DataLoader(
            self.train_ds, batch_size=batch_size, num_workers=num_workers
        )
        self.inference_dl = DataLoader(
            self.inference_ds, batch_size=batch_size, num_workers=num_workers
        )

        logging.info('creating ViT')
        encoder = ViT(
            image_size=self.train_transform.output_size[0],
            patch_size=self.train_transform.output_patch_size,
            num_classes=self.sae_args.num_classes,
            dim=self.sae_args.encoder_dim,
            depth=self.sae_args.encoder_depth,
            heads=self.sae_args.heads,
            mlp_dim=self.sae_args.mlp_dim,
            channels=len(self.channels),
        )

        self.sae = SAE(
            encoder=encoder,
            decoder_type='pixels',
            decoder_dim=self.sae_args.decoder_dim,
            n_slides=len(self.section_to_img),
            decoder_depth=self.sae_args.decoder_depth,
            decoder_dim_head=self.sae_args.decoder_dim_head,
            triplet_scaler=self.sae_args.triplet_scaler,
            recon_scaler=self.sae_args.recon_scaler,
        )

        self.sae = self.sae.to(self.device)

    def train(
            self,
            num_iters=1000,
            lr=1e-4,
            save_every=100,
            log_every=10,
            save_dir='./',
            device=None
        ):
        device = device if device is not None else self.device
        self.sae = self.sae.to(device)
        opt = torch.optim.Adam(self.sae.parameters(), lr=lr)

        for i, b in enumerate(self.train_dl):
            opt.zero_grad()
            img_x = torch.stack((b['anchor_img'], b['pos_img'], b['neg_img']))
            section_x = torch.stack((b['anchor_idx'], b['pos_idx'], b['neg_idx']))
            img_x, section_x = img_x.to(device), section_x.to(device)

            losses, outputs = self.sae(img_x, section_x)
            loss = losses['overall_loss']
            loss.backward()
            opt.step()

            if i % log_every == 0:
                logging.info(f'iteration {i}: {losses}')
                # print(f'iteration {i}: {losses}')

            if i % save_every == 0:
                fp = os.path.join(save_dir, f'{i}iter.pt')
                logging.info(f'saving checkpoint to {fp}')
                torch.save(self.sae.state_dict(), fp)

            if i == num_iters:
                fp = os.path.join(save_dir, f'final.pt')
                logging.info(f'saving final checkpoint to {fp}')
                torch.save(self.sae.state_dict(), fp)
                break

    def embed_sections(self, device=None):
        device = device if device is not None else self.device
        self.sae = self.sae.to(device)

        if self.sae is None:
            raise RuntimeError('Must train model prior to embedding sections.')

        n = len(self.inference_ds)
        num_patches = self.size[0] // self.sae_args.patch_size
        embs = torch.zeros(
            n, num_patches, num_patches, self.sae.encoder.pos_embedding.shape[-1])
        pred_patches = torch.zeros(n, len(self.channels), self.train_transform.output_size[0], self.train_transform.output_size[1])

        bs = self.inference_dl.batch_size
        self.sae.eval()
        with torch.no_grad():
            for i, b in enumerate(self.inference_dl):
                x, section_idx = b['img'], b['section_idx']
                x, section_idx = x.to(device), section_idx.to(device)
                encoded_tokens, mu, std = self.sae.encode(x, section_idx, use_means=True)

                pred_pixel_values = self.sae.decode(encoded_tokens)

                encoded_tokens = rearrange(encoded_tokens[:, 1:], 'b (h w) d -> b h w d',
                                        h=num_patches, w=num_patches)
                pred_pixel_values = rearrange(
                    pred_pixel_values, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                    h=num_patches, w=num_patches,
                    p1=self.train_transform.output_patch_size, p2=self.train_transform.output_patch_size,
                    c=len(self.channels))
                
                embs[i * bs:(i + 1) * bs] = encoded_tokens.cpu().detach()
                pred_patches[i * bs:(i + 1) * bs] = pred_pixel_values.cpu().detach()
        recon_imgs = torch.stack(
            [self.inference_ds.section_from_tiles(
                pred_patches, i, size=(self.train_transform.output_size[0], self.train_transform.output_size[1])
            ) for i in range(len(self.inference_ds.sections))]
        )
        recon_embs = torch.stack(
            [self.inference_ds.section_from_tiles(
                rearrange(embs, 'n h w c -> n c h w'),
                i, size=(num_patches, num_patches))
                for i in range(len(self.inference_ds.sections))] 
        )
        # recon_embs = self.regress_patch_position(recon_embs)

        return recon_imgs, recon_embs
    
    def regress_patch_position(
            self,
            recon_embs # (n_sections, n_channels, height, width)
        ):
        target = rearrange(recon_embs, 'n c h w -> n h w c').numpy()
        num_patches = self.size[0] // self.sae_args.patch_size

        # get patch positions
        rc = torch.zeros(
            self.inference_ds.idx_to_coord.shape[0], num_patches, num_patches,
            dtype=torch.long
        )
        idx_to_var = [(r, c) for r in range(rc.shape[1]) for c in range(rc.shape[2])]
        var_to_idx = {v:i for i, v in enumerate(idx_to_var)}
        for i in range(self.inference_ds.idx_to_coord.shape[0]):
            for r in range(rc.shape[1]):
                for c in range(rc.shape[2]):
                    rc[i, r, c] = var_to_idx[(r, c)]
        hots = torch.nn.functional.one_hot(rc)

        # retile to same positions as recon_embs
        X_coords = torch.stack(
            [self.inference_ds.section_from_tiles(
                rearrange(hots, 'n h w c -> n c h w'),
                i, size=(num_patches, num_patches))
            for i in range(len(self.inference_ds.sections))]
        )
        X_coords = rearrange(X_coords, 'b c h w -> b h w c')
        
        # get sections
        X_sections = repeat(
            torch.arange(len(self.inference_ds.sections)),
            'b -> b h w', h=target.shape[1], w=target.shape[2]
        )
        X_sections = torch.nn.functional.one_hot(X_sections)

        # combine features
        X = torch.cat((X_coords, X_sections), dim=-1).numpy()

        # reshape for sklearn
        target = rearrange(target, 'b h w c -> (b h w) c')
        X = rearrange(X, 'b h w c -> (b h w) c')

        # fit linear model
        lm = LinearRegression()
        lm.fit(X, target)

        # get residuals
        residuals = lm.predict(X) - target

        # to original shape
        embs_regressed = rearrange(
            residuals, '(b h w) c -> b c h w',
            b=X_coords.shape[0], h=X_coords.shape[1], w=X_coords.shape[2]
        )

        return torch.tensor(embs_regressed)
