
import pytorch_lightning as pl
import torch
from einops import rearrange

from mushroom.utils import construct_tile_expression


class LoggingCallback(pl.Callback):
    def __init__(self, log_every=10, log_n_samples=8, plot_genes=['IL7R', 'EPCAM', 'SPARC']):
        self.log_every = log_every
        self.log_n_samples = log_n_samples
        self.plot_genes = plot_genes

    def log_epoch(self, key, trainer, pl_module, outputs, batch, batch_idx):
            img = batch['he'][:self.log_n_samples].clone().detach().cpu()
            img -= img.min()
            img /= img.max()
            trainer.logger.log_image(
                key=f"{key}/he",
                images=[i[0] if i.shape[0] not in [1, 3] else i for i in img],
                caption=[i for i in range(img.shape[0])]
            )
            
            img = batch['context_he'][:self.log_n_samples].clone().detach().cpu()
            img -= img.min()
            img /= img.max()
            trainer.logger.log_image(
                key=f"{key}/context_he",
                images=[i[0] if i.shape[0] not in [1, 3] else i for i in img],
                caption=[i for i in range(img.shape[0])]
            )
            
            img = batch['masks'][:self.log_n_samples].clone().detach().cpu()
            img = img.sum(1) > 0
            new = torch.zeros_like(img, dtype=torch.float32)
            new[img] = 1.
            print(new.shape)
            trainer.logger.log_image(
                key=f"{key}/voxels",
                images=[i for i in new],
                caption=[i for i in range(new.shape[0])]
            )

            gene_idxs = [pl_module.model.genes.index(g) for g in self.plot_genes]
            img = construct_tile_expression(batch['exp'][:, :, gene_idxs], batch['masks'], batch['n_voxels'])
            img = img[:self.log_n_samples]
            img = torch.tensor(rearrange(img, 'b h w c -> c b 1 h w'))
            trainer.logger.log_image(
                key=f"{key}/exp_groundtruth",
                images=[i for i in img],
                caption=[g for g in self.plot_genes]
            )

            gene_idxs = [pl_module.model.genes.index(g) for g in self.plot_genes]
            img = construct_tile_expression(outputs['result']['exp'][:, :, gene_idxs], batch['masks'], batch['n_voxels'])
            img = img[:self.log_n_samples]
            img = torch.tensor(rearrange(img, 'b h w c -> c b 1 h w'))
            trainer.logger.log_image(
                key=f"{key}/exp_prediction",
                images=[i for i in img],
                caption=[g for g in self.plot_genes]
            )  
  
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.log_every == 0 and batch_idx==0:
            self.log_epoch('train', trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch % self.log_every == 0 and batch_idx==0:
            self.log_epoch('val', trainer, pl_module, outputs, batch, batch_idx)