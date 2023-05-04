
import pytorch_lightning as pl
import seaborn as sns
import torch
from einops import rearrange

from mushroom.utils import construct_tile_expression, display_labeled_as_rgb, HidePrint
from kmeans_pytorch import kmeans


class STExpressionLoggingCallback(pl.Callback):
    def __init__(self, log_every=10, log_n_samples=8, plot_genes=['EPCAM']):
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


# class ClusteringLoggingCallback(pl.Callback):
#     def __init__(self, dl, slide_shape, cmap=None):
#         self.labels = []
#         self.dl = dl
#         self.slide_shape = slide_shape

#         extended = sns.color_palette('tab20') + sns.color_palette('tab20b') + sns.color_palette('tab20c')
#         self.cmap = cmap if cmap is not None else extended

#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
#         x = outputs['labels']
#         self.labels.append(x)
  
#     def on_validation_epoch_end(self, trainer, pl_module):
#         if len(self.labels) == len(self.dl):
#             labels = torch.concat(self.labels)
#             slides = torch.tensor([x for x, _ in self.dl.dataset.tups])

#             pool = torch.unique(slides)
#             imgs = []
#             for slide in pool:
#                 mask = slides==slide
#                 flat_labels = labels[mask]
#                 img_labels = rearrange(flat_labels, '(h w) -> h w', h=self.slide_shape[-2])
#                 img_labels = img_labels.clone().detach().cpu()
#                 rgb = display_labeled_as_rgb(img_labels, cmap=self.cmap)
#                 imgs.append(rgb)
#             trainer.logger.log_image(
#                 key=f"val/clustered_image",
#                 images=[i for i in imgs],
#                 caption=[i for i in range(len(imgs))]
#             )
#         self.labels = []

class ClusteringLoggingCallback(pl.Callback):
    def __init__(self, dl, slide_shape, cmap=None, n_clusters=20, tol=1.):
        self.embs = []
        self.dl = dl
        self.slide_shape = slide_shape
        self.n_clusters=n_clusters
        self.tol = tol

        extended = sns.color_palette('tab20') + sns.color_palette('tab20b') + sns.color_palette('tab20c')
        self.cmap = cmap if cmap is not None else extended

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        x = outputs['embs']
        self.embs.append(x)
  
    def on_validation_epoch_end(self, trainer, pl_module):
        if len(self.embs) == len(self.dl):
            embs = torch.concat(self.embs)
            with HidePrint():
                cluster_ids_x, cluster_centers = kmeans(
                    X=embs, num_clusters=self.n_clusters, tol=self.tol,
                    distance='euclidean', device=embs.device
                )
            labels = cluster_ids_x.to(torch.long)
            slides = torch.tensor([x for x, _ in self.dl.dataset.tups])

            pool = torch.unique(slides)
            imgs = []
            for slide in pool:
                mask = slides==slide
                flat_labels = labels[mask]
                img_labels = rearrange(flat_labels, '(h w) -> h w', h=self.slide_shape[-2])
                img_labels = img_labels.clone().detach().cpu()
                rgb = display_labeled_as_rgb(img_labels, cmap=self.cmap)
                imgs.append(rgb)
            trainer.logger.log_image(
                key=f"val/clustered_image",
                images=[i for i in imgs],
                caption=[i for i in range(len(imgs))]
            )
        self.embs = []

                 

            

            