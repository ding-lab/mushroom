import logging
import os
import re

import alphashape
import pytorch_lightning as pl
import shapely
import numpy as np
import torch
import trimesh
import torch.nn.functional as F
from einops import rearrange
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.cluster import DBSCAN
from timm import create_model

from mushroom.utils import HidePrint


def cluster_embeddings(embs, slices, mask=None, h=None, n_clusters=20, tol=.1, device='cuda:0'):
    cluster_ids_x, cluster_centers = kmeans(
        X=embs, num_clusters=n_clusters, tol=tol, distance='euclidean',
        device=torch.device(device) if device is None else device
    )
    if mask is None:
        slice_embs = []
        idx = 0
        for s in slices:
            emb = embs[idx:idx + len(s)]
            emb = rearrange(emb, '(h w) d -> d h w', h=h)
            clusters = cluster_ids_x[idx:idx + len(s)]
            reshaped = rearrange(clusters, '(ph pw) -> ph pw', ph=emb.shape[-2])


            slice_embs.append({
                'embs': emb,
                'clusters': reshaped,
            })

            idx += len(s)
    else:
        slice_embs = []
        idx = 0
        for s in slices:
            emb = embs[idx:idx + len(s)]
            clusters = cluster_ids_x[idx:idx + len(s)]
            reshaped_emb = torch.zeros((emb.shape[-1], mask.shape[-2], mask.shape[-1]))
            reshaped_clusters = torch.zeros((mask.shape[-2], mask.shape[-1]), dtype=cluster_ids_x.dtype) - 1
            for i, (r, c) in enumerate(torch.argwhere(mask)):
                reshaped_emb[:, r, c] = emb[i]
                reshaped_clusters[r, c] = clusters[i]

            slice_embs.append({
                'embs': reshaped_emb,
                'clusters': reshaped_clusters,
            })

            idx += len(s)
        
    return slice_embs


def cluster_pt_clouds(pts, clusters, collapse_z=True, eps=.01, alpha=.25):
    pts = pts.astype(np.float32)
    init_z = pts[:, 2].copy()
    if collapse_z:
        mapping = {x:i for i, x in enumerate(np.unique(pts[:, 2]))}
        pts[:, 2] = [mapping[x] for x in pts[:, 2]]
    scaler = pts.max()
    pts /= scaler

    mesh_dict = {}
    for c in np.unique(clusters):
        print(c)
        mesh_dict[c] = {}
        filtered = pts[clusters==c]
        clustering = DBSCAN(eps=eps)
        groups = clustering.fit_predict(filtered)
        filtered *= scaler
        filtered[:, 2] = init_z[clusters==c] # transform back
        for g in np.unique(groups):
            if g >= 0:
                print(c, g)
                group_pts = filtered[groups==g]
                print(np.unique(group_pts[:, -1]))
                if len(np.unique(group_pts[:, 2])) >= 2:
                    try:
                        mesh = alphashape.alphashape(group_pts, alpha)
                        d = trimesh.exchange.export.export_dict(mesh)
                        d['type'] =  '3d'
                    except:
                        d = {'type': '2d', 'vertices': []}
                else:
                    mesh = alphashape.alphashape(group_pts[:, :-1], 1.)
                    try:
                        x, y = mesh.exterior.xy
                    except AttributeError:
                        if isinstance(mesh, shapely.GeometryCollection):
                            try:
                                x, y = mesh.geoms[0].exterior.xy
                            except IndexError:
                                x, y = [], []
                        elif isinstance(mesh, shapely.MultiPolygon):
                            x, y = mesh.convex_hull.exterior.xy
                        else:
                            x, y = [], []
                    x, y = np.asarray(x), np.asarray(y)
                    coords = np.concatenate(
                        [np.expand_dims(arr, -1) for arr in [x, y, [np.unique(group_pts[:, 2])[0]] * len(x)]],
                        axis=-1)
                    d = {'type': '2d', 'vertices': coords.tolist()}
                mesh_dict[c][g] = d
    return mesh_dict


class SliceClustering(torch.nn.Module):
    def __init__(self, in_dim, n_clusters=20, emb_dim=64, triplet_scaler=1., cluster_scaler=1.):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.n_clusters = n_clusters
        self.triplet_scaler = triplet_scaler
        self.cluster_scaler = cluster_scaler
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, self.in_dim // 2),
            torch.nn.BatchNorm1d(self.in_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.in_dim // 2, self.in_dim // 4),
            torch.nn.BatchNorm1d(self.in_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(self.in_dim // 4, self.in_dim // 8),
            torch.nn.BatchNorm1d(self.in_dim // 8),
            torch.nn.ReLU(),
            torch.nn.Linear(self.in_dim // 8, self.emb_dim)
        )

        self.centroids = torch.nn.Parameter(torch.rand(self.n_clusters, self.emb_dim), requires_grad=True)

        self.triplet_loss = torch.nn.TripletMarginLoss()
        self.cluster_loss = torch.nn.CrossEntropyLoss()
        
    def calculate_loss(self, result, anchor_true):
        anchor, pos, neg = result['anchor_embs'], result['pos_embs'], result['neg_embs']
        anchor_label, pos_label, neg_label = result['anchor_labels'], result['pos_labels'], result['neg_labels']
        anchor_dists, pos_dists, neg_dists = result['anchor_dists'], result['pos_dists'], result['neg_dists']

        triplet_loss = self.triplet_loss(anchor, pos, neg) * 1.

        probs = torch.nn.functional.softmax(anchor_dists, dim=-1)
        anchor_cluster_loss = self.cluster_loss(probs, anchor_true) * .01

        probs = torch.nn.functional.softmax(pos_dists, dim=-1)
        pos_cluster_loss = self.cluster_loss(probs, anchor_true) * .99

        cluster_loss = (anchor_cluster_loss + pos_cluster_loss) * 0
        
        # pos_loss = self.cluster_loss(probs, pos_label)
        # neg_loss = self.cluster_loss(probs, neg_label)
        # cluster_loss = pos_loss / neg_loss * self.cluster_scaler
        # pos_loss = self.cluster_loss(probs, pos_label)

        # anchor_clust_dist = anchor_dists[anchor_label].mean()
        # pos_clust_dist = pos_dists[pos_label].mean()
        # neg_clust_dist = neg_dists[neg_label].mean()
        # attraction_loss = anchor_clust_dist + pos_clust_dist + neg_clust_dist

        # pos_dist_loss = self.mse(anchor_dists[anchor_label], pos_dists[anchor_label])
        # neg_dist_loss = self.mse(anchor_dists[anchor_label], neg_dists[anchor_label])

        # cluster_loss = pos_dist_loss / neg_dist_loss

        # het_loss = F.kl_div(
        #     F.softmax(anchor_dists.sum(0), dim=-1),
        #     F.softmax(self.target, dim=-1)
        # )


        # repulsion_loss = torch.cdist(self.centroids, self.centroids).mean()
        
        # if self.centroids is not None:
        #     dists = torch.cdist(anchor, self.centroids)
        #     probs = torch.nn.functional.softmax(dists, dim=-1)
        #     pos_loss = self.cluster_loss(probs, pos_label)
        #     cluster_loss = pos_loss
        # else:
        #     cluster_loss, pos_loss = 1., 1.

        return {
            'overall': triplet_loss + cluster_loss,
            # 'overall': triplet_loss
            'cluster': cluster_loss,
            'triplet': triplet_loss,
            'anchor_cluster_loss': anchor_cluster_loss,
            'pos_cluster_loss': pos_cluster_loss,
            # 'het': het_loss
            # 'pos_cluster_loss': pos_loss,
            # 'neg_cluster_loss': neg_loss,
            # 'repulsion_loss': repulsion_loss,
            # 'attraction_loss': attraction_loss
        }
    
    def cluster(self, embs):
        assert self.centroids is not None

        dists = torch.cdist(embs, self.centroids)
        labels = dists.argmin(dim=1)

        return dists, labels
    
    def encode(self, x):
        embs = self.encoder(x)
        dists, labels = self.cluster(embs)
        return {
            'embs': embs,
            'labels': labels,
            'dists': dists
        }
        
    def forward(self, anchor, pos, neg):
        anchor = self.encode(anchor)
        pos = self.encode(pos)
        neg = self.encode(neg)
        
        return {
            'anchor_embs': anchor['embs'],
            'anchor_labels': anchor['labels'],
            'anchor_dists': anchor['dists'],
            'pos_embs': pos['embs'],
            'pos_labels': pos['labels'],
            'pos_dists': pos['dists'],
            'neg_embs': neg['embs'],
            'neg_labels': neg['labels'],
            'neg_dists': neg['dists'],
        }
    

class LitSliceClustering(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        
        self.model = model
        self.lr = config['training']['lr']
        self.config = config # saving config so we can load from checkpoint
        
        self.save_hyperparameters(ignore=['model'])

    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        """Need to overwrite default method due to model pickling issue"""
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint['hyper_parameters']['config']
        m = SliceClustering(
            config['in_dim'],
            n_clusters=config['n_clusters'],
            emb_dim=config['emb_dim'],
        )
        d = {re.sub(r'^model.(.*)$', r'\1', k):v for k, v in checkpoint['state_dict'].items()}
        m.load_state_dict(d)

        return LitSliceClustering(m, config)

    def training_step(self, batch, batch_idx):
        anchor, pos, neg, anchor_true = batch['anchor'], batch['pos'], batch['neg'], batch['anchor_label']
        result = self.model(anchor, pos, neg)
        losses = self.model.calculate_loss(result, anchor_true)
        losses = {f'train/{k}':v for k, v in losses.items()}
        losses['train/loss'] = losses['train/overall']
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        losses['loss'] = losses['train/loss']
        losses['result'] = result
        
        return losses
    
    def validation_step(self, batch, batch_idx):
        emb = batch['emb']
        result = self.model.encode(emb)
        
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, batch):
        emb = batch['emb']
        return {
            'result': self.model.encode(emb),
            'batch': batch
        }


# class ClusteringCentroidCallback(pl.Callback):
#     def __init__(self, dl, iters=0):
#         self.dl = dl
#         self.embs = []
#         self.count = 0
#         self.iters = iters

#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
#         if self.count == self.iters:
#             x = outputs['embs']
#             self.embs.append(x)
  
#     def on_validation_epoch_end(self, trainer, pl_module):
#         if self.count == self.iters:
#             embs = torch.concat(self.embs)
#             # pl_module.model.set_centroids(embs)
#             self.embs = []
#         self.count += 1













# class SliceClustering(torch.nn.Module):
#     def __init__(self, in_dim, n_clusters=20, emb_dim=64, tol=1., triplet_scaler=1., cluster_scaler=1.):
#         super().__init__()
#         self.in_dim = in_dim
#         self.emb_dim = emb_dim
#         self.tol = tol
#         self.n_clusters = n_clusters
#         self.triplet_scaler = triplet_scaler
#         self.cluster_scaler = cluster_scaler
        
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(self.in_dim, self.in_dim // 2),
#             torch.nn.BatchNorm1d(self.in_dim // 2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.in_dim // 2, self.in_dim // 4),
#             torch.nn.BatchNorm1d(self.in_dim // 4),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.in_dim // 4, self.in_dim // 8),
#             torch.nn.BatchNorm1d(self.in_dim // 8),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.in_dim // 8, self.emb_dim)
#         )

#         self.centroids = torch.nn.Parameter(torch.rand(self.n_clusters, self.emb_dim), requires_grad=True)

#         self.triplet_loss = torch.nn.TripletMarginLoss()
#         # self.cluster_loss = torch.nn.CrossEntropyLoss()
#         self.mse = torch.nn.MSELoss()

#         self.target = torch.nn.Parameter(torch.ones((self.n_clusters,)), requires_grad=False)
        
#     def calculate_loss(self, result):
#         anchor, pos, neg = result['anchor_embs'], result['pos_embs'], result['neg_embs']
#         # anchor_label, pos_label, neg_label = result['anchor_labels'], result['pos_labels'], result['neg_labels']
#         # anchor_dists, pos_dists, neg_dists = result['anchor_dists'], result['pos_dists'], result['neg_dists']

#         # if np.random.choice(np.arange(100)) == 1:

#         #     print(torch.sum(anchor_label==pos_label))
#         #     print(anchor_label[:10], pos_label[:10])

#         triplet_loss = self.triplet_loss(anchor, pos, neg) * self.triplet_scaler

#         # probs = torch.nn.functional.one_hot(
#         #     anchor_label, num_classes=self.n_clusters).to(torch.float32)
#         # probs = torch.nn.functional.softmax(anchor_dists)
        
#         # pos_loss = self.cluster_loss(probs, pos_label)
#         # neg_loss = self.cluster_loss(probs, neg_label)
#         # cluster_loss = pos_loss / neg_loss * self.cluster_scaler
#         # pos_loss = self.cluster_loss(probs, pos_label)

#         # anchor_clust_dist = anchor_dists[anchor_label].mean()
#         # pos_clust_dist = pos_dists[pos_label].mean()
#         # neg_clust_dist = neg_dists[neg_label].mean()
#         # attraction_loss = anchor_clust_dist + pos_clust_dist + neg_clust_dist

#         # pos_dist_loss = self.mse(anchor_dists[anchor_label], pos_dists[anchor_label])
#         # neg_dist_loss = self.mse(anchor_dists[anchor_label], neg_dists[anchor_label])

#         # cluster_loss = pos_dist_loss / neg_dist_loss

#         # het_loss = F.kl_div(
#         #     F.softmax(anchor_dists.sum(0), dim=-1),
#         #     F.softmax(self.target, dim=-1)
#         # )


#         # repulsion_loss = torch.cdist(self.centroids, self.centroids).mean()
        
#         # if self.centroids is not None:
#         #     dists = torch.cdist(anchor, self.centroids)
#         #     probs = torch.nn.functional.softmax(dists, dim=-1)
#         #     pos_loss = self.cluster_loss(probs, pos_label)
#         #     cluster_loss = pos_loss
#         # else:
#         #     cluster_loss, pos_loss = 1., 1.

#         return {
#             # 'overall': triplet_loss + cluster_loss,
#             'overall': triplet_loss
#             # 'cluster': cluster_loss,
#             # 'triplet': triplet_loss,
#             # 'het': het_loss
#             # 'pos_cluster_loss': pos_loss,
#             # 'neg_cluster_loss': neg_loss,
#             # 'repulsion_loss': repulsion_loss,
#             # 'attraction_loss': attraction_loss
#         }
    
#     # def set_centroids(self, embs):
#     #     print('setting centroids')
#     #     with HidePrint():
#     #         cluster_ids_x, cluster_centers = kmeans(
#     #             X=embs, num_clusters=self.n_clusters, tol=self.tol,
#     #             distance='euclidean', device=embs.device
#     #         )
#     #     self.centroids = torch.nn.Parameter(cluster_centers.to(embs.device), requires_grad=True)
#     #     print('centroid shape', self.centroids.shape)
    
#     def cluster(self, embs):
#         assert self.centroids is not None

#         dists = torch.cdist(embs, self.centroids)
#         labels = dists.argmin(dim=1)

#         return dists, labels


#         # with HidePrint():
#         #     out = kmeans_predict(
#         #         embs,
#         #         self.centroids.clone().detach(),
#         #         distance='euclidean',
#         #         device=embs.device,
#         #     ).to(embs.device)
#         # return out
    
#     def encode(self, x):
#         embs = self.encoder(x)
#         dists, labels = self.cluster(embs)
#         return {
#             'embs': embs,
#             'labels': labels,
#             'dists': dists
#         }
        
#     def forward(self, anchor, pos, neg):
#         anchor = self.encode(anchor)
#         pos = self.encode(pos)
#         neg = self.encode(neg)
        
#         return {
#             'anchor_embs': anchor['embs'],
#             'anchor_labels': anchor['labels'],
#             'anchor_dists': anchor['dists'],
#             'pos_embs': pos['embs'],
#             'pos_labels': pos['labels'],
#             'pos_dists': pos['dists'],
#             'neg_embs': neg['embs'],
#             'neg_labels': neg['labels'],
#             'neg_dists': neg['dists'],
#         }


# class SliceClustering(torch.nn.Module):
#     def __init__(self, in_dim, emb_dim=64):
#         super().__init__()
#         self.in_dim = in_dim
#         self.emb_dim = emb_dim

#         feat_out_size = 1000
#         self.encoder = torch.nn.Sequential(
#             create_model('resnet18', in_chans=self.in_dim),
#             torch.nn.Linear(feat_out_size, feat_out_size // 2),
#             torch.nn.BatchNorm1d(feat_out_size // 2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(feat_out_size // 2, feat_out_size // 4),
#             torch.nn.BatchNorm1d(feat_out_size // 4),
#             torch.nn.ReLU(),
#             torch.nn.Linear(feat_out_size // 4, feat_out_size // 8),
#             torch.nn.BatchNorm1d(feat_out_size // 8),
#             torch.nn.ReLU(),
#             torch.nn.Linear(feat_out_size // 8, self.emb_dim)
#         )

#         self.triplet_loss = torch.nn.TripletMarginLoss()
        
#     def calculate_loss(self, result):
#         anchor, pos, neg = result['anchor_embs'], result['pos_embs'], result['neg_embs']

#         triplet_loss = self.triplet_loss(anchor, pos, neg)

#         return {
#             'overall': triplet_loss
#         }
    
#     def encode(self, x):
#         embs = self.encoder(x)
#         # dists, labels = self.cluster(embs)
#         return {
#             'embs': embs,
#             # 'labels': labels,
#             # 'dists': dists
#         }
        
#     def forward(self, anchor, pos, neg):
#         anchor = self.encode(anchor)
#         pos = self.encode(pos)
#         neg = self.encode(neg)
        
#         return {
#             'anchor_embs': anchor['embs'],
#             'pos_embs': pos['embs'],
#             'neg_embs': neg['embs'],
#         }
    