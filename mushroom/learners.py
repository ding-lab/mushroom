import os
from pathlib import Path

import pytorch_lightning as pl
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, reduce
from pytorch_lightning.callbacks import ModelCheckpoint

import mushroom.utils as utils
from mushroom.models.expression_prediction import STExpressionModel, STExpressionLightning
from mushroom.transforms import NormalizeHETransform, OverlaidHETransform
from mushroom.dataloaders import HEPredictionDataset, STDataset, MultisampleSTDataset
from mushroom.logging import STExpressionLoggingCallback


DEFAULT_ST_EXPRESSION_LEARNER_CONFIG = {
    'backbone': 'resnet34',
    'genes': None,
    'n_metagenes': 10,
    'he_scaler': 0.1,
    'kl_scaler': 0.001,
    'exp_scaler': 1.0,
    'size': (128, 128),
    'context_res': 2,
    'scale': .2,
    'max_voxels': 64,
    'means': (0.74600196, 0.67650163, 0.8227606),
    'stds': (0.20359719, 0.2339519 , 0.13040961),
    'training': {
        'log_n_samples': 8,
        'max_epochs': 25,
        'log_every': 1,
        'chkpt_every': None,
        'limit_train_batches': 1.,
        'limit_val_batches': .1,
        'accelerator': 'gpu',
        'devices': [1],
        'lr': 2e-5,
        'precision': 32
    },
}

class STExpressionLearner(object):
    def __init__(self, train_adatas, val_adatas, train_hes, val_hes, config,
                 batch_size=16, num_workers=20, logger=None):
        self.train_adatas, self.val_adatas = train_adatas, val_adatas
        self.train_hes, self.val_hes = train_hes, val_hes
        if not isinstance(train_adatas, dict):
            self.train_adatas = {f'train_sample_{i}':a for i, a in range(train_adatas)}
        if not isinstance(val_adatas, dict):
            self.val_adatas = {f'val_sample_{i}':a for i, a in range(val_adatas)}
        if not isinstance(train_hes, dict):
            self.train_hes = {f'train_sample_{i}':a for i, a in range(train_hes)}
        if not isinstance(val_hes, dict):
            self.val_hes = {f'val_sample_{i}':a for i, a in range(val_hes)}

        self.config = DEFAULT_ST_EXPRESSION_LEARNER_CONFIG.copy()
        self.config.update({k:v for k, v in config.items() if not isinstance(v, dict)})
        self.config['training'].update(config['training'])
        self.config['genes'] = next(iter(self.train_adatas.values())).var.index.to_list()

        self.logger = logger
        self.size = self.config['size']
        self.context_res = self.config['context_res']
        self.scale = self.config['scale']
        self.max_voxels = self.config['max_voxels']

        self.means, self.stds = utils.get_means_and_stds(self.train_adatas.values())
        self.config['means'], self.config['stds'] = self.means, self.stds

        self.train_transform = OverlaidHETransform(
            p=.95, size=(int(self.size[0] * self.context_res), int(self.size[1] * self.context_res)),
            means=self.means, stds=self.stds)
        self.sid_to_train_ds = {sid:STDataset(
                                    a, self.train_hes[sid],
                                    transform=self.train_transform, scale=self.scale,
                                    max_voxels_per_sample=self.max_voxels)
                                for sid, a in self.train_adatas.items()}

        self.val_transform = OverlaidHETransform(
            p=.0, size=(int(self.size[0] * self.context_res), int(self.size[1] * self.context_res)),
            means=self.means, stds=self.stds)
        self.sid_to_val_ds = {sid:STDataset(
                                    a, self.val_hes[sid],
                                    transform=self.val_transform, scale=self.scale,
                                    max_voxels_per_sample=self.max_voxels)
                                for sid, a in self.val_adatas.items()}

        self.train_ds = MultisampleSTDataset(self.sid_to_train_ds)
        self.val_ds = MultisampleSTDataset(self.sid_to_val_ds)

        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
        self.val_dl = DataLoader(self.val_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

        self.model = None
        self.trainer = None
        self.chkpt_dir = None

    def create_model(self):
        m = STExpressionModel(
            self.config['genes'],
            n_metagenes=self.config['n_metagenes'],
            he_scaler=self.config['he_scaler'],
            kl_scaler=self.config['kl_scaler'],
            exp_scaler=self.config['exp_scaler'],
        )

        self.model = STExpressionLightning(m, self.config)

    def create_trainer(self):
        callbacks = [
                STExpressionLoggingCallback(
                    log_every=self.config['training']['log_every'],
                    log_n_samples=self.config['training']['log_n_samples']
                )
        ]
        if self.config['training']['chkpt_every'] is not None:
            self.chkpt_dir = os.path.join(self.logger.save_dir, "ckpts")
            Path(self.chkpt_dir).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.chkpt_dir,
                    every_n_epochs=self.config['training']['chkpt_every'],
                )
            )
        self.trainer = pl.Trainer(
            callbacks=callbacks,
            devices=self.config['training']['devices'],
            accelerator=self.config['training']['accelerator'],
            enable_checkpointing=True if self.config['training']['chkpt_every'] is not None else False,
            max_epochs=self.config['training']['max_epochs'],
            precision=self.config['training']['precision'],
            limit_val_batches=self.config['training']['limit_val_batches'],
            limit_train_batches=self.config['training']['limit_train_batches'],
            logger=self.logger
        )

    def fit(self):
        if self.model is None:
            self.create_model()
        if self.trainer is None:
            self.create_trainer()
        self.trainer.fit(model=self.model,
                         train_dataloaders=self.train_dl, val_dataloaders=self.val_dl)
        
    def save_model(self, fp):
        self.trainer.save_checkpoint(fp)


class HEExpressionPredictor(object):
    def __init__(self, model, devices=None, accelerator=None):
        self.model = model

        if accelerator is None:
            accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        if accelerator == 'gpu' and devices is None:
            devices = [0]

        self.predictor = pl.Trainer(
            devices=devices,
            accelerator=accelerator,
        )

    def predict(self, he, genes=None, n_workers=10, batch_size=16):
        """predict expression for H&E image"""
        # if a file then read in
        if isinstance(he, str):
            he = tifffile.imread(he)
        if he.shape[0] == 3: # dataset expect (h w c)
            he = rearrange(he, 'c h w -> h w c')

        if genes is not None:
            self.model.set_prediction_genes(genes)

        pred_transform = NormalizeHETransform(means=self.model.config['means'], stds=self.model.config['stds'])
        ds = HEPredictionDataset(
                he, size=self.model.config['size'][0], context_res=self.model.config['context_res'],
                transform=pred_transform, scale=self.model.config['scale'])

        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=20)

        result = self.predictor.predict(self.model, dataloaders=dl)

        exp = torch.concat([r['exp'] for r in result])
        exp = rearrange(exp, 'b h w c -> b c h w')
        
        # # rescale each gene
        # for i in range(exp.shape[1]):
        #     img = exp[:, 1, :, :]
        #     min_thresh = np.percentile(img, .01)
        #     max_thresh = np.percentile(img, .99)
        #     img[img < min_thresh] = min_thresh
        #     img[img > max_thresh] = max_thresh
        #     exp[:, i, :, :] = img

        # convert to uint8 for faster tileing
        exp -= exp.min()
        exp /= exp.max()
        exp *= 255.
        exp = exp.to(torch.uint8)

        coord_to_tile = {k:v for k, v in zip(dl.dataset.coords, exp)}

        return dl.dataset.retile(coord_to_tile), self.model.model.genes