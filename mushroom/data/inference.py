import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Normalize, RandomCrop, Compose


class InferenceTransform(object):
    def __init__(self, normalize=None):
        self.transforms = Compose([
            normalize if normalize is not None else nn.Identity()
        ])

    def __call__(self, x):
        return self.transforms(x)
    

class InferenceSectionDataset(Dataset):
    def __init__(self, sections, section_to_img, size=(256, 256), transform=None):
        """"""
        self.size = size
        self.sections = sorted(section_to_img.keys())
        self.section_to_img = section_to_img
        
        # tiles are (ph pw c h w)
        self.section_to_tiles = {s:self.to_tiles(x) for s, x in self.section_to_img.items()}
        self.pw, self.ph = self.section_to_tiles[self.sections[0]].shape[:2]
        
        self.n_tiles_per_image = self.pw * self.ph
        outs = torch.stack(torch.meshgrid(
            torch.arange(len(self.sections)),
            torch.arange(self.section_to_tiles[self.sections[0]].shape[0]),
            torch.arange(self.section_to_tiles[self.sections[0]].shape[1]),
            indexing='ij'
        ))
        self.idx_to_coord = rearrange(
            outs, 'b n_sections n_rows n_cols -> (n_sections n_rows n_cols) b')

        self.transform = transform if transform is not None else nn.Identity()
        
    def to_tiles(self, x, size=None):
        size = self.size if size is None else size
        pad_h, pad_w = size[-2] - x.shape[-2] % size[-2], size[-1] - x.shape[-1] % size[-1]
        # left, top, right and bottom
        x = TF.pad(x, [pad_w // 2, pad_h // 2, pad_w // 2 + pad_w % 2, pad_h // 2 + pad_h % 2])
        x = x.unfold(-2, size[-2], size[-2] // 2)
        x = x.unfold(-2, size[-1], size[-1] // 2)

        x = rearrange(x, 'c ph pw h w -> ph pw c h w')

        return x

    def image_from_tiles(self, x):
        pad_h, pad_w = x.shape[-2] // 4, x.shape[-1] // 4
        x = x[..., pad_h:-pad_h, pad_w:-pad_w]
        return rearrange(x, 'ph pw c h w -> c (ph h) (pw w)')
    
    def section_from_tiles(self, x, section_idx, size=None):
        """
        x - (n c h w)
        """
        size = self.size if size is None else size
        mask = self.idx_to_coord[:, 0]==section_idx
        tiles = x[mask]
        ph, pw = self.idx_to_coord[mask, 1].max() + 1, self.idx_to_coord[mask, 2].max() + 1
        
        out = torch.zeros(ph, pw, x.shape[1], size[0], size[1])
        for idx, (_, r, c) in enumerate(self.idx_to_coord[mask]):
            out[r, c] = tiles[idx]
        
        return self.image_from_tiles(out)

    def __len__(self):
        return self.idx_to_coord.shape[0]

    def __getitem__(self, idx):
        section_idx, row_idx, col_idx = self.idx_to_coord[idx]
        section = self.sections[section_idx]
        return {
            'idx': section_idx,
            'row_idx': row_idx,
            'col_idx': col_idx,
            'tile': self.transform(self.section_to_tiles[section][row_idx, col_idx])
        }