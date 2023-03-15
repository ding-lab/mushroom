
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from mushroom.utils import rescale_img, create_circular_mask, project_expression
from mushroom.transforms import OverlaidHETransform


def incorporate_hi_res(adata, he, scale=.05):
    spot_diameter = next(iter(adata.uns['spatial'].values()))['scalefactors']['spot_diameter_fullres']
    spot_diameter, spot_radius = int(spot_diameter), int(spot_diameter / 2)
    c_min, r_min = np.min(adata.obsm['spatial'], axis=0) - spot_radius
    c_max, r_max = np.max(adata.obsm['spatial'], axis=0) + spot_radius

    adata.uns['trimmed'] = he[r_min:r_max, c_min:c_max]
    adata.obsm['spatial_trimmed'] = adata.obsm['spatial'] + np.asarray([-c_min, -r_min])

    adata.uns[f'trimmed_{scale}'] = rescale_img(adata.uns['trimmed'], scale=scale)
    adata.uns[f'spatial_trimmed_{scale}'] = adata.obsm['spatial_trimmed'] * scale

    sr = int(scale * spot_radius)
    labeled_img = np.zeros((adata.uns[f'trimmed_{scale}'].shape[0], adata.uns[f'trimmed_{scale}'].shape[1]),
                           dtype=np.uint32)
    footprint = create_circular_mask(sr * 2, sr * 2)
    for i, (c, r) in enumerate(adata.uns[f'spatial_trimmed_{scale}']):
        r, c = int(r), int(c)
        rect = np.zeros((sr * 2, sr * 2))
        rect[footprint>0] = i + 1
        labeled_img[r-sr:r+sr, c-sr:c+sr] = rect
    adata.uns[f'trimmed_{scale}_labeled_img'] = labeled_img

    return adata


def create_masks(labeled_mask, voxel_idxs, max_area, thresh=.25):
    voxel_idxs = torch.unique(labeled_mask)[1:].to(torch.long)
    lm = labeled_mask.squeeze()
    masks = torch.zeros((len(voxel_idxs), lm.shape[-2], lm.shape[-1]), dtype=torch.bool)
    for i, l in enumerate(voxel_idxs):
        m = masks[i]
        m[lm==l] = 1
    
    keep = masks.sum(dim=(-1,-2)) / max_area > thresh
    
    if keep.sum() > 0:
        masks = masks[keep]
        voxel_idxs = voxel_idxs[keep]
        return masks, voxel_idxs
    
    return None, None


class STDataset(Dataset):
    """Registration Dataset"""
    def __init__(self, adata, he, size=(256, 256), transform=OverlaidHETransform(),
                 scale=.5, length=None, max_voxels_per_sample=16):
        self.scale = scale
        self.size = size

        self.adata = incorporate_hi_res(adata, he, scale=scale)
        self.genes = self.adata.var.index.to_list()
        self.length = self.adata.shape[0] if length is None else length
        self.max_voxels_per_sample = max_voxels_per_sample

        # images
        he = self.adata.uns[f'trimmed_{scale}']
        labeled_img = self.adata.uns[f'trimmed_{scale}_labeled_img']

        he = rearrange(torch.tensor(he, dtype=torch.float32), 'h w c -> c h w')
        he -= he.min()
        he /= he.max()

        self.he = he
        self.labeled_img = torch.tensor(labeled_img.astype(np.int32)).unsqueeze(dim=0)
        self.max_voxel_area = self.labeled_img[self.labeled_img==1].sum()

        self.transform = transform

        # expression
        if 'sparse' in str(type(self.adata.X)):
            self.X = torch.tensor(self.adata.X.toarray().astype(np.int32))
        else:
            self.X = torch.tensor(self.adata.X.astype(np.int32))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # images
        context_he, context_labeled_img = self.transform(self.he, self.labeled_img)
        context_labeled_img = context_labeled_img.to(torch.int32)

        he = TF.center_crop(context_he, self.size)
        labeled_img = TF.center_crop(context_labeled_img, self.size)

        context_he = TF.resize(context_he, self.size)
        context_labeled_img = TF.resize(context_labeled_img, self.size)

        voxel_idxs = torch.unique(labeled_img).to(torch.long)
        voxel_idxs = voxel_idxs[voxel_idxs!=0]
        n_voxels = len(voxel_idxs)
        masks, voxel_idxs = create_masks(labeled_img, voxel_idxs, self.max_voxel_area, thresh=.25)

        if voxel_idxs is None:
            voxel_idxs = torch.tensor([0], dtype=torch.long)
            masks = torch.zeros((1, he.shape[-2], he.shape[-1]), dtype=torch.bool)
            X = torch.zeros((1, self.X.shape[1]), dtype=torch.int32)
        else:
            voxel_idxs -= 1
            X = self.X[voxel_idxs]
            voxel_idxs += 1

        padding = self.max_voxels_per_sample - len(voxel_idxs)
        if padding < 0:
            raise RuntimeError(f'more voxels than max voxel size: {len(voxel_idxs)} , {self.max_voxels_per_sample} . increase max voxel size to nearest power of 2.')
        voxel_idxs = F.pad(voxel_idxs, (0, padding))
        X = torch.concat((X, torch.zeros((padding, X.shape[1]))))
        masks = torch.concat((masks, torch.zeros((padding, masks.shape[-2], masks.shape[-1]), dtype=torch.bool)))

        return {
            'he': he,
            'labeled_img': labeled_img,
            'context_he': context_he,
            'context_labeled_img': context_labeled_img,
            'voxel_idxs': voxel_idxs,
            'masks': masks,
            'exp': X,
            'n_voxels': n_voxels
        }

    def sanity_check(self, gene='IL7R'):
        d = self[0]
        print(f'keys: {d.keys()}')

        img = rearrange(d['he'].clone().detach().cpu().numpy(), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('H&E')
        plt.show()

        plt.imshow(d['labeled_img'][0])
        plt.title('labeled image')
        plt.show()

        img = rearrange(d['context_he'].clone().detach().cpu().numpy(), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('context H&E')
        plt.show()

        plt.imshow(d['context_labeled_img'][0])
        plt.title('context labeled image')
        plt.show()

        print('voxel idxs: ' + str(d['voxel_idxs']))
        print('labeled image voxel idxs: ' + str(torch.unique(d['labeled_img'])))

        print(f'masks shape: ' + str(d['masks'].shape))
        plt.imshow(d['masks'][0])
        plt.title('first voxel')
        plt.show()
        plt.imshow(d['masks'].sum(0))
        plt.title('summed masks')
        plt.show()

        print('expression counts shape: ' + str(d['exp'].shape))
        print(d['exp'])

        if gene in self.genes:
            idx =  self.genes.index(gene)
            x = project_expression(d['labeled_img'], d['exp'][:, idx:idx + 1], d['voxel_idxs'])
            plt.imshow(x[..., 0])
            plt.title(gene)
            plt.show()


class MultisampleSTDataset(Dataset):
    def __init__(self, ds_dict):
        super().__init__()
        self.samples = list(ds_dict.keys())
        self.ds_dict = ds_dict

        self.mapping = [(k, i) for k, ds in ds_dict.items() for i in zip(range(len(ds)))]
        
    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        k, i = self.mapping[idx]

        d = self.ds_dict[k][i]

        return d
    

class HEPredictionDataset(Dataset):
    """Tiled H&E dataset"""
    def __init__(self, he, size=256, context_res=2., transform=OverlaidHETransform(),
                 scale=.5, overlap_pct=.5):
        self.scale = scale
        self.size = size
        self.overlap_pct = overlap_pct
        self.step_size = int(size * overlap_pct)
        self.tile_width = int(size * context_res)
        self.transform = transform

        he = rearrange(torch.tensor(he, dtype=torch.float32), 'h w c -> c h w')
        self.full_res_he_shape = he.shape
        he -= he.min()
        he /= he.max()
        he = TF.resize(he, (int(he.shape[-2] * scale), int(he.shape[-1] * scale)))
        self.scaled_he_shape = he.shape
        he = self.transform(he)
        he = TF.pad(he, self.tile_width, padding_mode='reflect')
        self.he = he

        n_steps_width = (self.he.shape[-1] - self.tile_width) // self.step_size
        n_steps_height = (self.he.shape[-2] - self.tile_width) // self.step_size # assuming square tile
        self.coord_to_tile = {}
        self.coords = []
        self.absolute_coords = []
        for i in range(n_steps_height):
            for j in range(n_steps_width):
                r, c = i * self.step_size, j * self.step_size
                tile = TF.crop(self.he, r, c, self.tile_width, self.tile_width)
                self.coord_to_tile[(i, j)] = tile
                self.coords.append((i, j))
                self.absolute_coords.append((r, c))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        i, j = self.coords[idx]
        r, c = self.absolute_coords[idx]
        context_he = self.coord_to_tile[(i, j)]
        he = TF.center_crop(context_he, (self.size, self.size))
        context_he = TF.resize(context_he, (self.size, self.size))

        return {
            'coord': torch.tensor([i, j], dtype=torch.long),
            'absolute': torch.tensor([r, c], dtype=torch.long),
            'he': he,
            'context_he': context_he,
        }
    
    def retile(self, coord_to_tile, trim_to_original=True, scale_to_original=True):
        tile = next(iter(coord_to_tile.values()))
        max_i = max([i for i, _ in coord_to_tile.keys()])
        max_j = max([j for _, j in coord_to_tile.keys()])
        new = None
        for i in range(max_i):
            row = None
            for j in range(max_j):
                tile = coord_to_tile[i, j]
                tile = TF.center_crop(
                    tile, (int(self.size * self.overlap_pct), int(self.size * self.overlap_pct)))
                if row is None:
                    row = tile
                else:
                    row = torch.concat((row, tile), dim=-1)
            if new is None:
                new = row
            else:
                new = torch.concat((new, row), dim=-2)
        if trim_to_original:
            new = new[..., self.size:self.size + self.scaled_he_shape[-2], self.size:self.size + self.scaled_he_shape[-1]]
        if scale_to_original:
            new = TF.resize(new, (self.full_res_he_shape[-2], self.full_res_he_shape[-1]))

        return new


    def sanity_check(self):
        print(f'num tiles: {len(self.coords)}, step size: {self.step_size}, tile width: {self.tile_width}')

        d = self[0]
        print(f'keys: {d.keys()}')

        img = rearrange(d['he'].clone().detach().cpu().numpy(), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('H&E')
        plt.show()

        img = rearrange(d['context_he'].clone().detach().cpu().numpy(), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('context H&E')
        plt.show()

        img = rearrange(self.retile(self.coord_to_tile), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('retiled to full res H&E')
        plt.show()

        print(f'original H&E shape: {self.full_res_he_shape}')
        print(f'retiled H&E shape: {img.shape}')