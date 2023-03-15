import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ColorJitter, RandomCrop, RandomRotation, CenterCrop, Compose, Normalize

class OverlaidHETransform(object):
    def __init__(self, p=.95, size=(256,256), degrees=180,
                 brightness=.1, contrast=.1, saturation=.1, hue=.1,
                 normalize=True, means=(0.771, 0.651, 0.752), stds=(0.229, 0.288, 0.224)):
        
        self.color_transform = ColorJitter(brightness=brightness, contrast=contrast,
                                           saturation=saturation, hue=hue)
        self.spatial_transform = Compose([
            RandomCrop((size[0] * 2, size[1] * 2), padding=size, padding_mode='reflect'),
            RandomRotation(degrees),
            CenterCrop(size)
        ])
        
        if normalize:
            self.normalize = Normalize(means, stds) # from HT397B1-H2 ffpe H&E image
        else:
            self.normalize = nn.Identity()
 
        self.p = p
        
    def __call__(self, he, overlay):
        """
        he - (3, H, W)
        overlay - (n, H, W)
        """
        if torch.rand(size=(1,)) < self.p:
            x = torch.concat((he, overlay))
            x = self.spatial_transform(x)
            
            he, overlay = x[:3], x[3:]
        
            he = self.color_transform(he)
            
        he = self.normalize(he)
        
        return he, overlay


class NormalizeHETransform(object):
    def __init__(self, normalize=True,
                 means=(0.771, 0.651, 0.752), stds=(0.229, 0.288, 0.224)):
        if normalize:
            self.normalize = Normalize(means, stds) # from HT397B1-H2 ffpe H&E image
        else:
            self.normalize = nn.Identity()
  
    def __call__(self, he):
        """
        he - (3, H, W)
        """
        return self.normalize(he)