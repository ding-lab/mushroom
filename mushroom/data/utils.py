from dataclasses import dataclass

import torch.nn as nn
from torch.utils.data import Dataset

from mushroom.data.inference import InferenceSectionDataset

@dataclass
class LearnerData:
    section_to_img: dict
    train_transform: object
    inference_transform: object
    train_ds: Dataset
    inference_ds: InferenceSectionDataset
    channels: list = None