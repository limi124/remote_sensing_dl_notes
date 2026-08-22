from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchgeo.datasets import LandCoverAI


class LandCoverDataset(Dataset):
    def __init__(self, root, split, size=256, augment=False, limit=0):
        self.base = LandCoverAI(root=root, split=split, download=False)
        self.size, self.augment = size, augment
        self.length = min(limit, len(self.base)) if limit else len(self.base)

    def __len__(self): return self.length

    def __getitem__(self, index):
        sample = self.base[index]
        image, mask = sample["image"].float() / 255.0, sample["mask"].long()
        if mask.ndim == 3: mask = mask.squeeze(0)
        image = TF.resize(image, [self.size, self.size], antialias=True)
        mask = TF.resize(mask.unsqueeze(0), [self.size, self.size], interpolation=InterpolationMode.NEAREST).squeeze(0)
        if self.augment and torch.rand(()) < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if self.augment and torch.rand(()) < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        return image, mask
