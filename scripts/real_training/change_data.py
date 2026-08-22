from __future__ import annotations

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class LEVIRDataset(Dataset):
    def __init__(self, root, split, size=256, augment=False, limit=0):
        base = Path(root) / split
        if not base.exists():
            candidates = list(Path(root).rglob(split))
            base = next((p for p in candidates if (p/"A").exists()), base)
        self.a_dir, self.b_dir = base/"A", base/"B"
        self.label_dir = base/("label" if (base/"label").exists() else "labels")
        self.names = sorted(p.name for p in self.a_dir.glob("*.png") if (self.b_dir/p.name).exists() and (self.label_dir/p.name).exists())
        if limit: self.names = self.names[:limit]
        if not self.names: raise FileNotFoundError(f"No aligned LEVIR-CD triples under {base}")
        self.size, self.augment = size, augment

    def __len__(self): return len(self.names)
    def __getitem__(self,index):
        name=self.names[index]
        a=TF.to_tensor(Image.open(self.a_dir/name).convert("RGB")); b=TF.to_tensor(Image.open(self.b_dir/name).convert("RGB"))
        mask=torch.from_numpy(__import__("numpy").array(Image.open(self.label_dir/name).convert("L"),copy=True)).long()
        a=TF.resize(a,[self.size,self.size],antialias=True); b=TF.resize(b,[self.size,self.size],antialias=True)
        mask=TF.resize(mask.unsqueeze(0),[self.size,self.size],interpolation=InterpolationMode.NEAREST).squeeze(0)
        mask=(mask>0).long()
        if self.augment and torch.rand(())<.5: a,b,mask=TF.hflip(a),TF.hflip(b),TF.hflip(mask)
        if self.augment and torch.rand(())<.5: a,b,mask=TF.vflip(a),TF.vflip(b),TF.vflip(mask)
        return a,b,mask
