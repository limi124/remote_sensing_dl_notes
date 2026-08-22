from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base=24):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        x = self.b(self.pool(e3))
        x = self.d3(torch.cat([self.u3(x), e3], 1))
        x = self.d2(torch.cat([self.u2(x), e2], 1))
        x = self.d1(torch.cat([self.u1(x), e1], 1))
        return self.head(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1 if r == 1 else 3,
                                    padding=0 if r == 1 else r, dilation=r, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)) for r in rates
        ])
        self.project = nn.Sequential(nn.Conv2d(out_ch * len(rates), out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], 1))


class TinyDeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base=24):
        super().__init__()
        self.low = ConvBlock(in_channels, base)
        self.encoder = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base, base * 2),
                                     nn.MaxPool2d(2), ConvBlock(base * 2, base * 4),
                                     nn.MaxPool2d(2), ConvBlock(base * 4, base * 8))
        self.aspp = ASPP(base * 8, base * 2)
        self.low_proj = nn.Sequential(nn.Conv2d(base, base, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU())
        self.decoder = ConvBlock(base * 3, base * 2)
        self.head = nn.Conv2d(base * 2, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        high = self.aspp(self.encoder(low))
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        out = self.decoder(torch.cat([high, self.low_proj(low)], 1))
        return F.interpolate(self.head(out), size=size, mode="bilinear", align_corners=False)


class SiameseUNet(nn.Module):
    def __init__(self, num_classes=2, base=24):
        super().__init__()
        self.encoder = nn.Sequential(ConvBlock(3, base), nn.MaxPool2d(2),
                                     ConvBlock(base, base * 2), nn.MaxPool2d(2),
                                     ConvBlock(base * 2, base * 4))
        self.decoder = nn.Sequential(ConvBlock(base * 4, base * 2),
                                     nn.ConvTranspose2d(base * 2, base, 2, 2),
                                     ConvBlock(base, base), nn.ConvTranspose2d(base, base, 2, 2),
                                     nn.Conv2d(base, num_classes, 1))

    def forward(self, a, b):
        fa, fb = self.encoder(a), self.encoder(b)
        return self.decoder(torch.abs(fa - fb))


def build_segmentation_model(name: str, num_classes: int):
    if name == "unet":
        return TinyUNet(num_classes=num_classes)
    if name == "deeplabv3plus":
        return TinyDeepLabV3Plus(num_classes=num_classes)
    if name == "segformer_b0":
        from transformers import SegformerForSemanticSegmentation
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=num_classes, ignore_mismatched_sizes=True
        )
    raise ValueError(f"Unknown model: {name}")


def segmentation_logits(model, images, model_name: str):
    out = model(images)
    logits = out.logits if hasattr(out, "logits") else out
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return logits
