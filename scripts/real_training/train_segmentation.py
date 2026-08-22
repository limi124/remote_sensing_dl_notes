from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from common import atomic_checkpoint, benchmark_inference, confusion_update, metrics_from_confusion, save_history, seed_everything, write_json
from landcover import LandCoverDataset
from models import build_segmentation_model, segmentation_logits


def run_epoch(model, loader, optimizer, scaler, device, classes, name, train):
    model.train(train); total_loss = 0.0; count = 0; cm = torch.zeros(classes, classes, dtype=torch.long)
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if train: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = segmentation_logits(model, images, name)
                loss = F.cross_entropy(logits, masks)
            if train:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * images.size(0); count += images.size(0)
            confusion_update(cm, logits.argmax(1), masks, classes)
    return total_loss / count, metrics_from_confusion(cm)


def save_predictions(model, loader, device, name, output):
    model.eval(); images, masks = next(iter(loader)); images = images.to(device)
    with torch.no_grad(): preds = segmentation_logits(model, images, name).argmax(1).cpu()
    images = images.cpu(); n = min(4, len(images)); fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
    if n == 1: axes = axes[None, :]
    for i in range(n):
        axes[i,0].imshow(images[i].permute(1,2,0).clamp(0,1)); axes[i,0].set_title("image")
        axes[i,1].imshow(masks[i], vmin=0, vmax=4); axes[i,1].set_title("target")
        axes[i,2].imshow(preds[i], vmin=0, vmax=4); axes[i,2].set_title("prediction")
        for ax in axes[i]: ax.axis("off")
    fig.tight_layout(); fig.savefig(output / "predictions.png", dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["unet","deeplabv3plus","segformer_b0"], required=True)
    p.add_argument("--data-root", default="data/landcoverai"); p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=8); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--size", type=int, default=256); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--patience", type=int, default=4)
    p.add_argument("--train-limit", type=int, default=1200); p.add_argument("--val-limit", type=int, default=300)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--resume", action="store_true")
    p.add_argument("--overfit", action="store_true", help="Validate on the same fixed training subset (sanity check only).")
    args = p.parse_args(); seed_everything(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); write_json(output/"config.json", vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); classes = 5
    train_ds = LandCoverDataset(args.data_root,"train",args.size,True,args.train_limit)
    # For the explicit overfit test, use the exact same tiles but disable random
    # augmentation during validation.  Formal experiments always use the spatially
    # independent validation split above.
    val_ds = (LandCoverDataset(args.data_root,"train",args.size,False,args.train_limit)
              if args.overfit else LandCoverDataset(args.data_root,"val",args.size,False,args.val_limit))
    train = DataLoader(train_ds,args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    val = DataLoader(val_ds,args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    model = build_segmentation_model(args.model,classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    start=1; best=-1.0; history=[]
    if args.resume and (output/"last.pt").exists():
        ck=torch.load(output/"last.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"]); start=ck["epoch"]+1; best=ck["best"]; history=ck["history"]
    stale=0
    for epoch in range(start,args.epochs+1):
        t=time.perf_counter(); tr_loss,tr_m=run_epoch(model,train,optimizer,scaler,device,classes,args.model,True)
        va_loss,va_m=run_epoch(model,val,optimizer,scaler,device,classes,args.model,False)
        row={"epoch":epoch,"train_loss":tr_loss,"val_loss":va_loss,"train_miou":tr_m["miou"],"val_miou":va_m["miou"],"seconds":time.perf_counter()-t}
        history.append(row); payload={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"best":max(best,va_m["miou"]),"history":history}
        atomic_checkpoint(output/"last.pt",payload)
        if va_m["miou"]>best:
            best=va_m["miou"]; stale=0; atomic_checkpoint(output/"best.pt",payload); write_json(output/"metrics.json",va_m)
        else: stale+=1
        print(json.dumps(row));
        if stale>=args.patience: break
    ck=torch.load(output/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"])
    save_history(history,output); save_predictions(model,val,device,args.model,output)
    params=sum(p.numel() for p in model.parameters()); metrics=json.loads((output/"metrics.json").read_text())
    example,_=next(iter(val)); example=example[:1].to(device)
    metrics.update({"parameters":params,"device":str(device),"peak_gpu_mb":torch.cuda.max_memory_allocated()/1024**2 if device.type=="cuda" else 0,
                    "inference_ms_per_image":benchmark_inference(model,example,device)})
    write_json(output/"metrics.json",metrics); print("complete",output,metrics)


if __name__ == "__main__": main()
