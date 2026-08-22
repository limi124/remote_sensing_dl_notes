from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose,Resize,ToTensor,Normalize
from torchgeo.models import scalemae_large_patch16,ScaleMAELarge16_Weights

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from common import atomic_checkpoint,seed_everything,write_json

def stratified_indices(ds,max_samples,seed):
    g=torch.Generator().manual_seed(seed); groups={}
    for i,(_,y) in enumerate(ds.samples): groups.setdefault(y,[]).append(i)
    tr=[]; va=[]; per=max_samples//len(groups)
    for ids in groups.values():
        order=torch.randperm(len(ids),generator=g).tolist(); chosen=[ids[i] for i in order[:per]]; cut=int(.8*len(chosen)); tr+=chosen[:cut]; va+=chosen[cut:]
    return tr,va

def build(pretrained,classes):
    weights=ScaleMAELarge16_Weights.FMOW_RGB if pretrained else None
    model=scalemae_large_patch16(weights=weights)
    in_features=model.head.in_features; model.head=nn.Linear(in_features,classes)
    for name,p in model.named_parameters(): p.requires_grad=name.startswith("head")
    return model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="data/eurosat/EuroSAT_RGB"); p.add_argument("--output",default="outputs/part8_scalemae")
    p.add_argument("--epochs",type=int,default=5); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--max-samples",type=int,default=3000); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); seed_everything(a.seed); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); write_json(out/"config.json",vars(a)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf=Compose([Resize((224,224)),ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])]); ds=ImageFolder(a.data_root,transform=tf); ti,vi=stratified_indices(ds,a.max_samples,a.seed)
    tr=DataLoader(Subset(ds,ti),a.batch_size,shuffle=True,num_workers=0,pin_memory=True); va=DataLoader(Subset(ds,vi),a.batch_size,num_workers=0,pin_memory=True)
    results={}; combined_history=[]; prediction_batches={}
    for label,pre in (("random_frozen",False),("scalemae_frozen",True)):
        model=build(pre,len(ds.classes)).to(device); opt=torch.optim.AdamW(model.head.parameters(),lr=1e-3); rows=[]; best=0
        for ep in range(1,a.epochs+1):
            model.train(); correct=n=0; loss_sum=0
            for x,y in tr:
                x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); logits=model(x); loss=nn.functional.cross_entropy(logits,y); loss.backward(); opt.step()
                loss_sum+=loss.item()*len(x); correct+=(logits.argmax(1)==y).sum().item(); n+=len(x)
            model.eval(); vc=vn=0
            with torch.no_grad():
                for x,y in va: x,y=x.to(device),y.to(device); vc+=(model(x).argmax(1)==y).sum().item(); vn+=len(x)
            row={"model":label,"epoch":ep,"train_loss":loss_sum/n,"train_acc":correct/n,"val_acc":vc/vn}; rows.append(row); combined_history.append(row); print(label,json.dumps(row))
            payload={"head":model.head.state_dict(),"epoch":ep,"val_acc":max(best,row["val_acc"]),"pretrained":pre}
            atomic_checkpoint(out/f"{label}_last.pt",payload)
            if row["val_acc"]>best:
                best=row["val_acc"]; atomic_checkpoint(out/f"{label}_best.pt",payload)
                if pre: atomic_checkpoint(out/"best.pt",payload)
            if pre: atomic_checkpoint(out/"last.pt",payload)
        with (out/f"{label}_history.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
        model.eval(); x,y=next(iter(va))
        with torch.no_grad(): pred=model(x.to(device)).argmax(1).cpu()
        prediction_batches[label]=(x[:8].cpu(),y[:8].cpu(),pred[:8])
        results[label]={"best_val_acc":best,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"total_parameters":sum(p.numel() for p in model.parameters())}
        del model; torch.cuda.empty_cache()
    # Keep one auditable, combined CSV while the plot below reports validation
    # accuracy (not segmentation mIoU).
    with (out/"history.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=combined_history[0].keys()); w.writeheader(); w.writerows(combined_history)
    write_json(out/"metrics.json",results); fig,ax=plt.subplots(figsize=(5,4)); ax.bar(results.keys(),[v["best_val_acc"] for v in results.values()]); ax.set_ylim(0,1); ax.set_ylabel("validation accuracy"); fig.tight_layout(); fig.savefig(out/"training_curves.png",dpi=160); plt.close(fig)
    x,y,pred=prediction_batches["scalemae_frozen"]; fig,axes=plt.subplots(2,4,figsize=(10,5))
    for i,ax in enumerate(axes.flat):
        img=x[i].permute(1,2,0); img=img*torch.tensor([.229,.224,.225])+torch.tensor([.485,.456,.406])
        ax.imshow(img.clamp(0,1)); ax.set_title(f"true={ds.classes[y[i]]}\\npred={ds.classes[pred[i]]}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out/"predictions.png",dpi=160); plt.close(fig)
    print("complete",results)
if __name__=="__main__": main()
