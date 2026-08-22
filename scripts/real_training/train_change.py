from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from change_data import LEVIRDataset
from common import atomic_checkpoint,confusion_update,metrics_from_confusion,save_history,seed_everything,write_json
from models import SiameseUNet

def epoch(model,loader,opt,scaler,device,train, class_weights):
    model.train(train); cm=torch.zeros(2,2,dtype=torch.long); loss_sum=n=0
    context=torch.enable_grad if train else torch.no_grad
    with context():
        for a,b,y in loader:
            a,b,y=a.to(device),b.to(device),y.to(device)
            if train: opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                logits=model(a,b); loss=F.cross_entropy(logits,y,weight=class_weights)
            if train: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum+=loss.item()*a.size(0); n+=a.size(0); confusion_update(cm,logits.argmax(1),y,2)
    return loss_sum/n,metrics_from_confusion(cm)

def change_metrics(metrics):
    # Rows are targets and columns are predictions; class 1 denotes change.
    iou=metrics["class_iou"][1]; f1=metrics["class_dice"][1]
    return {**metrics, "iou":iou, "f1":f1}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="data/levir_cd"); p.add_argument("--output",default="outputs/part7_change")
    p.add_argument("--epochs",type=int,default=8); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--size",type=int,default=256)
    p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--seed",type=int,default=42); p.add_argument("--train-limit",type=int,default=1200); p.add_argument("--val-limit",type=int,default=300)
    p.add_argument("--overfit",action="store_true",help="Validate on the same fixed training subset (sanity check only).")
    p.add_argument("--resume",action="store_true",help="Resume model, optimizer, and epoch history from last.pt.")
    a=p.parse_args(); seed_everything(a.seed); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); write_json(out/"config.json",vars(a))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr=DataLoader(LEVIRDataset(a.data_root,"train",a.size,True,a.train_limit),a.batch_size,shuffle=True,num_workers=0,pin_memory=True)
    va_ds=LEVIRDataset(a.data_root,"train",a.size,False,a.train_limit) if a.overfit else LEVIRDataset(a.data_root,"val",a.size,False,a.val_limit)
    va=DataLoader(va_ds,a.batch_size,num_workers=0,pin_memory=True)
    model=SiameseUNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    start,best,history=1,-1,[]
    if a.resume and (out/"last.pt").exists():
        ck=torch.load(out/"last.pt",map_location=device,weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start,best,history=ck["epoch"]+1,ck["best"],ck["history"]
    for ep in range(start,a.epochs+1):
        # LEVIR-CD is strongly imbalanced: weight the sparse change pixels so
        # the model cannot maximize accuracy by predicting only "no change".
        class_weights=torch.tensor([1.0,5.0],device=device)
        t=time.perf_counter(); tl,tm=epoch(model,tr,opt,scaler,device,True,class_weights); vl,vm=epoch(model,va,opt,scaler,device,False,class_weights); vm=change_metrics(vm)
        row={"epoch":ep,"train_loss":tl,"val_loss":vl,"train_miou":tm["miou"],"val_miou":vm["miou"],"seconds":time.perf_counter()-t}; history.append(row)
        payload={"epoch":ep,"model":model.state_dict(),"optimizer":opt.state_dict(),"best":max(best,vm["miou"]),"history":history}; atomic_checkpoint(out/"last.pt",payload)
        if vm["miou"]>best: best=vm["miou"]; atomic_checkpoint(out/"best.pt",payload); write_json(out/"metrics.json",vm)
        print(json.dumps(row))
    save_history(history,out); model.load_state_dict(torch.load(out/"best.pt",map_location=device,weights_only=False)["model"]); model.eval()
    aa,bb,yy=next(iter(va));
    with torch.no_grad(): pp=model(aa.to(device),bb.to(device)).argmax(1).cpu()
    n=min(4,len(aa)); fig,ax=plt.subplots(n,4,figsize=(12,3*n)); ax=ax[None,:] if n==1 else ax
    for i in range(n):
        for j,img in enumerate((aa[i].permute(1,2,0),bb[i].permute(1,2,0),yy[i],pp[i])): ax[i,j].imshow(img,cmap=None if j<2 else "gray"); ax[i,j].axis("off")
    fig.tight_layout(); fig.savefig(out/"predictions.png",dpi=160); plt.close(fig)
    m=json.loads((out/"metrics.json").read_text())
    # Recompute a compact binary confusion matrix and the requested detection metrics.
    cm=torch.zeros(2,2,dtype=torch.long)
    with torch.no_grad():
        for a,b,y in va: confusion_update(cm,model(a.to(device),b.to(device)).argmax(1),y.to(device),2)
    tp,fp,fn=int(cm[1,1]),int(cm[0,1]),int(cm[1,0])
    m.update({"precision":tp/(tp+fp+1e-12),"recall":tp/(tp+fn+1e-12),"confusion_matrix":cm.tolist(),
              "parameters":sum(p.numel() for p in model.parameters()),"device":str(device),"peak_gpu_mb":torch.cuda.max_memory_allocated()/1024**2})
    write_json(out/"metrics.json",m); print("complete",m)
    fig,ax=plt.subplots(figsize=(4,3)); im=ax.imshow(cm.numpy(),cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j,i,str(int(cm[i,j])),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["no change","change"],yticklabels=["no change","change"],xlabel="prediction",ylabel="target")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=160); plt.close(fig)
if __name__=="__main__": main()
