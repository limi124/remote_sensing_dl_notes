from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument("--root",default="outputs/part9_reproduction"); a=p.parse_args(); root=Path(a.root)
models=("unet","deeplabv3plus","segformer_b0"); rows=[]
for model in models:
    for seed in (13,42,87):
        path=root/model/f"seed_{seed}"/"metrics.json"
        if not path.exists(): raise FileNotFoundError(path)
        m=json.loads(path.read_text()); rows.append({"model":model,"seed":seed,"miou":m["miou"],"dice":m["mean_dice"],"parameters":m["parameters"],"peak_gpu_mb":m["peak_gpu_mb"],"inference_ms_per_image":m["inference_ms_per_image"]})
summary=[]
for model in models:
    group=[r for r in rows if r["model"]==model]
    summary.append({"model":model,"miou_mean":np.mean([r["miou"] for r in group]),"miou_std":np.std([r["miou"] for r in group],ddof=1),"dice_mean":np.mean([r["dice"] for r in group]),"parameters":group[0]["parameters"],"peak_gpu_mb":max(r["peak_gpu_mb"] for r in group),"inference_ms_per_image_mean":np.mean([r["inference_ms_per_image"] for r in group])})
for name,data in (("runs.csv",rows),("summary.csv",summary)):
    with (root/name).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=data[0]); w.writeheader(); w.writerows(data)
(root/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
fig,ax=plt.subplots(figsize=(7,4)); ax.bar([r["model"] for r in summary],[r["miou_mean"] for r in summary],yerr=[r["miou_std"] for r in summary],capsize=5); ax.set_ylabel("mIoU mean ± SD (3 seeds)"); fig.tight_layout(); fig.savefig(root/"comparison.png",dpi=160); plt.close(fig)
print(json.dumps(summary,indent=2))
