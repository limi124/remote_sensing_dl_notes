import sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from models import TinyUNet,TinyDeepLabV3Plus,SiameseUNet,build_segmentation_model,segmentation_logits

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
x=torch.randn(2,3,128,128,device=device); y=torch.randint(0,5,(2,128,128),device=device)
for name,model in [("unet",TinyUNet()),("deeplabv3plus",TinyDeepLabV3Plus())]:
    model=model.to(device); out=model(x); loss=torch.nn.functional.cross_entropy(out,y); loss.backward()
    assert out.shape==(2,5,128,128); print(name,"ok",float(loss))
change=SiameseUNet().to(device); out=change(x,x.roll(2,-1)); loss=out.mean(); loss.backward(); assert out.shape==(2,2,128,128)
print("siamese ok",tuple(out.shape),"device",device)
