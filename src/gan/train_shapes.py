# Procedural dataset: square, circle and triangle. Educational GAN experiment.
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .models import ConditionalGenerator, ConditionalDiscriminator

class Shapes(Dataset):
    def __init__(self,n=6000): self.n=n
    def __len__(self): return self.n
    def __getitem__(self,i):
        rng=np.random.default_rng(i); label=i%3
        im=Image.new("RGB",(64,64),(0,0,0)); d=ImageDraw.Draw(im)
        color=tuple(int(v) for v in rng.integers(70,256,3))
        if label==0:
            x,y=rng.integers(8,20,2); d.rectangle((x,y,64-x,64-y),fill=color)
        elif label==1:
            x,y=rng.integers(8,18,2); d.ellipse((x,y,64-x,64-y),fill=color)
        else: d.polygon([(32,5),(5,59),(59,59)],fill=color)
        arr=np.asarray(im).astype("float32")/127.5-1
        return torch.tensor(arr).permute(2,0,1),torch.tensor(label)

def train(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    loader=DataLoader(Shapes(args.samples),batch_size=args.batch,shuffle=True)
    G=ConditionalGenerator(attention=args.attention).to(device); D=ConditionalDiscriminator(attention=args.attention).to(device)
    og=torch.optim.Adam(G.parameters(),lr=2e-4,betas=(.5,.999))
    od=torch.optim.Adam(D.parameters(),lr=2e-4,betas=(.5,.999)); loss=nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        for real,y in loader:
            real,y=real.to(device),y.to(device); n=len(y)
            z=torch.randn(n,64,device=device); fake=G(z,y)
            od.zero_grad(set_to_none=True)
            dl=loss(D(real,y),torch.ones(n,1,device=device))+loss(D(fake.detach(),y),torch.zeros(n,1,device=device))
            dl.backward(); od.step()
            og.zero_grad(set_to_none=True); gl=loss(D(fake,y),torch.ones(n,1,device=device))
            gl.backward(); og.step()
        print(f"Epoch {epoch+1}/{args.epochs} D={dl.item():.4f} G={gl.item():.4f}")
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
    "generator": G.state_dict(),
    "discriminator": D.state_dict(),
    "attention": args.attention,
    "num_classes": 3,
    "latent": 64,
    "image_size": 64
}, out)

    print("Saved", out)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch",type=int,default=64)
    p.add_argument("--samples",type=int,default=6000); p.add_argument("--attention",action="store_true")
    p.add_argument("--out",default="outputs/cgan_shapes.pt"); train(p.parse_args())
