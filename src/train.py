import os
import argparse
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from dataset import MVTecADDataset
from model import MultiScaleUNetAE




def set_seed(seed=42):
   import random
   import numpy as np
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)




def train_one_epoch(model, loader, optimizer, criterion, device):
   model.train()
   running_loss = 0.0


   for imgs, labels, defect_types, paths in loader:
       # imgs: [B,3,H,W]
       imgs = imgs.to(device)


       optimizer.zero_grad()
       recons = model(imgs)


       loss = criterion(recons, imgs)
       loss.backward()
       optimizer.step()


       running_loss += loss.item() * imgs.size(0)


   return running_loss / len(loader.dataset)




@torch.no_grad()
def val_one_epoch(model, loader, criterion, device):
   model.eval()
   running_loss = 0.0


   for imgs, labels, defect_types, paths in loader:
       imgs = imgs.to(device)
       recons = model(imgs)
       loss = criterion(recons, imgs)
       running_loss += loss.item() * imgs.size(0)


   return running_loss / len(loader.dataset)




def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--dataset", type=str, required=True,
                       help="Path to mvtec_anomaly_detection folder")
   parser.add_argument("--category", type=str, required=True,
                       help="bottle / capsule / hazelnut / screw etc.")
   parser.add_argument("--epochs", type=int, default=10)
   parser.add_argument("--batch_size", type=int, default=8)
   parser.add_argument("--lr", type=float, default=1e-3)
   parser.add_argument("--image_size", type=int, default=256)
   parser.add_argument("--num_workers", type=int, default=2)
   parser.add_argument("--save_dir", type=str, default="checkpoints")
   parser.add_argument("--seed", type=int, default=42)
   args = parser.parse_args()


   set_seed(args.seed)


   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print("Using device:", device)


   # ---- datasets ----
   train_ds = MVTecADDataset(
       root_dir=args.dataset,
       category=args.category,
       split="train",
       image_size=args.image_size
   )


   # validate on normal ("good") images from test
   val_ds = MVTecADDataset(
       root_dir=args.dataset,
       category=args.category,
       split="test",
       image_size=args.image_size
   )


   # Filter val to only good images
   val_ds.samples = [s for s in val_ds.samples if s[1] == 0]


   train_loader = DataLoader(
       train_ds,
       batch_size=args.batch_size,
       shuffle=True,
       num_workers=args.num_workers,
       pin_memory=True
   )


   val_loader = DataLoader(
       val_ds,
       batch_size=args.batch_size,
       shuffle=False,
       num_workers=args.num_workers,
       pin_memory=True
   )


   print("Train samples:", len(train_ds))
   print("Val (good only) samples:", len(val_ds))


   # ---- model ----
   model = MultiScaleUNetAE().to(device)


   criterion = nn.MSELoss()
   optimizer = optim.Adam(model.parameters(), lr=args.lr)


   save_dir = Path(args.save_dir)
   save_dir.mkdir(parents=True, exist_ok=True)


   best_val = float("inf")


   # ---- training loop ----
   for epoch in range(1, args.epochs + 1):
       train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
       val_loss = val_one_epoch(model, val_loader, criterion, device)


       print(f"Epoch {epoch}/{args.epochs} | "
             f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")


       # save last every epoch
       last_ckpt = save_dir / f"{args.category}_last.pth"
       torch.save(model.state_dict(), last_ckpt)


       # save best
       if val_loss < best_val:
           best_val = val_loss
           best_ckpt = save_dir / f"{args.category}_best.pth"
           torch.save(model.state_dict(), best_ckpt)
           print(f"Saved best checkpoint: {best_ckpt}")


   print("Training completed.")
   print("Best validation loss:", best_val)




if __name__ == "__main__":
   main()


