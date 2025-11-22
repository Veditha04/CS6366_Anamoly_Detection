import os
import sys
import glob
import argparse

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from model import MultiScaleUNetAE


def load_image(img_path, image_size=256):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0)  # (1,3,H,W)
    return x, img


def anomaly_heatmap(x, recon):
    # pixelwise squared error averaged over channels
    err = (x - recon) ** 2
    heat = err.mean(dim=1).squeeze(0).detach().cpu().numpy()
    # normalize to [0,1]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat


def find_default_test_image(dataset_root, category):
    """
    If user doesn't pass --img, we automatically choose a test image.
    Preference: test/good/*.png
    """
    good_dir = os.path.join(dataset_root, category, "test", "good")
    if os.path.isdir(good_dir):
        imgs = sorted(glob.glob(os.path.join(good_dir, "*.png")))
        if imgs:
            return imgs[0]

    # fallback: any test image
    test_dir = os.path.join(dataset_root, category, "test")
    imgs = sorted(glob.glob(os.path.join(test_dir, "**", "*.png"), recursive=True))
    if imgs:
        return imgs[0]

    return None


def run_inference(checkpoint, img_path, out_dir="results",
                  image_size=256, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)

    model = MultiScaleUNetAE().to(device)

    # ----- load checkpoint if exists -----
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        print("No checkpoint found — running with random weights (demo mode).")
        if checkpoint:
            print(f"Expected checkpoint at: {checkpoint}")

    model.eval()

    # ----- load image -----
    x, pil_img = load_image(img_path, image_size=image_size)
    x = x.to(device)

    # ----- forward -----
    with torch.no_grad():
        recon = model(x)

    heat = anomaly_heatmap(x, recon)

    # ----- save visualization -----
    fig_path = os.path.join(out_dir, "inference_result.png")

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(pil_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Reconstruction")
    recon_img = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imshow(recon_img)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Anomaly Heatmap")
    plt.imshow(pil_img)
    plt.imshow(heat, alpha=0.6)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("Saved:", fig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-scale autoencoder inference and save anomaly heatmap."
    )

    parser.add_argument("--dataset", "-d",
                        default="data/mvtec_anomaly_detection",
                        help="Path to mvtec_anomaly_detection folder")
    parser.add_argument("--category", "-cat",
                        default="bottle",
                        help="Category name (bottle/capsule/hazelnut/screw/...)")
    parser.add_argument("--checkpoint", "-c",
                        default="checkpoints/bottle_best.pth",
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--img", "-i",
                        default=None,
                        help="Path to input test image (.png). If not given, picks one automatically.")
    parser.add_argument("--out", "-o",
                        default="results",
                        help="Output directory for results")
    parser.add_argument("--size", "-s",
                        type=int, default=256,
                        help="Image size")
    parser.add_argument("--device",
                        default="cpu",
                        help="cpu or cuda")

    args = parser.parse_args()

    # pick default image if none provided
    if args.img is None:
        default_img = find_default_test_image(args.dataset, args.category)
        if default_img is None:
            print("ERROR: Could not find any test images.")
            print("Check dataset path + category.")
            sys.exit(2)
        args.img = default_img
        print("Auto-selected test image:", args.img)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_inference(
        checkpoint=args.checkpoint,
        img_path=args.img,
        out_dir=args.out,
        image_size=args.size,
        device=device
    )
