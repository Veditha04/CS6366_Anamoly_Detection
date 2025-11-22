import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat

def run_inference(checkpoint, img_path, out_dir="results", image_size=256, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)

    model = MultiScaleUNetAE().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    x, pil_img = load_image(img_path, image_size=image_size)
    x = x.to(device)

    with torch.no_grad():
        recon = model(x)

    heat = anomaly_heatmap(x, recon)

    # Save visualization
    fig_path = os.path.join(out_dir, "inference_result.png")
    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(pil_img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Reconstruction")
    recon_img = recon.squeeze(0).permute(1,2,0).cpu().numpy()
    plt.imshow(recon_img)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Anomaly Heatmap")
    plt.imshow(pil_img)
    plt.imshow(heat, alpha=0.6)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("Saved:", fig_path)


if __name__ == "__main__":
    # Example usage (will work once you have a checkpoint)
    run_inference(
        checkpoint="results/bottle_best.pth",
        img_path="data/sample_test.png",
        out_dir="results"
    )

