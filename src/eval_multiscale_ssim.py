import os
import glob
import random

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import cv2
from skimage.metrics import structural_similarity as ssim

from models import MultiScaleAutoencoder

CATEGORIES = ["bottle", "hazelnut", "cable", "tile"]


def build_transform(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])


def get_test_image_paths(root_dir, categories):
    items = []
    for cat in categories:
        cat_dir = os.path.join(root_dir, cat, "test")
        if not os.path.isdir(cat_dir):
            continue

        for defect_type in sorted(os.listdir(cat_dir)):
            defect_dir = os.path.join(cat_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue

            img_paths = sorted(glob.glob(os.path.join(defect_dir, "*.png")))
            label = 0 if defect_type == "good" else 1
            for p in img_paths:
                items.append((p, label, cat, defect_type))
    return items


def load_image(path, transform, img_size=256):
    try:
        img = Image.open(path).convert("RGB")
        img = transform(img)
    except Exception:
        img = torch.zeros(3, img_size, img_size, dtype=torch.float32)
    return img


def compute_ssim_error(img, recon):
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    rec_np = (recon.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    rec_gray = cv2.cvtColor(rec_np, cv2.COLOR_RGB2GRAY)

    score, diff = ssim(img_gray, rec_gray, full=True)
    diff = (1.0 - diff)
    diff = cv2.GaussianBlur(diff, (21, 21), 3)
    return diff


def compute_ssim_score_and_map(model, device, img_tensor):
    model.eval()
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        recon = model(x).squeeze(0).cpu()

    diff = compute_ssim_error(img_tensor, recon)

    raw_mean = float(diff.mean())
    scalar_score = raw_mean

    dmin, dmax = diff.min(), diff.max()
    if dmax > dmin:
        diff_norm = (diff - dmin) / (dmax - dmin)
    else:
        diff_norm = diff

    heatmap = torch.from_numpy(diff_norm.astype("float32"))
    return scalar_score, heatmap


def visualize_example(model, device, img_path, img_size=256):
    transform = build_transform(img_size)
    img = load_image(img_path, transform, img_size)

    score, heatmap = compute_ssim_score_and_map(model, device, img)

    img_np = img.permute(1, 2, 0).cpu().numpy()
    heat_np = heatmap.numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.axis("off")
    plt.imshow(img_np)

    plt.subplot(1, 3, 2)
    plt.title("SSIM error map")
    plt.axis("off")
    plt.imshow(heat_np, cmap="inferno")

    plt.subplot(1, 3, 3)
    plt.title(f"Score: {score:.4f}")
    plt.axis("off")
    plt.imshow(heat_np, cmap="inferno")

    plt.tight_layout()
    plt.show()


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root    = os.path.join(PROJECT_ROOT, "data", "mvtec")
    models_dir   = os.path.join(PROJECT_ROOT, "models")

    ckpt_path = os.path.join(models_dir, "multiscale_ae_best_enhanced.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Data root:", data_root)
    print("Loading model from:", ckpt_path)

    model = MultiScaleAutoencoder(in_channels=3, base_channels=32).to(device)

    # ⭐ FIXED: correctly handle checkpoint dict
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)

    test_items = get_test_image_paths(data_root, CATEGORIES)
    print(f"Found {len(test_items)} test images.")
    if len(test_items) == 0:
        raise RuntimeError("No test images found.")

    transform = build_transform(img_size=256)

    scores = []
    labels = []

    for img_path, label, cat, defect_type in test_items:
        img = load_image(img_path, transform, img_size=256)
        score, _ = compute_ssim_score_and_map(model, device, img)
        scores.append(score)
        labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    if (labels == 0).any() and (labels == 1).any():
        auroc = roc_auc_score(labels, scores)
        print(f"AUROC (SSIM error as anomaly score): {auroc:.4f}")
    else:
        print("Not enough class variety to compute AUROC.")

    good_scores = scores[labels == 0] if (labels == 0).any() else np.array([])
    anom_scores = scores[labels == 1] if (labels == 1).any() else np.array([])

    good_mean = good_scores.mean() if good_scores.size > 0 else float("nan")
    anom_mean = anom_scores.mean() if anom_scores.size > 0 else float("nan")

    print(f"Mean SSIM error - good: {good_mean:.4f}, anomaly: {anom_mean:.4f}")

    print("Showing some example SSIM heatmaps...")
    random.shuffle(test_items)
    for img_path, label, cat, defect_type in test_items[:6]:
        print(f"Category: {cat}, type: {defect_type}, label: {label}, path: {os.path.basename(img_path)}")
        visualize_example(model, device, img_path, img_size=256)


if __name__ == "__main__":
    main()
