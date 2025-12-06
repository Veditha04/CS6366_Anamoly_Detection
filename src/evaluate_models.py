
import os
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd

from dataset import MvtecTestDataset
from models import BaselineAutoencoder, MultiScaleAutoencoder

CATEGORIES = ["bottle", "hazelnut", "cable", "tile"]


def compute_metrics(scores, labels):
    """Compute various evaluation metrics"""
    metrics = {}
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    good_scores = scores[labels == 0]
    anom_scores = scores[labels == 1]
    
    metrics["mean_good"] = float(np.mean(good_scores)) if len(good_scores) > 0 else np.nan
    metrics["std_good"]  = float(np.std(good_scores)) if len(good_scores) > 0 else np.nan
    metrics["mean_anom"] = float(np.mean(anom_scores)) if len(anom_scores) > 0 else np.nan
    metrics["std_anom"]  = float(np.std(anom_scores)) if len(anom_scores) > 0 else np.nan

    if len(np.unique(labels)) > 1:
        metrics["auroc"] = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics["prauc"] = auc(recall, precision)
    else:
        metrics["auroc"] = np.nan
        metrics["prauc"] = np.nan

    return metrics


def evaluate_model(model, test_dataset, device, error_type="l1"):
    """
    Evaluate a model on the test dataset.

    error_type:
      - "l1"      : mean absolute error over all pixels
      - "l1_topk" : mean of top-k% largest pixel errors (focus on anomalous regions)
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    category_scores = {cat: [] for cat in CATEGORIES}
    category_labels = {cat: [] for cat in CATEGORIES}
    
    print(f"Evaluating error_type = {error_type} ...")
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img, label, category, defect_type, _ = test_dataset[idx]
            img = img.unsqueeze(0).to(device)  # [1, C, H, W]
            
            recon = model(img)
            
            if error_type == "l1":
                error = torch.abs(recon - img).mean().item()
            elif error_type == "l1_topk":
                # Per-pixel L1, then take top k%
                per_pix = torch.abs(recon - img).mean(dim=1, keepdim=False)  # [1, H, W]
                flat = per_pix.view(1, -1)  # [1, H*W]
                num_pixels = flat.size(1)
                k = max(1, int(0.01 * num_pixels))  # top 1%
                topk_vals, _ = torch.topk(flat, k, dim=1)
                error = topk_vals.mean().item()
            else:
                raise ValueError(f"Unknown error_type: {error_type}")
            
            all_scores.append(error)
            all_labels.append(label)
            
            if category in category_scores:
                category_scores[category].append(error)
                category_labels[category].append(label)
    
    overall_metrics = compute_metrics(all_scores, all_labels)
    
    category_metrics = {}
    for cat in CATEGORIES:
        if len(category_scores[cat]) > 0:
            category_metrics[cat] = compute_metrics(category_scores[cat], category_labels[cat])
    
    return {
        "overall":  overall_metrics,
        "category": category_metrics,
        "scores":   all_scores,
        "labels":   all_labels,
    }


def plot_results(baseline_results, multiscale_results, save_dir):
    """Plot comparison results for the standard L1 score"""
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # 1–2: score distributions
    for i, (model_name, results) in enumerate([
        ("Baseline", baseline_results),
        ("MultiScale", multiscale_results),
    ]):
        scores = np.array(results["scores"])
        labels = np.array(results["labels"])
        good_scores = scores[labels == 0]
        anom_scores = scores[labels == 1]
        
        axes[i].hist(good_scores, bins=50, alpha=0.7, label="Normal", density=True)
        axes[i].hist(anom_scores, bins=50, alpha=0.7, label="Anomalous", density=True)
        axes[i].set_title(f"{model_name} - Score Distribution (L1)")
        axes[i].set_xlabel("Reconstruction Error")
        axes[i].set_ylabel("Density")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # 3: per-category AUROC
    categories = CATEGORIES
    baseline_aurocs = []
    multiscale_aurocs = []
    
    for cat in categories:
        if cat in baseline_results["category"]:
            baseline_aurocs.append(baseline_results["category"][cat].get("auroc", np.nan))
        else:
            baseline_aurocs.append(np.nan)
            
        if cat in multiscale_results["category"]:
            multiscale_aurocs.append(multiscale_results["category"][cat].get("auroc", np.nan))
        else:
            multiscale_aurocs.append(np.nan)
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[2].bar(x - width/2, baseline_aurocs, width, label="Baseline", alpha=0.8)
    axes[2].bar(x + width/2, multiscale_aurocs, width, label="MultiScale", alpha=0.8)
    axes[2].set_xlabel("Category")
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Per-Category AUROC Comparison (L1)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories, rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")
    
    # 4: overall metrics table (L1)
    metrics_df = pd.DataFrame({
        "Model": ["Baseline", "MultiScale"],
        "AUROC": [
            baseline_results["overall"].get("auroc", np.nan),
            multiscale_results["overall"].get("auroc", np.nan),
        ],
        "PRAUC": [
            baseline_results["overall"].get("prauc", np.nan),
            multiscale_results["overall"].get("prauc", np.nan),
        ],
        "Mean Good": [
            baseline_results["overall"].get("mean_good", np.nan),
            multiscale_results["overall"].get("mean_good", np.nan),
        ],
        "Mean Anom": [
            baseline_results["overall"].get("mean_anom", np.nan),
            multiscale_results["overall"].get("mean_anom", np.nan),
        ],
    })
    
    axes[3].axis("off")
    table = axes[3].table(
        cellText=metrics_df.round(4).values,
        colLabels=metrics_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[3].set_title("Overall Metrics Comparison (L1)")
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)

  
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_comparison_{timestamp}.png"
    out_path = os.path.join(save_dir, filename)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"New comparison plot saved to: {out_path}")


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root   = os.path.join(PROJECT_ROOT, "data", "mvtec")
    models_dir  = os.path.join(PROJECT_ROOT, "models")
    results_dir = os.path.join(PROJECT_ROOT, "results", "comparison")

    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = MvtecTestDataset(
        root_dir=data_root,
        categories=CATEGORIES,
        img_size=256,
    )
    print(f"Loaded {len(test_dataset)} test images")
    
    # Load models
    print("\nLoading models...")
    
    baseline_model = BaselineAutoencoder(in_channels=3, base_channels=32).to(device)
    baseline_ckpt = torch.load(
        os.path.join(models_dir, "baseline_ae_best_enhanced.pth"),
        map_location=device,
    )
    baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])
    print(f"Baseline model loaded (best val loss: {baseline_ckpt.get('best_val_loss', 'N/A'):.6f})")
    
    multiscale_model = MultiScaleAutoencoder(in_channels=3, base_channels=32).to(device)
    multiscale_ckpt = torch.load(
        os.path.join(models_dir, "multiscale_ae_best_enhanced.pth"),
        map_location=device,
    )
    multiscale_model.load_state_dict(multiscale_ckpt["model_state_dict"])
    print(f"Multi-scale model loaded (best val loss: {multiscale_ckpt.get('best_val_loss', 'N/A'):.6f})")
    
    # Evaluate models – standard L1
    print("\nEvaluating models with mean L1 error...")
    baseline_l1 = evaluate_model(baseline_model, test_dataset, device, error_type="l1")
    multiscale_l1 = evaluate_model(multiscale_model, test_dataset, device, error_type="l1")
    
    # Evaluate models – top-1% L1 (small tweak)
    print("\nEvaluating models with top-1% L1 error (focus on hotspots)...")
    baseline_topk = evaluate_model(baseline_model, test_dataset, device, error_type="l1_topk")
    multiscale_topk = evaluate_model(multiscale_model, test_dataset, device, error_type="l1_topk")
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS (MEAN L1)")
    print("="*60)
    print("\nBaseline Autoencoder (L1):")
    print(f"  AUROC: {baseline_l1['overall'].get('auroc', 'N/A'):.4f}")
    print(f"  PRAUC: {baseline_l1['overall'].get('prauc', 'N/A'):.4f}")
    print(f"  Normal mean error:  {baseline_l1['overall'].get('mean_good', 'N/A'):.6f}")
    print(f"  Anomaly mean error: {baseline_l1['overall'].get('mean_anom', 'N/A'):.6f}")
    
    print("\nMulti-Scale Autoencoder (L1):")
    print(f"  AUROC: {multiscale_l1['overall'].get('auroc', 'N/A'):.4f}")
    print(f"  PRAUC: {multiscale_l1['overall'].get('prauc', 'N/A'):.4f}")
    print(f"  Normal mean error:  {multiscale_l1['overall'].get('mean_good', 'N/A'):.6f}")
    print(f"  Anomaly mean error: {multiscale_l1['overall'].get('mean_anom', 'N/A'):.6f}")
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS (TOP-1% L1)")
    print("="*60)
    print("\nBaseline Autoencoder (Top-1% L1):")
    print(f"  AUROC: {baseline_topk['overall'].get('auroc', 'N/A'):.4f}")
    
    print("\nMulti-Scale Autoencoder (Top-1% L1):")
    print(f"  AUROC: {multiscale_topk['overall'].get('auroc', 'N/A'):.4f}")
    
    # Save L1 results (main ones) to npz
    results_file = os.path.join(results_dir, "evaluation_results.npz")
    np.savez(
        results_file,
        baseline_results=baseline_l1,
        multiscale_results=multiscale_l1,
    )
    print(f"\nL1 results saved to: {results_file}")
    
    # Plot L1 comparison
    plot_results(baseline_l1, multiscale_l1, results_dir)
    
    print("\n" + "="*60)
    print("PER-CATEGORY RESULTS (L1)")
    print("="*60)
    for cat in CATEGORIES:
        if cat in baseline_l1["category"] and cat in multiscale_l1["category"]:
            print(f"\n{cat.upper()}:")
            print(f"  Baseline AUROC:   {baseline_l1['category'][cat].get('auroc', 'N/A'):.4f}")
            print(f"  MultiScale AUROC: {multiscale_l1['category'][cat].get('auroc', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
