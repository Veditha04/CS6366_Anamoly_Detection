# CS6366 – Industrial Anomaly Detection with Autoencoders

# Project: Unsupervised Multi-Scale Anomaly Detection in Industrial Manufacturing

## Team Members
- Devi Annamreddy(G42683473) - GitHub: [devi7037](https://github.com/devi7037)  
- Harichandana Samudrala(G48786002) - GitHub: [harichandana94](https://github.com/harichandana94) 
- Veditha Reddy Avuthu(G436964371) - GitHub: [Veditha04](https://github.com/Veditha04)

## Project Summary

This project aims to automatically detect and localize defects in industrial products using **only defect-free images for training**. We compare:

- a **baseline convolutional autoencoder**, and  
- a **multi-scale U-Net–style autoencoder with skip connections**

on multiple categories from the **MVTec Anomaly Detection Dataset (MVTec AD)**.

The models learn to reconstruct normal patterns. At test time, regions that do not match the learned patterns reconstruct poorly, and these reconstruction differences (L1 error and SSIM-based maps) are used to **identify and localize defects**.

### Key Objectives

- Train on **defect-free industrial images only** (unsupervised anomaly detection).
- Detect **subtle, localized, and structural defects** across multiple categories.
- Evaluate performance at **image-level and per-category** using standard metrics.
- Provide **interpretable heatmaps** highlighting anomalous regions via SSIM.

---

## Dataset

We use a subset of the **MVTec AD** dataset:

- Official source: <https://www.mvtec.com/company/research/datasets/mvtec-ad>

### Categories Used

| Category | Train (good) | Test (good / defective) | Typical Defects                    | Challenge                    |
|----------|--------------|-------------------------|------------------------------------|------------------------------|
| bottle   | 209          | 20 / 63                 | Contamination, scratches           | Transparent background       |
| hazelnut | 391          | 40 / 70                 | Cracks, cuts, print defects        | Natural texture variation    |
| cable    | 224          | 58 / 90                 | Damage, bends, local deformations  | Thin and elongated structure |
| tile     | 230          | 57 / 100                | Chips, cracks, surface defects     | Texture-based anomalies      |

- **Total training images:** 1,054  
- **Total test images:** 460 (normal + anomalous)

> **Note:** Due to dataset size and license, the **MVTec images are not included in this repository**.  
> For reproduction, download the dataset from the official source and place the relevant categories under:
>
> `data/mvtec/<category>/train` and `data/mvtec/<category>/test`

---

## Model Architectures

### 1. Baseline Autoencoder

- Symmetric **encoder–decoder** with convolutional blocks and max-pooling / upsampling.
- No skip connections.
- Output activation: `Sigmoid` in [0, 1].
- **Parameters:** ~1.73M  
- **Best validation loss:** **0.021365** (L1)

Implemented in: `src/models.py` (`BaselineAutoencoder`)

### 2. Multi-Scale Autoencoder (U-Net Style)

- U-Net–style autoencoder with:
  - Multi-scale feature extraction
  - Skip connections from encoder to decoder
  - Transposed convolutions for upsampling
- Designed to better preserve spatial details and small defects.
- **Parameters:** ~1.93M  
- **Best validation loss:** **0.006195** (L1)

Implemented in: `src/models.py` (`MultiScaleAutoencoder`)

---

## Training & Evaluation

### Training Setup

- Framework: **PyTorch**
- Loss: **L1 loss** between input and reconstruction
- Optimizer: **Adam**
- Regularization: weight decay
- Learning rate scheduling: `ReduceLROnPlateau`
- Early stopping based on validation loss
- Gradient clipping (for multi-scale model)
- Mixed precision (GradScaler) for faster training on GPU

Training scripts:

- Baseline: `src/train_baseline_enhanced.py`
- Multi-scale: `src/train_multiscale_enhanced.py`

### Evaluation Metrics

Implemented in `src/evaluate_models.py`:

- **Image-level anomaly scores**:
  - Mean **L1 reconstruction error**
  - Top-1% L1 error (focus on high-error pixels)
- **Metrics**:
  - AUROC (Area Under ROC)
  - PR-AUC (Precision–Recall AUC)
  - Mean error on normal vs anomalous samples
- **Per-category AUROC** for all four categories.

### SSIM-Based Heatmaps

Implemented in `src/eval_multiscale_ssim.py`:

- Use **SSIM** between original and reconstructed image.
- Convert SSIM map to a **“(1 – SSIM)” error map**.
- Apply Gaussian smoothing and visualize as a heatmap.
- Provides qualitative defect localization examples for multiple categories.

---

## Results

### Overall Performance (Mean L1 Error)

| Model       | AUROC  | PR-AUC | Normal Mean Error | Anomaly Mean Error |
|------------|--------|--------|-------------------|--------------------|
| Baseline   | 0.5425 | 0.7264 | 0.021152          | 0.021468           |
| Multi-scale| 0.5368 | 0.6690 | 0.006091          | 0.006246           |

### Per-Category AUROC (Mean L1)

| Category | Baseline | Multi-scale |
|----------|----------|-------------|
| bottle   | 0.4587   | 0.5159      |
| hazelnut | 0.7200   | 0.9636      |
| cable    | 0.4996   | 0.4856      |
| tile     | 0.6595   | 0.7385      |

### Key Findings

- The **multi-scale model** achieves substantially **lower reconstruction error** (≈0.006 vs 0.021).
- On **hazelnut**, the multi-scale model reaches **96.36% AUROC**, indicating very strong anomaly separation.
- On **tile** (texture-based anomalies), multi-scale also outperforms baseline (73.85% vs 65.95% AUROC).
- Baseline slightly edges multi-scale on **cable**, suggesting room to explore architecture tuning or category-specific thresholds.
- SSIM-based heatmaps qualitatively highlight defective regions, making the model behavior more interpretable.

---

## Project Structure

```text
CS6366_Anamoly_Detection/
├── src/                        # Source code
│   ├── dataset.py              # Data loading and preprocessing (MvtecTrain/TestDataset, dataloaders)
│   ├── models.py               # Baseline + Multi-scale autoencoder architectures
│   ├── train_baseline_enhanced.py    # Training loop for baseline AE
│   ├── train_multiscale_enhanced.py  # Training loop for multi-scale AE
│   ├── evaluate_models.py      # L1-based evaluation, metrics, comparison plots
│   ├── eval_multiscale_ssim.py # SSIM-based scoring and heatmaps
│   └── run_all.py              # End-to-end pipeline (train both + evaluate)
│
├── models/                     # Trained model checkpoints (.pth)
│   ├── baseline_ae_best_enhanced.pth
│   └── multiscale_ae_best_enhanced.pth
│
├── results/
│   ├── baseline/               # Baseline training curves & loss values
│   ├── multiscale/             # Multi-scale training curves & loss values
│   └── comparison/             # Model comparison plot & metrics (model_comparison.png, evaluation_results.npz)
│
├── data/                       # MVTec dataset (not tracked in Git)
│   └── mvtec/                  # Place downloaded categories here
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)



## Dataset
We are using the **MVTec Anomaly Detection Dataset (MVTec AD)**:

- Official Source: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

**Note:** Due to the large size of the dataset, each team member has stored the MVTec AD dataset locally on their own machine while working on it.



## Initial Work
For the first stage of the project, we have implemented the following:

1. **Data Pipeline**
   - Dataset loader for all categories.
   - Data transformations: resizing, tensor conversion.
   - Dataset downloaded from the official source and stored locally in the repository (data/mvtec_anomaly_detection).

2. **Model Implementation**
   - Multi-Scale U-Net Autoencoder skeleton.
   - Encoder-decoder with skip connections.

3. **Training Setup**
   - Placeholder training loop implemented.
   - Loss function structure defined (MSE + SSIM + optional perceptual loss).
   - Model checkpoint saving/loading system.

4. **Evaluation**
   - Code ready for reconstruction visualization.
   - Template for computing image-level and pixel-level metrics.

```
# Installation

Clone the repository:

```
git clone https://github.com/Veditha04/Anomaly_Detection.git
cd Anomaly_Detection
```
Create and activate a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```
# Run the Project

1. Prepare the Dataset

Download the MVTec AD dataset (categories: bottle, hazelnut, cable, tile) from the official website:

https://www.mvtec.com/company/research/datasets/mvtec-ad

2. Run the Entire Pipeline (Train + Evaluate)

This will train both models and run evaluation automatically.

From the project root:
```
python -m src.run_all --all
```

This command will
- Train baseline autoencoder
- Train multi-scale autoencoder
- Run full evaluation
- Generate all metrics and plots

3. Run Individual Components 
- Train only the baseline model:
```
python src/train_baseline_enhanced.py
```

- Train only the multi-scale U-Net autoencoder:
```
python src/train_multiscale_enhanced.py
```

- Evaluate both models (L1 scores + AUROC):
```
python src/evaluate_models.py
```

Outputs go to:
```
results/comparison/model_comparison.png
results/comparison/evaluation_results.npz
```

- Generate SSIM Heatmaps (for defect localization):
```
python src/eval_multiscale_ssim.py
```

4. Output Locations

After running the pipeline, results will be generated in:
```
results/baseline/                 # Baseline training plots
results/multiscale/               # Multiscale training plots
results/comparison/model_comparison.png
results/comparison/evaluation_results.npz
```
Models saved in:
```
models/baseline_ae_best_enhanced.pth
models/multiscale_ae_best_enhanced.pth
```

# Results Summary (Short)

- Multi-scale autoencoder: Best validation loss = 0.006195
- Baseline autoencoder: Best validation loss = 0.021365
- Hazelnut category AUROC = 0.9636 (excellent)
- SSIM heatmaps provide clear defect localization
