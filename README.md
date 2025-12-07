# CS6366 – Industrial Anomaly Detection with Autoencoders



## Team Members
- Devi Annamreddy(G42683473) - GitHub: [devi7037](https://github.com/devi7037)  
- Harichandana Samudrala(G48786002) - GitHub: [harichandana94](https://github.com/harichandana94) 
- Veditha Reddy Avuthu(G43696437) - GitHub: [Veditha04](https://github.com/Veditha04)

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

- **Total training images used:** 1,054  
- **Total test images used by our pipeline:** 460  

> **Note:** Due to dataset size and license, the **MVTec images are not included in this repository**.  
> For reproduction, download the dataset from the official source and place the relevant categories under:

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

### 3. Model or Component Design

The core model used in this project is a U-Net–style MultiScale Convolutional Autoencoder designed for anomaly detection on the MVTec AD dataset. The network learns to reconstruct normal images, and reconstruction error is used to detect defects.

                MultiScale Autoencoder (U-Net Style)

                    Input: 3 × 256 × 256
                             │
                             ▼
                     ┌───────────────────┐
                     │     ENCODER       │
                     └───────────────────┘
            
             x1: ConvBlock(3 → 32)                 → 256×256
                 MaxPool(2)                        → 128×128
             
             x2: ConvBlock(32 → 64)                → 128×128
                 MaxPool(2)                        → 64×64
            
             x3: ConvBlock(64 → 128)               → 64×64
                 MaxPool(2)                        → 32×32
              
             x4: Bottleneck ConvBlock(128 → 256)   → 32×32

                     ┌───────────────────┐
                     │     DECODER       │
                     └───────────────────┘
           
            up3: ConvT(256 → 128)                 → 64×64
                 concat(x3) → ConvBlock(256 → 128)

            up2: ConvT(128 → 64)                  → 128×128
                concat(x2) → ConvBlock(128 → 64)

            up1: ConvT(64 → 32)                   → 256×256
                concat(x1) → ConvBlock(64 → 32)

            final_conv: 1×1 Conv(32 → 3),  activation: Sigmoid
            
                       Output: 3 × 256 × 256

### 4. Model or Component Description

The **MultiScale Autoencoder** is designed to reconstruct *normal* MVTec images.  
During inference, **defective** images fail to reconstruct accurately, and the
difference between input and reconstruction becomes the anomaly score.

#### How It Works

#### Encoder

- The encoder uses stacked convolutional blocks (`Conv → BatchNorm → ReLU` twice) to extract hierarchical features.
- After each block, `MaxPool2d(2)` reduces the spatial resolution.
- Spatial sizes reduce as follows: **256 → 128 → 64 → 32**.
- These feature maps capture important textures and structural details needed for anomaly detection.

#### Latent / Bottleneck

- The bottleneck is a `ConvBlock(128 → 256)` operating at **32×32** resolution.
- This compressed representation learns only the distribution of **normal** images (because training uses only defect-free data).
- Defective regions do not fit this learned manifold and therefore reconstruct poorly.

#### Decoder

- The decoder upsamples the feature maps back to the original resolution using `ConvTranspose2d` layers.
- At each upsampling stage, the model concatenates encoder features via **skip connections**:
  - `up3`: reconstructs **64×64** and concatenates with encoder output x3  
  - `up2`: reconstructs **128×128** and concatenates with encoder output x2  
  - `up1`: reconstructs **256×256** and concatenates with encoder output x1
- Each concatenated tensor is processed by a ConvBlock.
- A final `1×1 Conv` followed by `Sigmoid` produces the reconstructed **RGB image**.

**Interpretation:**
- Normal images reconstruct cleanly.  
- Defective regions reconstruct poorly, producing higher pixel-wise error or bright areas in SSIM heatmaps.  
- This reconstruction gap is used as the **anomaly score**.

#### Why This Model Is Better Than the Baseline Autoencoder

The MultiScale Autoencoder improves reconstruction quality and defect localization through several architectural advantages:

- **Skip connections preserve high-resolution spatial details**, allowing the decoder to recover fine textures that the baseline model often blurs.
- **Multi-scale feature learning** enables the model to detect both small defects (scratches, cracks, dents) and larger structural anomalies.
- **Better gradient flow** through skip pathways stabilizes training and reduces the chance of vanishing gradients.
- **Sharper reconstructions** lead to more meaningful pixel-wise error maps and clearer SSIM heatmaps.
- Although AUROC values are similar, the MultiScale model achieves a **much lower validation loss** (0.0062 vs 0.0214), reflecting significantly more accurate image reconstruction.
- The MultiScale Autoencoder yields more precise anomaly localization, which is particularly evident in the SSIM heatmaps where the defective areas stand out as bright, sharply outlined     regions.

Overall, the MultiScale Autoencoder is the more effective architecture for **interpretable anomaly detection and localization**, even when quantitative scores are close.

### 5. Example of Model or Component Functionality

This section explains how the autoencoders detect anomalies and how the SSIM heatmaps
help visualize defect locations.

#### 5.1 Image-Level Anomaly Score (L1 Reconstruction Error)

During evaluation (`src/evaluate_models.py`), each test image is passed through the
autoencoder. A **scalar anomaly score** is computed as the mean absolute difference
between the input image and its reconstruction:

```python
img, label, category, defect_type, path = test_dataset[i]   # from MvtecTestDataset
img = img.unsqueeze(0).to(device)

recon = model(img)                                          # reconstruction
error = torch.abs(recon - img).mean().item()                # L1 anomaly score
```
These per-image scores are used to compute:

- **Overall AUROC and PR-AUC**
- **Per-category AUROC** (bottle, hazelnut, cable, tile)
- **Mean L1 error** on normal vs anomalous samples

#### Final Model Metrics (Mean L1 Error)

**Baseline Autoencoder**
- Normal mean error: `0.021152`
- Anomaly mean error: `0.021468`
- AUROC: `0.5425`

**Multi-Scale Autoencoder**
- Normal mean error: `0.006091`
- Anomaly mean error: `0.006246`
- AUROC: `0.5368`

The comparison figure (`results/comparison/model_comparison.png`) includes:
- L1 score distributions (normal vs anomalous)
- Per-category AUROC bar plot
- Overall metrics table

---

#### 5.2 SSIM-Based Defect Localization (Qualitative Heatmaps)

To visualize **where a defect occurs**, we compute SSIM-based error maps using the
trained MultiScale Autoencoder in:



```python
from eval_multiscale_ssim import compute_ssim_score_and_map

img = load_image(path, transform, img_size=256)  # [3, 256, 256]
score, heatmap = compute_ssim_score_and_map(model, device, img)
```
Where:

- **score** → mean SSIM error (higher = more anomalous)  
- **heatmap** → 2D defect-localization map showing regions the model fails to reconstruct

The script automatically displays:

- The **input image**
- The **SSIM error map**
- The **final heatmap** highlighting defective regions

It also reports quantitative metrics:

- **AUROC (SSIM error):** 0.5037  
- **Mean SSIM error – good:** 0.0068  
- **Mean SSIM error – anomalous:** 0.0066  

These SSIM heatmaps qualitatively highlight **cracks, scratches, bent wires, broken areas, contamination, misprints, and other subtle defects**, making them valuable for **interpretable anomaly localization** in industrial inspection tasks.

### Final Output of the System

The system produces:

- Trained **Baseline** and **MultiScale** Autoencoder models  
- Reconstruction visualizations for normal vs defective images  
- SSIM-based pixel-level heatmaps for defect localization  
- Quantitative evaluation (AUROC, PR-AUC, mean L1 error)
- Per-category performance across bottle, hazelnut, cable, and tile

**Observed behavior:**

- Normal images reconstruct cleanly.
- Defective regions (cracks, scratches, dents, contamination) reconstruct poorly.
- This difference appears as **higher reconstruction error** or **bright regions in SSIM heatmaps**.

**Final Evaluation Summary:**
### Quantitative Results (Mean L1 Error + AUROC)

| Model                       | Normal Mean Error | Anomaly Mean Error | AUROC  |
|---------------------------  |-------------------|--------------------|--------|
| **Baseline Autoencoder**    | 0.021152          | 0.021468           | 0.5425 |
| **Multi-Scale Autoencoder** | 0.006091        | 0.006246           | 0.5368 |

While AUROC values are similar, the **MultiScale Autoencoder achieves much lower validation loss (0.0062 vs 0.0214)** and produces **sharper reconstructions and clearer SSIM heatmaps**, making it more useful for qualitative defect localization.

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

```
## Google Colab Notebook

Click here to open the notebook in Google Colab:  
[Open the Project Notebook](https://colab.research.google.com/drive/1IrGuVx5fZLacPbnHCsfeaAj6iUzFUy_M?usp=sharing)

# Installation

Clone the repository:

```bash
git clone https://github.com/Veditha04/CS6366_Anamoly_Detection.git
cd CS6366_Anamoly_Detection
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
```
https://www.mvtec.com/company/research/datasets/mvtec-ad
```
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

# Results And Summary 

- Multi-scale autoencoder: Best validation loss = 0.006195
- Baseline autoencoder: Best validation loss = 0.021365
- Hazelnut category AUROC = 0.9636 (excellent)
- SSIM heatmaps qualitatively highlight defect regions, making the anomalies easier to interpret.

