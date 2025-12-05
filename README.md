# Unsupervised Multi-Scale Anomaly Detection in Industrial Manufacturing

## Team Members
- Devi Annamreddy(G42683473) - Github: devi7037  
- Harichandana Samudrala(G48786002) - Github: harichandana94  
- Veditha Reddy Avuthu(G436964371) - Github: Veditha04
## Project Summary
This project aims to automatically detect and localize defects in industrial products using only defect-free images for training. The approach is based on a multi-scale U-Net autoencoder, which learns to reconstruct normal patterns. During testing, regions that do not match the learned patterns reconstruct poorly, and these reconstruction differences are used to identify and localize defects.

**Key Objectives:**
- Train on defect-free industrial images only.
- Detect subtle, localized, and structural defects in multiple categories.
- Evaluate performance at both image and pixel levels.
- Provide interpretable heatmaps for defect localization.

**Defect Categories / Datasets:**
| Category  | Training Images | Test Images (Good/Defective) | Defect Types | Challenge |
|-----------|----------------|-----------------------------|--------------|-----------|
| Bottle    | 209            | 20 / 63                     | Contamination, scratches | Transparency |
| Capsule   | 219            | 42 / 71                     | Print defects, scratches, dents | Small localized defects |
| Hazelnut  | 391            | 40 / 70                     | Cracks, cuts, print defects | Natural texture variation |
| Screw     | 320            | 41 / 119                    | Manipulated tips, scratches | Structural anomalies |


## Dataset
We are using the **MVTec Anomaly Detection Dataset (MVTec AD)**:

- Official Source: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

**Note:** Due to the large size of the dataset, each team member has stored the MVTec AD dataset locally on their own machine while working in VS Code.



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



## Repository Structure

```

Anomaly_Detection/
│
├── src/                        # Source code
│   ├── dataset.py               # Data loader and preprocessing
│   ├── model.py                 # Multi-Scale U-Net Autoencoder
│   ├── train.py                 # Training script
│   ├── inference.py             # Inference / evaluation
│   └── utils.py                 # Utility functions
│
├── results/                     # Generated outputs
│   └── inference_results.png
│
├── notebooks/                   # Jupyter notebooks
│   └── inference_demo.ipynb
│
├── data/                        # Local dataset folder
│   └── mvtec_anomaly_detection/
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files/folders
└── README.md                    # This file
```



Installation
Clone the repository:

bash

Copy code
```
git clone https://github.com/Veditha04/Anomaly_Detection.git
cd Anomaly_Detection
Create and activate a virtual environment:
```

bash

Copy code
```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
Install dependencies:
```

bash

Copy code
```
pip install --upgrade pip
pip install -r requirements.txt
Do not commit the .venv folder — it is ignored via .gitignore.
```




Contributing
Fork the repository.

Create a new branch: 
``` git checkout -b feature-name.```

Commit your changes: 
```git commit -m "Add feature".```

Push to your branch: 
```git push origin feature-name.```



License
This project is licensed under the MIT License — see LICENSE for details.



%%writefile "/content/drive/My Drive/CS6366_Anamoly_Detection/README.md"
# CS6366: Industrial Anomaly Detection with Autoencoders

## Team Members
1. **Member 1**: Dataset preprocessing, Baseline autoencoder architecture
2. **Member 2**: Multi-scale U-Net architecture, Training pipeline optimization  
3. **Member 3**: Evaluation metrics, Visualization, Project documentation

## Project Overview
Industrial anomaly detection system using two autoencoder architectures on the MVTec AD dataset. The project compares a baseline symmetric autoencoder with a multi-scale U-Net style autoencoder.

## Key Features
- **Two Model Architectures**: Baseline vs Multi-scale autoencoders
- **Comprehensive Evaluation**: L1 error, SSIM, AUROC, PRAUC metrics
- **Visualization**: Training curves, heatmaps, score distributions
- **Category-wise Analysis**: Per-category performance comparison
- **Complete Pipeline**: End-to-end training and evaluation

## Dataset
- **Source**: MVTec Anomaly Detection Dataset
- **Categories Used**: bottle (209), hazelnut (391), cable (224), tile (230)
- **Total Training Images**: 1,054
- **Total Test Images**: 460

## Model Architectures

### 1. Baseline Autoencoder
- Symmetric encoder-decoder structure
- No skip connections
- 1.73 million parameters
- Best validation loss: 0.021365

### 2. Multi-scale Autoencoder  
- U-Net style with skip connections
- Multi-scale feature learning
- 1.93 million parameters
- Best validation loss: 0.006195

## Results

### Overall Performance
| Model | AUROC | PRAUC | Normal Error | Anomaly Error |
|-------|-------|-------|--------------|---------------|
| Baseline | 0.5425 | 0.7264 | 0.021152 | 0.021468 |
| Multi-scale | 0.5368 | 0.6690 | 0.006091 | 0.006246 |

### Per-Category AUROC
| Category | Baseline | Multi-scale |
|----------|----------|-------------|
| Bottle | 0.4587 | 0.5159 |
| Hazelnut | 0.7200 | **0.9636** |
| Cable | 0.4996 | 0.4856 |
| Tile | 0.6595 | **0.7385** |

### Key Findings
1. **Multi-scale model achieves lower reconstruction error** (0.006 vs 0.021)
2. **Hazelnut category shows excellent performance** with multi-scale (96.36% AUROC)
3. **Multi-scale excels on texture-based anomalies** (tile: 73.85% vs 65.95%)
4. **Baseline slightly better on cable category** (49.96% vs 48.56%)

## Project Structure
CS6366_Anamoly_Detection/
├── src/ # Source code
│ ├── dataset.py # Data loading and preprocessing
│ ├── models.py # Autoencoder architectures
│ ├── train_baseline_enhanced.py # Baseline training
│ ├── train_multiscale_enhanced.py # Multi-scale training
│ ├── evaluate_models.py # Comprehensive evaluation
│ ├── eval_multiscale_ssim.py # SSIM-based evaluation
│ ├── run_all.py # Complete pipeline
│ └── summary.py # Results summary
├── models/ # Trained model checkpoints
├── results/ # Training and evaluation results
│ ├── baseline/ # Baseline model results
│ ├── multiscale/ # Multi-scale model results
│ └── comparison/ # Comparative analysis
├── data/ # Dataset (not included in repo)
└── README.md # This file


## Installation
```bash
# Install dependencies
pip install torch torchvision numpy opencv-python scikit-learn scikit-image matplotlib seaborn tqdm Pillow pandas

Usage
Run Complete Pipeline
python
from run_all import main
main()
Train Specific Model
bash
# Train baseline model
python train_baseline_enhanced.py

# Train multi-scale model  
python train_multiscale_enhanced.py
Evaluate Models
bash
# Comprehensive evaluation
python evaluate_models.py

# SSIM-based evaluation
python eval_multiscale_ssim.py

# Generate summary report
python summary.py
Technical Details
Framework: PyTorch 2.0+

Training: Adam optimizer, L1 loss, learning rate scheduling

Validation: Early stopping, model checkpointing

Evaluation: AUROC, PRAUC, L1 error, SSIM

Hardware: NVIDIA GPU (Google Colab)

Files Generated
models/baseline_ae_best_enhanced.pth - Best baseline model

models/multiscale_ae_best_enhanced.pth - Best multi-scale model

results/comparison/model_comparison.png - Performance comparison plot

results/comparison/evaluation_results.npz - All evaluation metrics

project_summary.json - Complete project summary

Conclusion
The project successfully implements and compares two autoencoder architectures for industrial anomaly detection. The multi-scale autoencoder shows superior performance on texture-based anomalies while maintaining competitive performance overall. The modular design allows for easy extension to additional categories or architectures.
