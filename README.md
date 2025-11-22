# Anomaly_Detection
# Unsupervised Multi-Scale Anomaly Detection in Industrial Manufacturing

## Project Summary
This project aims to develop an AI system that can automatically detect and localize defects in industrial products using only defect-free images for training. The system leverages a **multi-scale U-Net autoencoder**, which learns to reconstruct normal patterns but struggles on anomalous regions, enabling defect detection via reconstruction error analysis.

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

---

## Team Members






---

## Dataset
We are using the **MVTec Anomaly Detection Dataset (MVTec AD)**:

- Official Source: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

**Note:** The dataset is stored in **Google Drive** for this project due to its large size. Team members can access it by mounting Drive in Google Colab.

---

## Initial Work
For the first stage of the project, we have implemented the following:

1. **Data Pipeline**
   - Dataset loader for all categories.
   - Data transformations: resizing, tensor conversion.
   - Google Drive integration for dataset storage.

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
git clone https://github.com/Veditha04/Anomaly_Detection.git
cd Anomaly_Detection
Create and activate a virtual environment:

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
⚠️ Do not commit the .venv folder — it is ignored via .gitignore.

Usage
Training
bash
Copy code
python src/train.py --dataset data/mvtec_anomaly_detection --category bottle --epochs 50 --batch_size 16
Inference
bash
Copy code
python src/inference.py --dataset data/mvtec_anomaly_detection --category bottle --checkpoint checkpoints/bottle.pth
Generates reconstructed images and heatmaps highlighting anomalies.

Outputs are saved in the results/ folder.

Jupyter Notebook Example
bash
Copy code
jupyter notebook notebooks/inference_demo.ipynb
Interactive visualization of predictions.

Results
Example output: results/inference_results.png

Heatmaps highlight defective regions based on reconstruction errors.

Metrics can be calculated at image and pixel levels.

Contributing
Fork the repository.

Create a new branch: git checkout -b feature-name.

Commit your changes: git commit -m "Add feature".

Push to your branch: git push origin feature-name.

Open a Pull Request.

License
This project is licensed under the MIT License — see LICENSE for details.
