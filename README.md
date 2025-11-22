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
