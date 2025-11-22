# CS6366_Anamoly_Detection
This project presents an unsupervised deep learning approach for industrial anomaly detection using a multi-scale convolutional autoencoder architecture. The system is trained exclusively on defect-free samples from the MVTec AD benchmark dataset and demonstrates robust performance in detecting and localizing various types of manufacturing defects across four diverse industrial categories.

1. Introduction
1.1 Problem Context
Industrial quality inspection remains a critical challenge in manufacturing, with manual inspection being time-consuming, costly, and prone to human error. Traditional computer vision approaches require extensive labeled defect data, which is often scarce and expensive to obtain.

1.2 Project Objective
Develop an unsupervised anomaly detection system that:

Learns normal patterns from defect-free industrial images

Identifies anomalies through reconstruction error analysis

Generates interpretable defect localization heatmaps

Eliminates need for annotated defective samples during training
