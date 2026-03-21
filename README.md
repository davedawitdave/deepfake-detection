# Deepfake Detection

## disclamer 
the notebook deeplearning_deepfake_detection is only spacial video frame authenticity detection which can not able to utilize the full potentical of LSTM so It is just basic ResNet18 CNN model. So ignore mesocnn inside CNN+LSTM for deep learning and you can see the full potential of feeding selected frames to LSTM model after reshaping the output of ResNet CNN.

 
** Traditional Features → Modern Deep Learning**

A complete Colab-based project for detecting deepfakes using the **DFD (FaceForensics-style)** dataset.

###  Results Summary

| Approach                  | Accuracy | Key Insight                          |
|---------------------------|----------|--------------------------------------|
| Traditional (LBP + HOG + Color + RF) feed SVM | **70.0%** | Handcrafted features hit a hard ceiling |
| **EfficientNet-B0**       | **0.87 %** | Strong single-frame performance     |
| **Resnet CNN** (only frame)     | **0.8 %** | spacial only              |
| **CNN + LSTM** (for image)     | **0.95 %** | Temporal modeling wins              |

###  Project Structure

- `Traditional_deepfake_detection.ipynb` → Handcrafted baseline (LBP + HOG + Color + Random Forest)
- `deeplearning_deepfake_detection.ipynb` → Full deep learning pipeline:
  - YOLO face detection + 10-frame extraction
  - Pre-extracted faces for speed
  - EfficientNet-B0 & CNN+LSTM with mixed precision
  - ROC-AUC, Confusion Matrix & ROC Curve

###  How to Run

1. Open either notebook in Google Colab
2. Run top-to-bottom (dataset auto-downloads from Kaggle)
3. For deep learning part: uses pre-extracted faces for stability on free GPU

**Requirements** (already included):
```bash
pip install -r requirements.txt# deepfake-detection

