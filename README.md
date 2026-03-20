# Deepfake Detection

** Traditional Features → Modern Deep Learning**

A complete Colab-based project for detecting deepfakes using the **DFD (FaceForensics-style)** dataset.

### 📊 Results Summary

| Approach                  | Accuracy | Key Insight                          |
|---------------------------|----------|--------------------------------------|
| Traditional (LBP + HOG + Color + RF) | **70.0%** | Handcrafted features hit a hard ceiling |
| **EfficientNet-B0**       | **0.87 %** | Strong single-frame performance     |
| **CNN + LSTM** (Best)     | **0.8 %** | Temporal modeling wins              |

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
