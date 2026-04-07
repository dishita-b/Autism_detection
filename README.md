
# 🧠 Early Detection of Autism Spectrum Disorder via CNN with Self-Attention

An end to end deep learning pipeline for automated binary classification of Autism Spectrum Disorder in children aged 2 to 8 using facial images. The final model achieved 83.67% accuracy and 84.10% F1-score outperforming all classical ML baselines. Research accepted and published on IEEE Xplore.

IEEE Publication: https://ieeexplore.ieee.org/document/11210996

Institution: Indian Institute of Engineering Science and Technology Shibpur — Department of Information Technology

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Classical ML | Scikit-learn |
| Image Processing | OpenCV NumPy |
| Notebooks | Jupyter |
| Dataset | Kaggle — Autistic Children Facial Images (2940 images) |

---

## Project Structure

```
Autism_detection/
├── final-pro (1).ipynb                          # Main CNN + Self-Attention model
├── each-step-output-and-feature-extraction.ipynb  # Step-by-step feature extraction
├── face-mesh.ipynb                              # MediaPipe face landmark extraction
├── vit-autism.ipynb                             # Vision Transformer experiment
├── xgandrf.ipynb                                # XGBoost and Random Forest baseline
├── xgandrf.py                                   # Python script version of baseline
├── VGG16_features.csv                           # Extracted VGG16 feature vectors
├── image_labels.csv                             # Dataset labels
├── xgb_model_upto_best.json                     # Saved XGBoost model
└── Final Project Report.pdf                     # Full research report
```

---

## Dataset

2940 grayscale facial images of children collected from Kaggle. Autistic and non-autistic classes are evenly distributed. Age range 2 to 8 years. Male to female ratio in the autistic class is 3:1 and 1:1 in the non-autistic class.

---

## Preprocessing Pipeline

All images were standardized through three steps before model input:

Resize — all images resized to 128x128 pixels using OpenCV to ensure uniform input dimensions across the dataset

Grayscale Conversion — BGR images converted to single-channel grayscale using cv2.cvtColor reducing computational overhead while preserving structural and textural information

Array Transformation — images converted to normalized NumPy arrays with pixel values in range 0 to 1 for compatibility with TensorFlow tensor operations

---

## Model Architecture — Custom CNN with Self-Attention

The final architecture combines convolutional layers for local spatial feature extraction with a self-attention mechanism for global dependency modelling across the entire feature map

```
Input (128 x 128 x 1)
    ↓
2x Conv2D — 64 filters (3x3) + ReLU
    ↓
MaxPooling (2x2)
    ↓
Conv2D — 128 filters (3x3) + ReLU
    ↓
MaxPooling (2x2)
    ↓
Conv2D — 128 filters (3x3) + ReLU
    ↓
Self-Attention Layer
    ↓
Flatten
    ↓
Dense (256) + Dropout (0.2)
Dense (256) + Dropout (0.2)
Dense (128) + Dropout (0.2)
    ↓
Dense (1) — Sigmoid Output
```

### Self-Attention Mechanism

The self-attention layer operates on the final convolutional feature map of shape (batch size x height x width x filters). Three 1x1 convolutions generate Query K and Value matrices

```
Q = W_Q * X
K = W_K * X
V = W_V * X

Attention Scores    = Q * K^T
Scaled Scores       = Attention Scores / sqrt(d_k)
Attention Weights   = softmax(Scaled Scores)
Attention Output    = Attention Weights * V
```

This allows the model to dynamically weight every spatial location relative to all others — enabling it to focus on clinically relevant facial regions like gaze patterns and periorbital features rather than treating the entire face uniformly

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 128 |
| Epochs | 60 |
| Validation Strategy | 10-Fold Cross-Validation |

ModelCheckpoint callback was used to retain the best-performing weights per fold based on validation accuracy

---

## Baseline Models — Classical ML

HOG (Histogram of Oriented Gradients) features were extracted from all images and fed into classical classifiers as the baseline

HOG divides the image into cells computes gradient magnitude and orientation per pixel quantizes orientations into bins and normalizes across overlapping blocks to produce a single feature vector per image

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 56.14% | 57.0% | 59.0% | 58.0% |
| Decision Tree | 62.07% | 62.12% | 61.90% | 62.01% |
| Random Forest | 68.20% | 70.3% | 70.8% | 70.5% |
| CNN + Self-Attention | 83.67% | 81.93% | 86.39% | 84.10% |

---

## Results

The custom CNN with self-attention outperformed all baselines by a significant margin. The attention mechanism proved critical — enabling the model to identify long-range spatial dependencies across facial regions that HOG-based methods could not capture

---

## Publication

Title: Early Detection of Autism from Facial Images
Venue: IEEE Xplore
Link: https://ieeexplore.ieee.org/document/11210996
Institution: IIEST Shibpur — Department of Information Technology

Supervisor: Dr. Shyamalendu Kandar

---

## Author

Dishita Barman

GitHub: https://github.com/dishita-b
LinkedIn: https://www.linkedin.com/in/dishita-barman5/
