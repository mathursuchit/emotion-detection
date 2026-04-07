# Emotion Detection from Scratch

Deep learning pipeline to classify facial expressions into 7 emotions using the FER-2013 dataset.

**Live demo:** *(add Streamlit Cloud URL here once deployed)*

## Overview

Builds and compares three deep learning approaches:
1. **Custom CNN** — baseline model trained from scratch
2. **CNN + Image Augmentation** — improved generalization
3. **ResNet50 Transfer Learning** — best performance using ImageNet pre-trained weights

## Dataset

[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) — 35,000+ grayscale face images (48x48px), 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

## Results

| Model | Validation Accuracy |
|-------|-------------------|
| Custom CNN | ~55% |
| CNN + Augmentation | ~58% |
| ResNet50 (Transfer Learning) | ~65% |

## Run Locally

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb   # train models
streamlit run app.py               # launch app
```

## Features

- Real-time emotion detection via webcam (browser-based)
- Image upload support
- Confidence scores and probability breakdown per emotion
- Face bounding boxes with OpenCV Haar Cascade

## Tech Stack

Python · TensorFlow/Keras · ResNet50 · OpenCV · Streamlit · Jupyter

## Author

**Suchit Mathur** — [LinkedIn](https://www.linkedin.com/in/mathursuchit/) | [Email](mailto:suchitmathur96@gmail.com)
