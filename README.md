# Emotion Detection — Model Comparison

Side-by-side comparison of a custom ResNet50 trained on FER-2013 vs DeepFace (pre-trained on AffectNet+).

**Live demo:** https://mathursuchit-emotion-detection.streamlit.app/

## What it does

Upload a photo or use your webcam — the app runs both models simultaneously and shows you the predicted emotion, confidence score, and probability breakdown for each.

The point isn't just to detect emotions. It's to show what happens when you train on a small, noisy dataset (FER-2013) vs a large, curated one (AffectNet+). The gap in real-world performance is significant.

## Models

**My model (ResNet50)** — trained from scratch on FER-2013 (35K images, 48×48 grayscale, webcam-scraped). Weights hosted on HuggingFace, loaded at runtime.

**DeepFace** — pre-trained on AffectNet (450K+ images), RAF-DB, and others. Much more robust in real-world conditions.

## Why FER-2013 is hard

- Images are 48×48 grayscale and scraped from Google — noisy by nature
- Human annotators only agree ~65% of the time on the correct label
- Heavy class imbalance (Happy is overrepresented; Disgust has <800 samples)
- Real-world faces look very different from the training distribution

Despite these constraints, the custom model demonstrates the full pipeline: transfer learning, class imbalance handling, and deployment on free Kaggle GPU.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python · TensorFlow/Keras · ResNet50 · DeepFace · OpenCV · Streamlit · HuggingFace

## Author

Suchit Mathur — [LinkedIn](https://www.linkedin.com/in/mathursuchit/)
