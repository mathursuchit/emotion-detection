import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from huggingface_hub import hf_hub_download
from deepface import DeepFace

st.set_page_config(page_title="Emotion Detector", page_icon=None, layout="wide")

st.title("Emotion Detection: Model Comparison")
st.markdown(
    "A side-by-side comparison of a **custom-trained ResNet50** (trained from scratch on FER-2013) "
    "vs **DeepFace** (pre-trained on AffectNet + multiple large-scale datasets)."
)

with st.expander("About this project", expanded=False):
    st.markdown("""
    ### What's being compared?

    | | My Model (ResNet50) | DeepFace |
    |---|---|---|
    | **Architecture** | ResNet50 + custom head | Ensemble of pre-trained CNNs |
    | **Training data** | FER-2013 (35K images) | AffectNet (450K+), RAF-DB, and more |
    | **Image quality** | 48×48 grayscale, webcam-scraped | High-res, diverse, curated |
    | **Label quality** | ~65% inter-rater agreement | Multi-annotator consensus |
    | **Compute used** | Single GPU, ~2 hours (Kaggle) | Large-scale industry training |

    ### Why does my model underperform in real-world conditions?
    FER-2013 is a well-known benchmark with significant limitations:
    - Images are small (48×48), grayscale, and scraped from Google Image Search
    - Labels are noisy — humans only agree ~65% of the time on the correct emotion
    - Heavy class imbalance (Happy is overrepresented; Disgust has <800 samples)
    - Real-world faces (different lighting, angles, ethnicities) differ from the training distribution

    Despite these constraints, the custom model demonstrates the full ML pipeline:
    transfer learning, fine-tuning, class imbalance handling, and deployment — with the compute available on a free Kaggle GPU.
    """)

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'happy':    '', 'Happy':    '',
    'neutral':  '', 'Neutral':  '',
    'sad':      '', 'Sad':      '',
    'angry':    '', 'Angry':    '',
    'fear':     '', 'Fear':     '',
    'disgust':  '', 'Disgust':  '',
    'surprise': '', 'Surprise': '',
}

HF_REPO_ID = "mathursuchit/emotion-detection"

@st.cache_resource
def load_custom_model():
    with st.spinner("Loading custom model... (first run downloads ~100 MB)"):
        weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename="emotion_detection.h5")
        base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(7, activation='softmax')
        ])
        model.build((None, 224, 224, 3))
        model.load_weights(weights_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return model, face_cascade

model, face_cascade = load_custom_model()

def predict_custom(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    annotated = image_array.copy()

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face.astype('float32') / 255.0
        face = np.stack([face] * 3, axis=-1)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)[0]
        emotion = EMOTION_LABELS[np.argmax(preds)]
        confidence = float(np.max(preds))

        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 165, 0), 2)
        cv2.putText(annotated, f'{emotion} ({confidence:.0%})',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': dict(zip(EMOTION_LABELS, preds.tolist()))
        })

    return annotated, results, len(faces)

def predict_deepface(image_array):
    annotated = image_array.copy()
    results = []

    try:
        faces = DeepFace.analyze(
            image_array,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='opencv',
            silent=True
        )
    except ValueError:
        return annotated, [], 0

    if isinstance(faces, dict):
        faces = [faces]

    for face in faces:
        region = face.get('region', {})
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        emotion = face['dominant_emotion']
        probs = face['emotion']
        confidence = probs[emotion] / 100.0

        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.putText(annotated, f'{emotion.capitalize()} ({confidence:.0%})',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        results.append({
            'emotion': emotion.capitalize(),
            'confidence': confidence,
            'probabilities': {k.capitalize(): v / 100.0 for k, v in probs.items()}
        })

    return annotated, results, len(results)

def show_results(col_custom, col_deepface, image_array):
    ann_custom, res_custom, n_custom = predict_custom(image_array)
    ann_deepface, res_deepface, n_deepface = predict_deepface(image_array)

    with col_custom:
        st.markdown("#### My Model (ResNet50 on FER-2013)")
        st.image(ann_custom, use_container_width=True)
        if n_custom == 0:
            st.warning("No faces detected.")
        else:
            for i, r in enumerate(res_custom):
                emoji = EMOTION_COLORS.get(r['emotion'], '⚪')
                st.subheader(f"Face {i+1}: {r['emotion']} ({r['confidence']:.0%})")
                probs = dict(sorted(r['probabilities'].items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(probs)

    with col_deepface:
        st.markdown("#### DeepFace (Pre-trained on AffectNet+)")
        st.image(ann_deepface, use_container_width=True)
        if n_deepface == 0:
            st.warning("No faces detected.")
        else:
            for i, r in enumerate(res_deepface):
                emoji = EMOTION_COLORS.get(r['emotion'], '⚪')
                st.subheader(f"Face {i+1}: {r['emotion']} ({r['confidence']:.0%})")
                probs = dict(sorted(r['probabilities'].items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(probs)

tab1, tab2 = st.tabs(["Camera", "Upload Image"])

with tab1:
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image).convert('RGB')
        image_array = np.array(image)
        col1, col2 = st.columns(2)
        show_results(col1, col2, image_array)

with tab2:
    uploaded = st.file_uploader("Upload a photo", type=['jpg', 'jpeg', 'png'])
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        image_array = np.array(image)
        col1, col2 = st.columns(2)
        show_results(col1, col2, image_array)

st.divider()
st.caption("Custom Model: ResNet50 fine-tuned on FER-2013 (35K images, 7 emotions) | Comparison: DeepFace pre-trained on AffectNet+ | Author: Suchit Mathur")
