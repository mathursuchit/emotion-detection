import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")

st.title("😊 Real-Time Emotion Detector")
st.markdown("Detects facial emotions using a ResNet50 model trained on the FER-2013 dataset.")

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Happy':    '🟡',
    'Neutral':  '⚪',
    'Sad':      '🔵',
    'Angry':    '🔴',
    'Fear':     '🟠',
    'Disgust':  '🟢',
    'Surprise': '🟣',
}

HF_REPO_ID = "mathursuchit/emotion-detection"

@st.cache_resource
def load_model():
    with st.spinner("Loading model... (first run downloads ~100 MB)"):
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

model, face_cascade = load_model()

def detect_and_predict(image_array):
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

        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated, f'{emotion} ({confidence:.0%})',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': dict(zip(EMOTION_LABELS, preds.tolist()))
        })

    return annotated, results, len(faces)

tab1, tab2 = st.tabs(["📷 Camera", "🖼️ Upload Image"])

with tab1:
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image).convert('RGB')
        image_array = np.array(image)
        annotated, results, num_faces = detect_and_predict(image_array)

        st.image(annotated, caption="Detected Faces", use_container_width=True)

        if num_faces == 0:
            st.warning("No faces detected. Try better lighting or move closer.")
        else:
            for i, r in enumerate(results):
                emoji = EMOTION_COLORS.get(r['emotion'], '⚪')
                st.subheader(f"Face {i+1}: {emoji} {r['emotion']} ({r['confidence']:.0%} confidence)")
                probs = dict(sorted(r['probabilities'].items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(probs)

with tab2:
    uploaded = st.file_uploader("Upload a photo", type=['jpg', 'jpeg', 'png'])
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        image_array = np.array(image)
        annotated, results, num_faces = detect_and_predict(image_array)

        st.image(annotated, caption="Detected Faces", use_container_width=True)

        if num_faces == 0:
            st.warning("No faces detected. Try a clearer photo with good lighting.")
        else:
            for i, r in enumerate(results):
                emoji = EMOTION_COLORS.get(r['emotion'], '⚪')
                st.subheader(f"Face {i+1}: {emoji} {r['emotion']} ({r['confidence']:.0%} confidence)")
                probs = dict(sorted(r['probabilities'].items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(probs)

st.divider()
st.caption("Model: ResNet50 (Transfer Learning) | Dataset: FER-2013 (35,000+ images, 7 emotions) | Author: Suchit Mathur")
