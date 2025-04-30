import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    json_file = open("emotionRecognition.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("emotionRecognition.keras")
    return model

model = load_model()
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotions(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face_features = extract_features(face)
        pred = model.predict(face_features)
        emotion = labels[pred.argmax()]
        results.append(((x, y, w, h), emotion))
    return results

st.title("Emotion Detection Using Facial Expression")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Detect emotions
    results = detect_emotions(image)

    # Display results
    for ((x, y, w, h), emotion) in results:
        st.write(f"Emotion: {emotion} at location (x: {x}, y: {y}, w: {w}, h: {h})")
    st.success("Emotion detection completed!")

