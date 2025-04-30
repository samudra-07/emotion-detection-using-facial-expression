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

def detect_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

def main():
    st.set_page_config(page_title="Emotion Detection", page_icon="😃", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .title {
            color: #4CAF50;
            text-align: center;
            font-family: Arial, sans-serif;
        }
        .sidebar .sidebar-content {
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='title'>Emotion Detection Using Facial Expression</h1>", unsafe_allow_html=True)

    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose an input source:", ("Image Upload", "Camera Feed"))

    if option == "Image Upload":
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Processing...")

            frame = np.array(image)
            results = detect_emotions(frame)

            for ((x, y, w, h), emotion) in results:
                st.write(f"**Emotion:** {emotion} at location (x: {x}, y: {y}, w: {w}, h: {h})")
            st.success("Emotion detection completed!")

    elif option == "Camera Feed":
        st.subheader("Live Camera Feed")
        st.write("Please allow camera access to proceed.")
        run = st.checkbox("Start Camera")
        FRAME_WINDOW = st.image([])
        webcam = cv2.VideoCapture(0)

        while run:
            ret, frame = webcam.read()
            if not ret:
                st.error("Failed to capture image. Is your camera working?")
                break

            results = detect_emotions(frame)

            for ((x, y, w, h), emotion) in results:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        webcam.release()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Developed by:** Your Name")
    st.sidebar.markdown("**Contact:** your.email@example.com")

if __name__ == "__main__":
    main()
