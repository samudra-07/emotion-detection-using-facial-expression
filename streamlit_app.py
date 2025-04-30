import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
from PIL import Image

# Set page config as the first command
st.set_page_config(page_title="Emotion Detection", page_icon="😃")

# Load model
json_file = open("emotionRecognition.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotionRecognition.keras")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to detect emotions
def detect_emotions(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        img = extract_features(face_img)
        pred = model.predict(img)
        emotion = pred.argmax()
        results.append(((x, y, w, h), emotion))
    return results

# Streamlit UI
st.title("Real-Time Emotion Detection")
st.sidebar.markdown("**Contact:** your.email@example.com")

# Upload image or use webcam
st.write("Upload an image or use your webcam to detect emotions in real-time:")

image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    results = detect_emotions(image)
    for ((x, y, w, h), emotion) in results:
        st.text(f"Emotion: {emotion} at ({x}, {y}, {w}, {h})")
else:
    st.write("Webcam feed:")
    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        # Display the frame in Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

        # Call detect_emotions function and display the detected emotions
        results = detect_emotions(frame)
        for ((x, y, w, h), emotion) in results:
            st.text(f"Emotion: {emotion} at ({x}, {y}, {w}, {h})")

# Close the webcam
webcam.release()
