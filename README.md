# Emotion Detection Using Facial Expression

## Overview
This project is an Emotion Detection system using facial expressions. It employs a Convolutional Neural Network (CNN) model to recognize emotions from facial images in real-time. The application uses OpenCV for video capture and face detection, combined with a pre-trained deep learning model for emotion classification.

## Features
- Real-time face detection using Haar Cascades.
- Emotion recognition with seven categories: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.
- Interactive visualization of detected faces and predicted emotions.
- Modularized code for easy adaptation and reuse.

## Prerequisites
- Python 3.7+
- Keras 2.4+ (with TensorFlow backend)
- OpenCV 4.5+
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/samudra-07/emotion-detection-using-facial-expression.git
   cd emotion-detection-using-facial-expression
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have a webcam connected and working for real-time video capture.

## Files and Directories
- **`emotionRecognition.json`**: The serialized architecture of the trained CNN model.
- **`emotionRecognition.keras`**: The pre-trained weights of the CNN model.
- **`haarcascade_frontalface_default.xml`**: Haar cascade file for face detection.
- **`main.py`**: Main script for running the application.

## How to Use
1. Run the `main.py` script:
   ```bash
   python main.py
   ```
2. The application will access your webcam, detect faces, and display the predicted emotion for each detected face in real-time.
3. Press `Esc` to exit the application.

## Code Highlights
### Face Detection and Preprocessing
```python
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

### Emotion Prediction
```python
# Prepare input for the model
image = cv2.resize(image, (48, 48))
img = extract_features(image)
pred = model.predict(img)
prediction_label = labels[pred.argmax()]
```

## Model Details
The model used is a CNN trained on the FER-2013 dataset, achieving accurate emotion recognition across seven categories.

## Future Improvements
- Enhance performance with a more sophisticated face detection algorithm (e.g., Dlib or MTCNN).
- Train a custom model for more nuanced emotion recognition.
- Add multi-language support for user interfaces.
- Deploy as a web application using Flask or FastAPI.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or raise an issue for any bugs or suggestions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- OpenCV for face detection and video processing.
- Keras and TensorFlow for deep learning.
- FER-2013 dataset for training the emotion recognition model.

## Contact
For any inquiries or issues, please contact [Samudra](https://github.com/samudra-07).

---

Happy Coding!

