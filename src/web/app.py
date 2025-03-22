"""
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, Response
import cv2
from src.cv.gesture_classifier import load_model, predict_gesture
import mediapipe as mp

app = Flask(__name__)

# Initialize Mediapipe
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)

# Load the model
model = load_model()

# Webcam feed generator
def generate_frames():
    cap = cv2.VideoCapture(0)  # Webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Predict gesture
        gesture = predict_gesture(model, frame, mp_hands)
        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
"""

import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, Response
import cv2
from src.cv.gesture_classifier import load_model, predict_gesture
import mediapipe as mp

app = Flask(__name__)

# Initialize Mediapipe
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)

# Load the model
model = load_model()

# Global variable to store the latest gesture
current_gesture = "None"

# Webcam feed generator
def generate_frames():
    global current_gesture
    cap = cv2.VideoCapture(0)  # Webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Predict gesture
        gesture = predict_gesture(model, frame, mp_hands)
        if gesture:
            current_gesture = gesture
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', gesture=current_gesture)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
