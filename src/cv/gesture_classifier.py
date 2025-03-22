import torch
import torch.nn as nn
from torchvision import models
import mediapipe as mp
import cv2
import numpy as np

# Define gestures (same as Notebook)
GESTURES = ["hello", "help", "good", "stop", "please"]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
def load_model(model_path="data/models/trained/mobilenet_gestures.pth"):
    model = models.mobilenet_v2(pretrained=False)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(GESTURES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Predict gesture from webcam frame
def predict_gesture(model, frame, mp_hands):
    results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        landmarks = np.array([[lm.x, lm.y, lm.z] for hand in results.multi_hand_landmarks for lm in hand.landmark])
        if landmarks.shape[0] == 42:
            inputs = torch.tensor(landmarks.flatten(), dtype=torch.float32).to(device)
            inputs = inputs.view(-1, 1, 14, 9)
            with torch.no_grad():
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1).item()
                return GESTURES[pred]
    return None
