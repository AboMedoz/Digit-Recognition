import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT, 'data', 'test')
MODEL_PATH = os.path.join(ROOT, 'models')

model = load_model(os.path.join(MODEL_PATH, 'digit_recognition.keras'))

digits = []

for img_str in os.listdir(DATA_PATH):
    img_path = os.path.join(DATA_PATH, img_str)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = 255 - img

    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            roi = thresh[y: y + h, x: x + w]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype('float32') / 255.0
            roi = roi.reshape(-1, 28, 28, 1)

            prediction = model.predict(roi, verbose=0)
            digit = np.argmax(prediction)
            print(f"Number: {digit}")

