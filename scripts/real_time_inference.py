import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = '../models/gesture_model.h5'
IMG_SIZE = 64
GESTURES = sorted(os.listdir('../dataset'))  # assumes gesture folders are the labels

model = load_model(MODEL_PATH)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label = GESTURES[np.argmax(pred)]
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
