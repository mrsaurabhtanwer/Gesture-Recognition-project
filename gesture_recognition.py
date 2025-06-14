import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained gesture recognition model
model = load_model('gesture_model.h5')  # <-- Update with your model filename

# List of gesture labels as per your model's output order
labels = ['Palm', 'Fist', 'Thumbs Up', 'Thumbs Down', 'Okay', 'Peace']  # <-- Update as needed

# Start capturing video from webcam (change 0 if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for your model
    img = cv2.resize(frame, (64, 64))  # Change (64, 64) to your model's input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict gesture
    predictions = model.predict(img)
    predicted_label = labels[np.argmax(predictions)]

    # Show prediction on the frame
    cv2.putText(frame, predicted_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Gesture Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
