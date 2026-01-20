import cv2
import numpy as np
import tensorflow as tf

# Load the saved model [cite: 19]
model = tf.keras.models.load_model('mnist_model.h5')

# 4. Real-Time Inference: Access Webcam [cite: 21]
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess live frames: Convert to grayscale and resize to 28x28 [cite: 22]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (28, 28)) 
    roi = roi.astype('float32') / 255
    roi = np.expand_dims(roi, axis=(0, -1)) # Prepare for model input

    # Perform prediction [cite: 22]
    prediction = model.predict(roi, verbose=0)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display predictions on screen [cite: 24]
    label = f"Digit: {class_id} ({confidence*100:.2f}%)"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Object Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
