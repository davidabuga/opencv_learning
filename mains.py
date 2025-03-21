import os
import cv2 as cv
import numpy as np

# List of people (subdirectories in the dataset folder)
people = ['ALEX_SAIBULU', 'DAVID', 'ELKANAH', 'GICHURU', 'VICTOR']
DIR = r"C:\Users\DAVID ABUGA\Desktop\DATASETT"

# Load Haar cascade for face detection
haar_cascade = cv.CascadeClassifier(r"C:\Users\DAVID ABUGA\Downloads\haar_face.xml")

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            # Read the image
            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Skipping corrupted image: {img_path}")
                continue  # Skip if the image is unreadable

            # Convert to grayscale
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect faces
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

            # Process each detected face
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

print("Training done...")

# Convert lists to numpy arrays
features = np.array(features, dtype="object")
labels = np.array(labels)

# Ensure OpenCV Face Recognizer is available
if hasattr(cv, "face"):
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
else:
    print("Error: cv2.face module is missing. Ensure you installed 'opencv-contrib-python'.")
    exit()

# Train the recognizer
face_recognizer.train(features, labels)

# Save the trained model
face_recognizer.save("face_trained.yml")

# Save feature and label data
np.save("features.npy", features)
np.save("labels.npy", labels)

print("Model training and saving completed successfully!")
