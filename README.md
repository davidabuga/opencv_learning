# Face Recognition Using OpenCV

This project implements face detection and recognition using OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer. It detects faces using a Haar Cascade classifier and recognizes them using a pre-trained model.

## Features
- Face detection using Haar Cascade Classifier
- Face recognition using LBPH algorithm
- Training the model with a dataset of labeled images
- Predicting and labeling faces on new images
- Real-time recognition with webcam support (optional)

## Installation

### Prerequisites
Make sure you have Python installed. You can download it from [Python's official website](https://www.python.org/downloads/).

### Install Required Libraries
Run the following command to install the necessary dependencies:

```sh
pip install opencv-contrib-python numpy
```

## Dataset Preparation
1. Create a dataset folder, e.g., `DATASETT`, with subfolders named after each person.
2. Store multiple images of each person inside their respective subfolders.

Example:
```
DATASETT/
    ALEX_SAIBULU/
        img1.jpg
        img2.jpg
    DAVID/
        img1.jpg
        img2.jpg
```

## Training the Model
To train the model, run:
```sh
python train.py
```
This script:
- Reads images from `DATASETT`.
- Detects faces using the Haar cascade.
- Extracts features and labels.
- Trains and saves the LBPH face recognition model (`face_trained.yml`).

## Testing the Model
To test recognition on an image, run:
```sh
python test.py
```
This script:
- Loads `face_trained.yml`.
- Detects faces in an image.
- Recognizes and labels detected faces.

## Real-time Recognition (Optional)
To use a webcam for real-time face recognition, modify `test.py` to use:
```python
cap = cv.VideoCapture(0)
```
Run:
```sh
python webcam_test.py
```

## File Overview
- `train.py`: Trains the LBPH model on the dataset.
- `test.py`: Recognizes faces from an image.
- `webcam_test.py`: Runs real-time face recognition using a webcam.
- `face_trained.yml`: Saved trained model.
- `features.npy`, `labels.npy`: Extracted feature and label arrays.

## Troubleshooting
- If `cv2.face` is missing, install `opencv-contrib-python`:
  ```sh
  pip install opencv-contrib-python
  ```
- Ensure dataset images are clear and well-lit for better accuracy.
- If the model performs poorly, try increasing the dataset size.

## License
This project is licensed under the MIT License.

---
Developed by **DAVID ABUGA** 🚀

