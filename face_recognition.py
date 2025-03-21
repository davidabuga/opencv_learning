import numpy as np
import cv2 as cv

haar_cascade= cv.CascadeClassifier(r"C:\Users\DAVID ABUGA\Downloads\haar_face.xml")

people = ['ALEX_SAIBULU', 'DAVID', 'ELKANAH', 'GICHURU', 'VICTOR']

#features=np.load('features.npy')
#labels=np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r"C:\Users\DAVID ABUGA\Desktop\DATASETT\DAVID\20230802_223719.jpg")
resized= cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)

gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 2)

for (x, y, w, h) in faces_rect:
    faces_roi=gray[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label={people[label]} with a confidence of {confidence}')
#How to write a text on an image as well. Here you have:

    cv.putText(resized, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(resized, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected face', resized)

cv.waitKey(0)