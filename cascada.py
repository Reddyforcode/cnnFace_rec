import numpy as np
import cv2 as cv

face_cascade =cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalcatface.xml')
#eye_cascade  = cv.CascadeClassifier('cascades/haarcascade_eye.xml')
eye_cascade = cv.CascadeClassifier('cascades/haarcascade_eye_tree_eyeglasses.xml')

#img =cv.imread('cascades/glasses.jpg')
cap =cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)    #mitad de la calidad
    img = small_frame
    gray = small_frame[:, :, ::-1]

    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#roi = frame[ y:int((y+h)*0.7) , x:x+w ]﻿

    faces = []

    for (x, y, w, h) in faces:
        print("x: ", x,"y: ", y, " ", w," ",  h)
        #cv.imwrite('abel.jpg', img[y:y+h, x:x+w])
        #caras detectadas check
        aux = img[y:y+h, x:x+w]
        faces.append(aux)

        cv.rectangle(img, (x, y), (x+w, y+h), (123, 123, 123), 2)
        roi_gray  = gray[ y:int((y+h)*0.7) , x:x+w]
        roi_color = img[ y:int((y+h)*0.7) , x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    print("cantidad de rostros reconocidos: ", len(faces))
    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()




"""
import numpy as np
import cv2 as cv

face_cascade =cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalcatface.xml')
#eye_cascade  = cv.CascadeClassifier('cascades/haarcascade_eye.xml')
eye_cascade = cv.CascadeClassifier('cascades/haarcascade_eye_tree_eyeglasses.xml')

#img =cv.imread('cascades/glasses.jpg')
cap =cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)    #mitad de la calidad
    img = small_frame
    gray = small_frame[:, :, ::-1]

    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#roi = frame[ y:int((y+h)*0.7) , x:x+w ]﻿

    for (x, y, w, h) in faces:
        print("x: ", x,"y: ", y, " ", w," ",  h)
        #cv.imwrite('abel.jpg', img[y:y+h, x:x+w])
        #caras detectadas check
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        roi_gray  = gray[ y:int((y+h)*0.7) , x:x+w]
        roi_color = img[ y:int((y+h)*0.7) , x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()

#hasta aqui funciona el detectar caras
"""
