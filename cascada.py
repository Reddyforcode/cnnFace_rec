import numpy as np
import cv2 as cv
from cnn import *
import psycopg2

r = red()

class Person():
    def __init__(self, path_img, name):
        imgAux = cv.imread(path_img, 1)
        gray = imgAux[:, :, ::-1]
        face_cascade =cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces > 0):
            for (x, y, w, h) in faces:
                aux = imgAux[y:y+h, x:x+w]
        else:
            aux =imgAux
        self.img =aux
        #CARGAR UNA IMAGEN Y SACARLE LA CARA
        self.name = name
    def getImg():
        return self.img
    def getName():
        return self.name

def  getKnowPersonsFromDB():
    know_persons = []
    try:
        conn =psycopg2.connect("dbname=reconocimiento user=reddytintayaconde password=123456")
        cur = conn.cursor()
        sqlquery = "select nombre, img_src from know_users ORDER BY id;"
        cur.execute(sqlquery)
        row =cur.fetchone()
        while row is not None:
            print(row)
            know_persons.append(Persona("knowFaces/"+row[1], row[0]))
            print(know_persons[len(know_persons)-1].getNombre())    #print names
            row = cur.fetchone()
        cur.close()
        conn.close()
        return know_persons
    except:
        print("DB error")
#for testing
know_persons =getKnowPersonsFromDB()
reddy = cv2.imread("images/reddyCam.jpg", 1)
r.who_is_it(reddy)
print("\n\n.........\n\n")

face_cascade =cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalcatface.xml')
#eye_cascade  = cv.CascadeClassifier('cascades/haarcascade_eye.xml')
eye_cascade = cv.CascadeClassifier('cascades/haarcascade_eye_tree_eyeglasses.xml')

#img =cv.imread('cascades/glasses.jpg')
cap =cv.VideoCapture(0)

know_persons = []

while True:
    ret, frame = cap.read()
    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)    #mitad de la calidad
    img = small_frame
    gray = small_frame[:, :, ::-1]
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#roi = frame[ y:int((y+h)*0.7) , x:x+w ]﻿

    facesRec = []

    for (x, y, w, h) in faces:

        """zm = 0
        h = h + zm
        y = y - zm
        x = x - zm
        w = w + zm"""
        print("x: ", x,"y: ", y, " ", w," ",  h)
        #cv.imwrite('abel.jpg', img[y:y+h, x:x+w])
        #caras detectadas check
        try:
            aux = img[y:y+h, x:x+w]
            prob, ident = r.who_is_it(aux)
            #facesRec.append(aux)
        except:
            print("error :( )")

        cv.rectangle(img, (x, y), (x+w, y+h), (123, 123, 123), 2)
        cv.putText(img, ident, (x + 6, y+h - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 1)
        roi_gray  = gray[ y:int((y+h)*0.7) , x:x+w]
        roi_color = img[ y:int((y+h)*0.7) , x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    #print("cantidad de rostros reconocidos: ", len(faces))
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
