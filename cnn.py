from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import keras as kk
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

from time import time
from keras.models import Sequential, load_model

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Parametros: ", FRmodel.count_params())

def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)

    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    basic_loss = pos_dist - neg_dist + alpha

    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    print("loss = " + str(loss.eval()))


print("COMENZNDO A CARGAR EL MODELO ....")
start_time = time()
#FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

FRmodel = load_model('model.h5', custom_objects={'triplet_loss': triplet_loss})
#load_weights_from_FaceNet(FRmodel)
#FRmodel.save('model.h5')
#print("CREADO CON EXITO")

elapsed_time =time() - start_time
print("tiempo: ", elapsed_time)

database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["reddy"] = img_to_encoding("images/reddyCam.jpg", FRmodel)

#del

def verify(image_path, identity, database, model):
    encoding =img_to_encoding(image_path, model)
    dist =np.linalg.norm(encoding - database[identity])
    if dist < 0.7:
        print("Its " + str(identity) + " welcome! with: dist: "+ str(dist))
    else:
        print("its not " + str(identity) + ", go away with: " +str(dist))
    return dist

def who_is_it(image_path, database, model):
    encoding =img_to_encoding(image_path, model)
    min_dist = 100
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding-database[name])
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print ("Not recognized")
    else:
        print("its " + str(identity)+ ", the distance : "+  str(min_dist))
    return min_dist, identity

print("who is it..........")
start_time = time()
print("who is:...........................")
print("debe ser younes")
who_is_it("images/younes.jpg", database, FRmodel)
print("debe ser reddy")
who_is_it("images/reddyCam.jpg", database, FRmodel)
elapsed_time =time() - start_time
print("tiempo: ", elapsed_time)

start_time = time()
print("verify")
verify("images/kian.jpg", "kian", database, FRmodel)
elapsed_time =time() - start_time
print("tiempo verificar dude: ", elapsed_time)
