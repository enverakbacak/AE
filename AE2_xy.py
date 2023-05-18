#coding=utf-8
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input
import tensorflow.keras.optimizers
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense, Dropout, Flatten, UpSampling2D, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, InceptionV3, ResNet152V2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import math
from config import Config
from iGenerator import generator
from csv import reader
import numpy as np
import pandas as pd



from matplotlib import pyplot
from keras.datasets import cifar10

(trainX, trainy), (testX, testy) = cifar10.load_data()
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX  /= 255
testX /= 255

from tensorflow.keras.utils import to_categorical
trainY = to_categorical(trainy)
testY = to_categorical(testy)


base_model           = ResNet50(weights='imagenet', include_top=False, input_shape=(Config.resize, Config.resize, 3))
base_model.trainable = True


encoder_inputs       = tensorflow.keras.Input(shape = (Config.resize, Config.resize,3))

x         = base_model.output
x         = Flatten()(x)
x         = Dense(Config.hash_bits, name = "hashLayer")(x)
x         = Dense(16)(x)

x               = Reshape((4,4,1))(x) 
x               = Conv2D(1, (3, 3), activation ='relu', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)
x               = Conv2D(8, (3, 3), activation ='relu', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)
x               = Conv2D(16, (3, 3), activation ='relu', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)
y               = Conv2D(3, (3, 3), activation ="sigmoid", padding ='same')(x)

model           = Model(base_model.input, y)
model.summary()

sgd = tf.keras.optimizers.legacy.SGD(
        learning_rate = 0.001,
        decay = 1e-6,
        momentum=0.9,
        nesterov=True)


model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
              optimizer=sgd)

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        verbose=1,
        mode='auto')

rlrp = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=5,
        min_delta=1E-7)


model.fit(
    x=trainX,
    y=trainX,
    epochs=Config.num_epochs,
    batch_size=Config.batch_size,
    shuffle=True,
    callbacks = [early_stop, rlrp],
)



model_json = model.to_json()
with open("models/AE2_XY_16.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/AE2_XY_16.h5")  


