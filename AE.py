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
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
import math
from config import Config
from iGenerator import generator
from csv import reader
import numpy as np
import pandas as pd

data=[]
labels=[]
# skip first line i.e. read header first and then iterate over each row od csv as a list
with open('../Datasets/Coco/labels/train_split_coco.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            data.append(row)
data=np.array(data)



#train_generator = generator(data, labels, batch_size=Config.batch_size)
train_generator = generator(data, batch_size=Config.batch_size)

encoder_inputs       = tensorflow.keras.Input(shape = (Config.resize, Config.resize,3))
base_model           = ResNet50(weights='imagenet', include_top=False, input_shape=(Config.resize, Config.resize, 3))
base_model.trainable = True

features  = base_model(encoder_inputs)
x         = Flatten()(features)
x         = Dense(1024)(x)
x         = Dense(512)(x)


'''
encoder_inputs  = tensorflow.keras.Input(shape = (Config.resize, Config.resize,3))
x               = Conv2D(16, (3, 3), activation ='relu', padding ='same')(encoder_inputs)
x               = MaxPooling2D((2, 2), padding ='same')(x)
x               = Conv2D(8, (3, 3), activation ='relu', padding ='same')(x)
x               = MaxPooling2D((2, 2), padding ='same')(x)
x               = Conv2D(8, (3, 3), activation ='relu', padding ='same')(x)
x               = MaxPooling2D((2, 2), padding ='same')(x)
x               = Flatten()(x)
x               = Dense(256)(x)
'''
x               = Dense(Config.hash_bits, activation='tanh', name = 'hashLayer')(x)
x               = Dense(256)(x)
x               = Reshape((16,16,1))(x)
x               = Conv2D(1, (3, 3), activation ='relu', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)
x               = Conv2D(8, (3, 3), activation ='relu', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)
x               = Conv2D(16, (3, 3), activation ='relu', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)
x               = Conv2D(3, (3, 3), activation ='sigmoid', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)

model     = Model(encoder_inputs, x)
print(model.summary())



def c_loss_1(y_true, y_pred):
    return  ( tf.keras.losses.binary_crossentropy(y_true, y_pred)) 

def c_loss_2(noise_1, noise_2):
    noise_1 = (hashLayer > 0 )
    #noise_1 = tf.cast(np.sign(features) , tf.float32 )
    noise_2 = hashLayer 
    return  tf.keras.metrics.Sum((tf.keras.losses.binary_crossentropy(noise_1, noise_2)))

def c_loss_3(noise_3):
    noise_3 =  (hashLayer > 0 )
    return   tf.keras.metrics.Sum(noise_3)


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = [c_loss_1, c_loss_2, c_loss_3], 
              loss_weights = [1, 0.1, 0.1],
              optimizer=sgd, 
              )

model.fit(train_generator,
                    steps_per_epoch  = math.ceil(len(data) // Config.batch_size),
                    #validation_data=val_generator,
                    #validation_steps = math.ceil(len(val_data) // Config.batch_size),
                    verbose=1,
                    epochs=Config.num_epochs)


model_json = model.to_json()
with open("models/model_AE_Coco_48.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/weights_AE_Coco_48.h5")  
