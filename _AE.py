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
from tensorflow.keras.layers import Dense, Dropout, Flatten, UpSampling2D
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
with open('../Datasets/Lamda/labels/data_for_pAE.csv', 'r') as read_obj:
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

image_input          = tensorflow.keras.Input(shape = (Config.resize, Config.resize,3))




base_model           = VGG16(weights='imagenet', include_top=False, input_shape=(Config.resize, Config.resize, 3))
base_model.trainable = True


#-------------------encoder---------------------------- 
#--------(pretrained & trainable if selected)----------

#    block1
x=base_model.get_layer('block1_conv1')(image_input)
x=base_model.get_layer('block1_conv2')(x)
x=base_model.get_layer('block1_pool')(x)

#    block2
x=base_model.get_layer('block2_conv1')(x)
x=base_model.get_layer('block2_conv2')(x)
x=base_model.get_layer('block2_pool')(x)

#    block3
x=base_model.get_layer('block3_conv1')(x)
x=base_model.get_layer('block3_conv2')(x)
x=base_model.get_layer('block3_conv3')(x)    
x=base_model.get_layer('block3_pool')(x)

#    block4
x=base_model.get_layer('block4_conv1')(x)
x=base_model.get_layer('block4_conv2')(x)
x=base_model.get_layer('block4_conv3')(x)    
x=base_model.get_layer('block4_pool')(x)

#    block5
x=base_model.get_layer('block5_conv1')(x)
x=base_model.get_layer('block5_conv2')(x)
x=base_model.get_layer('block5_conv3')(x)
     
    
#--------latent space (trainable) ------------
x=base_model.get_layer('block5_pool')(x)     
x = Conv2D(512, (3, 3), activation='relu', padding='same',name='latent')(x)
x = UpSampling2D((2,2))(x)  
    
#--------------decoder (trainable)----------- 
        
  # Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock5_conv3')(x)
x = UpSampling2D((2,2))(x)

  # Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock4_conv3')(x)
x = UpSampling2D((2,2))(x)

  # Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock3_conv3')(x)
x = UpSampling2D((2,2))(x)     
     
  # Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock2_conv3')(x)
x = UpSampling2D((2,2))(x)        
 
  # Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dblock1_conv1')(x)
x = Conv2D(3, (3, 3), activation='relu', padding='same', name='dblock1_conv3')(x)
#    x = UpSampling2D((2,2))(x) 

model     = Model(image_input, x)
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
              #metrics=['accuracy'],
              )
#model.compile(loss = c_loss_2,  optimizer=sgd, metrics=['accuracy'],)

model.fit(train_generator,
                    steps_per_epoch  = math.ceil(len(data) // Config.batch_size),
                    #validation_data=val_generator,
                    #validation_steps = math.ceil(len(val_data) // Config.batch_size),
                    verbose=1,
                    epochs=Config.num_epochs)


model_json = model.to_json()
with open("models/model_Lamda_pAE_512.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/weights_Lamda_pAE_512.h5")  
