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
from tensorflow.keras.utils import to_categorical



data=[]
labels=[]
# skip first line i.e. read header first and then iterate over each row od csv as a list
with open('../Datasets/retinamnist/retinamnist_labels/retinamnist_train.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            data.append(row)
data=np.array(data)


'''
train_labels = data[:,1]
train_labels_onehot = to_categorical(train_labels, dtype ="uint8")
print(train_labels_onehot.shape)
train_labels_onehot = np.array(train_labels_onehot)
labels = pd.DataFrame(train_labels_onehot)
labels.to_csv (r'onehot.csv', index = False, header=True)
'''


#train_generator = generator(data, labels, batch_size=Config.batch_size)
train_generator = generator(data, batch_size=Config.batch_size)

encoder_inputs       = tensorflow.keras.Input(shape = (Config.resize, Config.resize,3))
base_model           = ResNet50(weights='imagenet', include_top=False, input_shape=(Config.resize, Config.resize, 3))
base_model.trainable = True

features  = base_model(encoder_inputs)
x         = Flatten()(features)
x         = Dense(2048)(x)
x         = Dense(256)(x)



#encoder_inputs  = tensorflow.keras.Input(shape = (Config.resize, Config.resize,3))
#x               = Conv2D(16, (3, 3), activation ='relu', padding ='same')(encoder_inputs)
#x               = MaxPooling2D((2, 2), padding ='same')(x)
#x               = Conv2D(8, (3, 3), activation ='relu', padding ='same')(x)
#x               = MaxPooling2D((2, 2), padding ='same')(x)
#x               = Conv2D(8, (3, 3), activation ='relu', padding ='same')(x)
#x               = MaxPooling2D((2, 2), padding ='same')(x)
#x               = Flatten()(x)
#x               = Dense(256)(x)


x               = Dense(Config.hash_bits, activation='tanh', name = 'hashLayer')(x)
x               = Dense(256)(x) # for Coco , for retina, 16
x               = Reshape((16,16,1))(x) # for Coco , for retina it is 4
x               = Conv2D(1, (3, 3), activation ='relu', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)
x               = Conv2D(8, (3, 3), activation ='relu', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)
x               = Conv2D(16, (3, 3), activation ='relu', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)
x               = Conv2D(3, (3, 3), activation ='sigmoid', padding ='same')(x)
x               = UpSampling2D((2, 2))(x)


#model     = Model(encoder_inputs, x)
model      = tf.keras.models.Model(encoder_inputs, x)
model.summary()



def c_loss_1(y_true, y_pred):
    #return  ( tf.keras.losses.binary_crossentropy(y_true, y_pred)) 
    return  ( tf.keras.losses.mean_squared_error(y_true, y_pred))  


def c_loss_2(noise_3):
    noise_3 =  (hashLayer > 0 )
    return   tf.keras.metrics.Sum(noise_3)


adam = Adam(lr=0.001)
model.compile(loss = [c_loss_1, c_loss_2], 
              #loss_weights = [0.9, 0.1],
              optimizer=adam, 
              metrics=['accuracy'],
              )
#model.compile(loss = c_loss_2,  optimizer=sgd, metrics=['accuracy'],)

model.fit(train_generator,
                    steps_per_epoch  = math.ceil(len(data) // Config.batch_size),
                    #validation_data=val_generator,
                    #validation_steps = math.ceil(len(val_data) // Config.batch_size),
                    verbose=1,
                    epochs=Config.num_epochs)


model_json = model.to_json()
with open("models/model_retina.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/weights_retina.h5")  

