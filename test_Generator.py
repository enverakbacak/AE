from csv import reader
import numpy as np
import pandas as pd
from config import Config
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



data=[]
labels=[]
# skip first line i.e. read header first and then iterate over each row od csv as a list
with open('../Datasets/NusWIDE/Groundtruth/TRAIN.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            data.append(row)
data=np.array(data)


num_samples = len(data)
batch_size = Config.batch_size

while True: # Loop forever so the generator never terminates

        data       =  np.array(data)
        #data      =  shuffle(data)
        samples    =  data[:,0]
        samples    =  samples.reshape(num_samples,1)
        labels     =  data[:,1:82]
        labels     =  labels.reshape(num_samples,81)
        labels     =  np.array(labels)
        labels     = np.asarray(labels, dtype = float)
        #print(labels.dtype)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size &lt;= num_samples]
        for offset in range(0, num_samples, batch_size):
            print("\n")
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]
            print(batch_samples[:])           
            label_samples = labels[offset:offset+batch_size]
            #print(label_samples[0])                  

            # Initialise X_train and y_train arrays for this batch
            X = []
            y = []
            # For each example
            for batch_sample in batch_samples:
              #filename = batch_sample
              image  = cv2.imread(batch_sample[0])
              if image.ndim != 3:
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
              image = cv2.resize(image,(Config.resize, Config.resize), interpolation=cv2.INTER_AREA)
              X.append(image)
            X = np.array(X)
            X = X / 255
            X = X.astype(float)
            #print(X.dtype)
            #print(X.shape)

            #for label_sample in label_samples:
            #  y.append(label_sample)
            #y = np.array(y)

            #y = y.astype(float)
            #print(y.dtype)
            # The generator-y part: yield the next training batch




