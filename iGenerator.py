from config import Config
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from PIL import Image

def generator(samples, batch_size=Config.batch_size):
  
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    
    num_samples = len(samples)
    

    while True: # Loop forever so the generator never terminates
        
        data_all   =  shuffle(samples)
        
        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size &lt;= num_samples]
        for offset in range(0, num_samples, Config.batch_size):
            #print("\n")
            # Get the samples you'll use in this batch
            batch_samples = data_all[offset:offset+batch_size]
            #print(batch_samples[:])           
            
            # Initialise X_train and y_train arrays for this batch
            X = []
            y = []
            # For each example
            for batch_sample in batch_samples:
              #filename = batch_sample
              image  = cv2.imread(batch_sample[0])
              #print(image.shape)
              #image = plt.imread(batch_sample[-1])
              if image.ndim == 1:
              	image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
              image = cv2.resize(image,(Config.resize, Config.resize), interpolation=cv2.INTER_AREA)
              X.append(image)
            X = np.array(X)
            X = X / 255
            X = X.astype(float)
            X = np.array(X)
            X = X / 255
            X = X.astype(float)
            
            y = X
            yield X, y
         

