from config import Config
import numpy as np
import cv2


def generator(samples, batch_size=Config.batch_size):
  
    num_samples = len(samples)
   
    while True: 

        for offset in range(0, num_samples, Config.batch_size):
            print("\n")
           
            batch_samples = samples[offset:offset+batch_size]
            print(batch_samples[0])                 
            X = []
            for batch_sample in batch_samples:
              image  = cv2.imread(batch_sample[0])
              image = cv2.resize(image,(Config.resize, Config.resize), interpolation=cv2.INTER_AREA)
              X.append(image)
            X = np.array(X)
            X = X / 255
            X = X.astype(float)
            yield X
