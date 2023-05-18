import math
import cv2
from tensorflow.keras import Input, Model
from tensorflow.keras.models import model_from_json, load_model
import numpy as np    # for mathematical operations
from config import Config
from iGenerator_forExtraction import generator



train_data=[]
from csv import reader
# skip first line i.e. read header first and then iterate over each row od csv as a list
with open('../Datasets/retinamnist/retinamnist_labels/retinamnist_test.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            train_data.append(row)
num_samples   =  len(train_data)
train_data    =  np.array(train_data)
train_data    =  train_data[:,0]
train_data    =  train_data.reshape(num_samples,1)
print(train_data.shape)

# Create generator
train_generator   = generator(train_data, batch_size=Config.batch_size)

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.models import model_from_json, load_model
import numpy as np    # for mathematical operations

np.set_printoptions(linewidth=1024)

json_file = open('./models/model_16.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("./models/weights_16.h5")
model.summary()
model = Model(model.input, model.get_layer('hashLayer').output)

features = model.predict_generator(train_generator, 
                                    steps=math.ceil(len(train_data) // Config.batch_size))

features = features.astype(float)
np.savetxt('../Datasets/retinamnist/hashCodes/features_test_16.txt',features, fmt='%f')
#features  = np.sign(features)
hashCodes = features > 0
hashCodes = hashCodes.astype(int)
np.savetxt('../Datasets/retinamnist/hashCodes/hashCodes_test_16.txt',hashCodes, fmt='%d')
