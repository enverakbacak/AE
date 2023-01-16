import math
import cv2
from tensorflow.keras import Input, Model
from tensorflow.keras.models import model_from_json, load_model
import numpy as np    # for mathematical operations
from config2 import Config2
from iGenerator_forExtraction import generator



train_data=[]
from csv import reader
with open('../Datasets/NusWIDE/labels/nuswide_test.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    if header != None:
        for row in csv_reader:
            train_data.append(row)
train_data=np.array(train_data)
print(train_data.shape)

train_generator   = generator(train_data, batch_size=Config2.batch_size)

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.models import model_from_json, load_model
import numpy as np    # for mathematical operations

np.set_printoptions(linewidth=1024)

json_file = open('./models/model_NUS_16.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("./models/weights_NUS_16_.h5")
model.summary()
model = Model(model.input, model.get_layer('hashLayer').output)

features = model.predict(train_generator, 
                                    steps=math.ceil(len(train_data) // Config2.batch_size))

features = features.astype(float)
np.savetxt('../Datasets/NusWIDE/hashCodes/features_NUS_test_16.txt',features, fmt='%f')
#features  = np.sign(features)
hashCodes = features > 0
hashCodes = hashCodes.astype(int)
np.savetxt('../Datasets/NusWIDE/hashCodes/hashCodes_NUS_test_16.txt',hashCodes, fmt='%d')
