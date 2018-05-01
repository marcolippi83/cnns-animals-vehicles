from keras.applications.densenet import DenseNet201
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import sys

data_dir = sys.argv[1]

model = DenseNet201(weights='imagenet')

for filename in os.listdir(data_dir):
    imgfile = data_dir + '/' + filename
    img = image.load_img(imgfile, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print(filename + ' --- ' + str(decode_predictions(preds, top=5)[0]))

