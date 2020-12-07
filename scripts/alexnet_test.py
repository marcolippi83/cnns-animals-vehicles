from keras.preprocessing import image
from keras.models import load_model
import PIL
from PIL import Image
import numpy as np
import os
import sys
from convnetskeras.convnets import convnet
from keras.layers import Dropout, Dense
from keras.models import Sequential

test_images_list_file = sys.argv[1]
test_datadir = sys.argv[2]
saved_model = sys.argv[3]
train_stats_file = sys.argv[4]
outfile = sys.argv[5]

import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras import backend
backend.set_image_data_format('channels_first')

list_test_images = []
test_labels = []

with open(test_images_list_file) as f:
    for line in f:
        values = line.strip().split()
        list_test_images.append(values[0])
        test_labels.append(values[1])

num_test_examples = len(test_labels)

X_test = np.zeros((num_test_examples,3,227,227))
y_test = np.zeros(num_test_examples, dtype=int)
for i in range(len(test_labels)):
    y_test[i] = int(test_labels[i])

for i in range(len(test_labels)):
    y_test[i] = int(test_labels[i])
for i in range(num_test_examples):
    img_file = test_datadir + '/' + list_test_images[i]
    img = image.load_img(img_file, target_size=(227, 227))
    x = image.img_to_array(img)
    x = x.astype('float32')
    x[0, :, :] -= 123.68
    x[1, :, :] -= 116.779
    x[2, :, :] -= 103.939
    X_test[i,:,:,:] = x

base_model = convnet('alexnet', weights_path="../models/alexnet_weights.h5", heatmap=False)
model = Sequential()
base_model.layers.pop()
base_model.layers.pop()
model.add(base_model)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.load_weights(saved_model)

#model = load_model(saved_model)

predictions_test = model.predict_proba(X_test)
np.savetxt(outfile, predictions_test, fmt='%f')

