from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import PIL
from PIL import Image
import numpy as np
import os
import sys

test_images_list_file = sys.argv[1]
test_datadir = sys.argv[2]
saved_model = sys.argv[3]
train_stats_file = sys.argv[4]
outfile = sys.argv[5]

list_test_images = []
test_labels = []

with open(test_images_list_file) as f:
    for line in f:
        values = line.strip().split()
        list_test_images.append(values[0])
        test_labels.append(values[1])

num_test_examples = len(test_labels)

X_test = np.zeros((num_test_examples,224,224,3))
y_test = np.zeros(num_test_examples, dtype=int)
for i in range(len(test_labels)):
    y_test[i] = int(test_labels[i])

with open(train_stats_file) as f:
    mean_train = float(f.readline())
    std_train = float(f.readline())

X_test = np.zeros((num_test_examples,224,224,3))
y_test = np.zeros(num_test_examples, dtype=int)
for i in range(len(test_labels)):
    y_test[i] = int(test_labels[i])
for i in range(num_test_examples):
    img_file = test_datadir + '/' + list_test_images[i]
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X_test[i,:,:,:] = x
X_test = X_test / 255.0
X_test -= mean_train
X_test /= std_train

model = load_model(saved_model)
predictions_test = model.predict_proba(X_test)
np.savetxt(outfile, predictions_test, fmt='%f')

