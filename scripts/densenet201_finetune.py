from keras.applications.densenet import DenseNet201
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.optimizers import SGD, Adam
from keras.layers import GlobalAveragePooling2D, Flatten, Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import math
import PIL
from PIL import Image
import numpy as np
import os
import sys

train_images_list_file = sys.argv[1]
datadir = sys.argv[2]
outdir = sys.argv[3]

list_train_images = []
train_labels = []

with open(train_images_list_file) as f:
    for line in f:
        values = line.strip().split()
        list_train_images.append(values[0])
        train_labels.append(values[1])

num_train_examples = len(train_labels)

X_train = np.zeros((num_train_examples,224,224,3))
y_train = np.zeros(num_train_examples, dtype=int)
for i in range(len(train_labels)):
    y_train[i] = int(train_labels[i])

for i in range(num_train_examples):
    img_file = datadir + '/' + list_train_images[i]
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X_train[i,:,:,:] = x

X_train = X_train / 255.0
mean_train = np.mean(X_train)
std_train = np.std(X_train)
X_train -= mean_train
X_train /= std_train

with open('X_train_stats_densenet201.txt', 'w') as out_file:
    out_file.write(str(mean_train) + "\n")
    out_file.write(str(std_train) + "\n")

base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224,224,3))

# this is the model we will train
model = Sequential()
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_acc', patience=5)
model.fit(X_train, y_train, epochs=3, validation_split=0.1, shuffle=True, callbacks=[early_stopping])

# we chose to train the top 2 blocks, i.e. we will freeze
# the first 199 layers and unfreeze the rest:
for layer in model.layers[:199]:
   layer.trainable = False
for layer in model.layers[199:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(X_train,y_train, epochs=3, validation_split=0.1, shuffle=True, callbacks=[early_stopping])

#SAVE MODEL HERE!!!
model.save('densenet201_vehicles_animals.h5')

