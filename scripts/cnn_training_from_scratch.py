from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import numpy as np
import random
import sys
import PIL
from PIL import Image
from keras import backend as K

def print_results(y_test, predictions):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(predictions)):
        if (y_test[i] == 1):
            if (predictions[i] == 1):
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if (predictions[i] == 1):
                fp = fp + 1
            else:
                tn = tn + 1

    if (tp + fp > 0):
        precision = float(tp) / (tp + fp)
    else:
        precision = 0

    if (tp + fn > 0):
        recall = float(tp) / (tp + fn)
    else:
        recall = 0

    if (precision + recall > 0):
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0

    accuracy = float(tp + tn) / (tp + tn + fp + fn)

    print('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn))
    print('Accuracy: ' + str(accuracy) + ' Precision: ' + str(precision) + ' Recall: ' + str(recall) + ' F1: ' + str(f1))

    return f1

K.set_image_dim_ordering('th')

train_images_list_file = sys.argv[1]
test_images_list_file = sys.argv[2]
train_labels_list_file = sys.argv[3]
test_labels_list_file = sys.argv[4]
datadir = sys.argv[5]
config_file = sys.argv[6]
outdir = sys.argv[7]
fold = sys.argv[8]

WIDTH = 200
HEIGHT = 150

with open(config_file) as f:
    for line in f:
        values = [val for val in line.split()]
        if (values[0][0] == '#'):
            continue
        if (values[0] == 'num_convolutional_layers'):
            num_convolutional_layers = int(values[2])
        if (values[0] == 'filter_dimension'):
            filter_dimension = [int(val) for val in values[2].split(',')]
        if (values[0] == 'num_filters'):
            num_filters = [int(val) for val in values[2].split(',')]
        if (values[0] == 'subsample_size'):
            subsample_size = [int(val) for val in values[2].split(',')]
        if (values[0] == 'validation_percentage'):
            validation_percentage = float(values[2])
        if (values[0] == 'num_iterations'):
            num_iterations = int(values[2])
        if (values[0] == 'batch_size'):
            batch_size = int(values[2])

list_train_images = []
train_labels = []
list_test_images = []
test_labels = []

with open(train_images_list_file) as f:
    for line in f:
        list_train_images.append(line.strip())

with open(train_labels_list_file) as f:
    for line in f:
        train_labels.append(line.strip())

with open(test_images_list_file) as f:
    for line in f:
        list_test_images.append(line.strip())

with open(test_labels_list_file) as f:
    for line in f:
        test_labels.append(line.strip())

num_test_examples = len(test_labels)
num_train_examples = len(train_labels)

y_train = np.zeros(num_train_examples, dtype=int)
y_test = np.zeros(num_test_examples, dtype=int)
X_train = np.zeros((num_train_examples,1,HEIGHT,WIDTH))
X_test = np.zeros((num_test_examples,1,HEIGHT,WIDTH))

for i in range(len(train_labels)):
    y_train[i] = int(train_labels[i])

for i in range(len(test_labels)):
    y_test[i] = int(test_labels[i])

for i in range(num_train_examples):
    img = Image.open(datadir + '/' + list_train_images[i])
    img = img.resize((WIDTH,HEIGHT), PIL.Image.ANTIALIAS)
    x = np.array(img)
    x = x.reshape((1,) + x.shape)
    X_train[i,:,:,:] = x

for i in range(num_test_examples):
    img = Image.open(datadir + '/' + list_test_images[i])
    img = img.resize((WIDTH,HEIGHT), PIL.Image.ANTIALIAS)
    x = np.array(img)
    x = x.reshape((1,) + x.shape)
    X_test[i,:,:,:] = x

X_train = X_train / 255.0
X_test = X_test / 255.0

mean_train = np.mean(X_train)
std_train = np.std(X_train)
X_train -= mean_train
X_train /= std_train
X_test -= mean_train
X_test /= std_train

model = Sequential()
for i in range(num_convolutional_layers):
    if i == 0:
        model.add(Conv2D(num_filters[0], filter_dimension[0], input_shape=X_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(subsample_size[i], subsample_size[i])))
        model.add(Dropout(0.2))
    else:
        model.add(Conv2D(num_filters[i], filter_dimension[i]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(subsample_size[i], subsample_size[i])))
        model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.0001, decay=1e-5, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])

#adam = Adam(lr=0.00001)
#model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

#rms = RMSprop(lr=0.0000001)
#model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['acc'])

earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
model.fit(X_train, y_train, batch_size=batch_size, validation_split=validation_percentage, epochs=100000, callbacks=[earlyStopping])

predictions_test = model.predict_classes(X_test)
print_results(y_test, predictions_test)
np.savetxt(outdir + '/predictions_' + str(fold) + '.txt', predictions_test, fmt='%d')
