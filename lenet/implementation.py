'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
from keras import applications
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
import keras
from keras import optimizers
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import cv2

batch_size = 128
num_classes = 2
epochs = 1

# input image dimensions
img_rows, img_cols = 128, 128

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
train_data_dir = "data/rotulos_train.txt"
test_data_dir = "data/rotulos_test.txt"


def read_file(path):
    with open(path) as f:
        data = f.readlines()
        data = [x.replace('\n', '').split(',') for x in data]
        return data


def recortar(img):
    nova_img = []
    for each in img[:128]:
        nova_img.append(each[:128])
    return np.array(nova_img)


input_train = read_file(train_data_dir)
input_test = read_file(test_data_dir)
x_train = np.zeros((2436, 128, 128, 3))
y_train = []
x_test = np.zeros((360, 128, 128, 3))
y_test = []


count = 0
for each in input_train:
    img = cv2.imread(each[0])
    if img is not None:
        if img.shape[0] == 128:
            x_train[count] = img
        elif img.shape[0] == 130:
            x_train[count] = recortar(img)
        y_train.append(int(each[1]))
        count += 1

count = 0
for each in input_test:
    img = cv2.imread(each[0])
    if img is not None:
        if img.shape[0] == 128:
            x_test[count] = img
        elif img.shape[0] == 130:
            x_test[count] = recortar(img)
        y_test.append(int(each[1]))
        count += 1

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = applications.VGG16(weights="imagenet", include_top=False,
                           input_shape=(img_rows, img_cols, 3))


# Freeze the layers which you don't want to train. Here I am freezing the
# first 5 layers.
for layer in model.layers[:15]:
    layer.trainable = False

# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)

# compile the model
model_final.compile(loss="categorical_crossentropy",
                    optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])


model_final.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))

# serialize model to JSON
model_json = model_final.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_final.save_weights("model.h5")
print("Saved model to disk")
