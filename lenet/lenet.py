'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import cv2

batch_size = 128
num_classes = 2
epochs = 50

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
x_train = np.zeros((2436, 128, 128))
y_train = []
x_test = np.zeros((360, 128, 128))
y_test = []

count = 0
for each in input_train:
    img = cv2.imread(each[0], 0)
    if img is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        if img.shape[0] == 128:
            x_train[count] = img
        elif img.shape[0] == 130:
            x_train[count] = recortar(img)
        y_train.append(int(each[1]))
        count += 1

count = 0
for each in input_test:
    img = cv2.imread(each[0], 0)
    if img is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        if img.shape[0] == 128:
            x_test[count] = img
        elif img.shape[0] == 130:
            x_test[count] = recortar(img)
        y_test.append(int(each[1]))
        count += 1

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

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

K.set_image_dim_ordering('tf')
# initialize the model
model = Sequential()

# define the first set of CONV => ACTIVATION => POOL layers
model.add(Convolution2D(20, 5, 5, border_mode="same",
          input_shape=input_shape))
model.add(Activation("tanh"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# define the second set of CONV => ACTIVATION => POOL layers
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(Activation("tanh"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# define the first FC => ACTIVATION layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("tanh"))

# define the second FC layer
model.add(Dense(2))

# lastly, define the soft-max classifier
model.add(Activation("softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# for count, each in enumerate(x_test):
#     each = each.reshape(-1, 128, 128, 1)
#     predictions = model.predict(each)
#     rounded = [round(x[0]) for x in predictions]
#     print('Predicao: '+str(rounded)+' | Classe:'+str(y_test))
