from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import keras
import cv2

img_width, img_height = 128, 128
train_data_dir = "data/train"
validation_data_dir = "data/test"
nb_train_samples = 2436
nb_validation_samples = 360
batch_size = 1
epochs = 1

train_data_dir = "data/rotulos.txt"


def read_file(path):
    with open(path) as f:
        data = f.readlines()
        data = [x.replace('\n', '').split(',') for x in data]
        return data


input_file = read_file(train_data_dir)
train_x = np.zeros((156, 128, 128))
train_y = np.zeros((156))

for each in input_file:
    try:
        img = cv2.imread(each[0])
        train_x.append(img)
        train_y.append(int(each[1]))
    except:
        pass

K.set_image_dim_ordering('tf')
# initialize the model
model = Sequential()

# define the first set of CONV => ACTIVATION => POOL layers
model.add(Convolution2D(20, 5, 5, border_mode="same",
          input_shape=(img_width, img_height, 3)))
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

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])

import ipdb; ipdb.set_trace()
# train_x = train_x.reshape(train_x.shape[0], 1, 128, 128)
train_x = train_x.astype('float32')
train_y /= 255
train_y = keras.utils.to_categorical(train_y, 2)

model.fit(train_x, train_y, batch_size=32, epochs=10,
          verbose=1, shuffle=True,
          initial_epoch=0)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
