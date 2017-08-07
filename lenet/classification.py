from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import cv2
import numpy as np

img_width, img_height = 128, 128
train_data_dir = "data/rotulos.txt"


def read_img():
    pass


def read_file(path):
    with open(path) as f:
        data = f.readlines()
        data = [x.replace('\n', '').split(',') for x in data]
        return data


input_file = read_file(train_data_dir)
train_x = []
train_y = []
for each in input_file:
    img = cv2.imread(each[0])
    train_x.append(img)
    train_y.append(int(each[1]))


model = applications.VGG16(weights="imagenet", include_top=False,
                           input_shape=(img_width, img_height, 3))


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


# Save the model according to the conditions
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1,
                      mode='auto')


train_y = np_utils.to_categorical(train_y)
# Train the model
model_final.fit(np.array(train_x), np.array(train_y), batch_size=32, epochs=10, verbose=1,
                callbacks=[checkpoint, early], validation_split=0.2,
                validation_data=None, shuffle=True, class_weight=None,
                sample_weight=None, initial_epoch=0)
model_final.save_weights('modelo_trainado.h5')


