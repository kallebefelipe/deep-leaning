from keras.models import model_from_json
from keras import optimizers
import cv2
import numpy as np

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss="categorical_crossentropy",
                     optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                     metrics=["accuracy"])


def read_file(path):
    with open(path) as f:
        data = f.readlines()
        data = [x.replace('\n', '').split(',') for x in data]
        return data


def resize(img):
    new_img = []
    for each in img[:128]:
        new_img.append(each[:128])
    return new_img


input_file = read_file('data/rotulos.txt')
for each in input_file:
    if each[1] == '0':
        img = cv2.imread(each[0])
        # elif each[1] == '1':
        #     img = cv2.imread('data/test/I/'+each[0])
        if img is not None:
            img = np.array(resize(img))
            img = img.reshape(-1, 128, 128, 3)
            predictions = loaded_model.predict(img)
            rounded = [round(x[0]) for x in predictions]
            print(rounded)
