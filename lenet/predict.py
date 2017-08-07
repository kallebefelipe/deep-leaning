from keras.models import model_from_json
from keras import optimizers
import cv2
from keras import backend as K
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


def recortar(img):
    nova_img = []
    for each in img[:128]:
        nova_img.append(each[:128])
    return np.array(nova_img)


test_data_dir = "data/rotulos_test.txt"
input_test = read_file(test_data_dir)
x_test = np.zeros((360, 128, 128))
y_test = []

count = 0
for each in input_test:
    img = cv2.imread(each[0], 0)
    if img is not None:
        x_test[count] = recortar(img)
        y_test.append(int(each[1]))
        count += 1

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, 128, 128)
else:
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

x_test = np.array(x_test)
x_test = x_test.astype('float32')
x_test /= 255


# for count, each in enumerate(x_test):
#     each = each.reshape(-1, 128, 128, 1)
#     predictions = loaded_model.predict(each)
#     rounded = [round(x[0]) for x in predictions]
#     print('Predicao: '+str(predictions)+' | Classe:'+str(y_test[count]))
