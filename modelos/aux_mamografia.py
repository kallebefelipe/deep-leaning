import cv2
import numpy as np
from sklearn import cross_validation
import pywt
import os

def load_mammography():

    arq =  open('/Users/filipe/Dropbox/Doutorado/Patches IRMA/12er-patches_separado_densidade/tipo1_circ_esp_misc.txt','r')
    lines = arq.readlines()
    img_names = [x.split(' ')[0] for x in lines]
    labels = [x.split(' ')[1][0] for x in lines]
    dim = 32.0

    database = np.zeros((len(labels),int(dim),int(dim)))
    for it, i in enumerate(img_names):
        #print i
        #load in grayscale
        image = cv2.imread('/Users/filipe/Dropbox/Doutorado/Patches IRMA/12er-patches_separado_densidade/I/'+ i+'.png', 0)

        rx = dim/ image.shape[1]
        ry = dim/ image.shape[0]
        #print rx,ry
        #imager = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        imager = cv2.resize(image, (0,0), fx=rx, fy=ry,interpolation=cv2.INTER_AREA)
        #print imager.shape

        #wavelet
        coeffs = pywt.dwt2(imager,'db1')
        cA, (cH, cV, cD) = coeffs

        wav = cV
        rx = dim/ wav.shape[1]
        ry = dim/ wav.shape[0]
        #print rx,ry
        #imager = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        img_wav = cv2.resize(wav, (0,0), fx=rx, fy=ry,interpolation=cv2.INTER_AREA)

        database[it] = imager
        #database[it] = img_wav

    idx = np.random.choice(len(database), len(database), replace=False)

    #data_random = []
    #target_random = []
    #data_random = np.zeros((len(labels), int(dim),int(dim)))
    data_random = np.zeros((len(labels), database[0].shape[0],database[0].shape[1]))
    target_random = np.zeros((len(labels),1))
    for it, k in enumerate(idx):
        #data_random.append(database[k])
        #target_random.append(labels[k])
        data_random[it] = database[k]
        target_random[it] = int(labels[k])-2   #para comecar de zero (ficar 0 e 1)
    #print type(data_random)
    #print type(database)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_random, target_random, test_size=0.4, random_state=0)
    return ((X_train, y_train), (X_test, y_test))

def load_cromossomo():
    positiveFolder = '/Users/filipe/Desktop/Fernanda/Recorte_pos/'
    negativeFolder = '/Users/filipe/Desktop/Fernanda/Recorte_neg/'

    pos_imgs = [x for x in os.listdir(positiveFolder)]
    pos_imgs = pos_imgs[:595]
    neg_imgs = [x for x in os.listdir(negativeFolder)]

    fixed_width = 50
    fixed_heigh = 50
    dim = 50

    #print pos_imgs
    total_imgs = pos_imgs + neg_imgs

    # initialize the local binary patterns descriptor along with
    # the data and label lists

    database = np.zeros((len(total_imgs),int(dim),int(dim)))
    labels = np.zeros((len(total_imgs),1))

    print 'calculando metrica...\n'
    # loop over the training images
    for k in range(len(total_imgs)):
        #print k
        if k<len(pos_imgs):
            labels[k]=0
            folder = positiveFolder
        else:
            labels[k]=1
            folder = negativeFolder

        #print total_imgs[k]
        #print folder+total_imgs[k]

        image = cv2.imread(folder+total_imgs[k],0)
        #cv2.imshow('tetse',image)
        #cv2.waitKey(0)
        resized = cv2.resize(image, (fixed_heigh,fixed_width), interpolation = cv2.INTER_AREA)

        database[k] = resized

        #thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]




    #idx = np.random.choice(np.arange(len(data),len(data[0])), len(data), replace=False)
    idx = np.random.choice(len(database), len(database), replace=False)

    data_random = np.zeros((len(labels), database[0].shape[0],database[0].shape[1]))
    target_random = np.zeros((len(labels),1))
    for it, k in enumerate(idx):
        #data_random.append(database[k])
        #target_random.append(labels[k])
        data_random[it] = database[k]
        target_random[it] = int(labels[k])   #para comecar de zero (ficar 0 e 1)
    #print type(data_random)
    #print type(database)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_random, target_random, test_size=0.4, random_state=0)
    return ((X_train, y_train), (X_test, y_test))


