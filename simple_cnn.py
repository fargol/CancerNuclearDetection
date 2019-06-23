
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm
#from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'E:/WTERLOO/waterloo_course/DATA_MINING/project/data/nuclei/images/'
TEST_PATH = 'E:/WTERLOO/waterloo_course/DATA_MINING/project/data/nuclei/images/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

 #Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float )
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)): #len(train_ids)
    path = TRAIN_PATH + id_
    img = cv2.imread(path + '/images/' + id_ + '.png') #[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = cv2.imread(path + '/masks/' + mask_file,0)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask/255


ix = random.randint(0, len(train_ids))
imgplot = plt.imshow(X_train[ix])
plt.show()
#imgplot = plt.imshow(Y_train[ix])
#plt.show()
#cv2.imshow('train_X',X_train[ix])
#cv2.waitKey(0)
cv2.imshow('train_Y',255*Y_train[ix])
cv2.waitKey(0)
cv2.destroyAllWindows()

###_____________________________ define error MEASURE__________________________

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 2-dice_coef(y_true, y_pred)


from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda


simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape = (None, None, IMG_CHANNELS), 
                                  name = 'NormalizeInput'))
simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))
simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))
# use dilations to get a slightly larger field of view
simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))
simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))

simple_cnn.add(Conv2D(32, kernel_size = (3,3), dilation_rate = 3, padding = 'same'))

############## hossein
simple_cnn.add(Conv2D(32, kernel_size = (3,3), dilation_rate = 3, padding = 'same'))
simple_cnn.add(Conv2D(64, kernel_size = (3,3), dilation_rate = 3, padding = 'same'))
simple_cnn.add(Conv2D(64, kernel_size = (3,3), dilation_rate = 3, padding = 'same'))
##########hossein

# the final processing
simple_cnn.add(Conv2D(16, kernel_size = (1,1), padding = 'same'))
simple_cnn.add(Conv2D(1, kernel_size = (1,1), padding = 'same', activation = 'sigmoid'))
simple_cnn.summary()



simple_cnn.compile(optimizer = 'adam', 
                   loss = dice_coef_loss, 
                   metrics = [dice_coef, 'acc', 'mse'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

earlystopper =EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-3.h5', verbose=1, save_best_only=True)
results = simple_cnn.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=50, 
                    callbacks=[earlystopper, checkpointer])

ix=523
preds_train = simple_cnn.predict(X_train[ix:ix+5], verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.float)
#ix = random.randint(0, len(preds_train_t))
cv2.imshow('r',X_train[ix])
cv2.waitKey(0)
cv2.imshow('b',np.squeeze(Y_train[ix]))
cv2.waitKey(0)
cv2.imshow('g',255*np.squeeze(preds_train_t[0]))
cv2.waitKey(0)


