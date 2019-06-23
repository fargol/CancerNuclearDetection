from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import numpy as np
import pickle

X_train = pickle.load( open( "local.pickle", "rb" ) )
y_train = pickle.load( open( "target.pickle", "rb" ) )
Y_train = np_utils.to_categorical(y_train, 2)
X_train = X_train.astype('float32')
X_train /= np.max(X_train)

batch_size = 30
num_epochs = 2

kernel_size_1 = 5
kernel_size_2 = 6
kernel_size_3 = 7
kernel_size_4 = 8

pool_size_1 = 2
pool_size_2 = 2

conv_depth_1 = 10
conv_depth_2 = 20
conv_depth_3 = 40

drop_prob_1 = 0.25
drop_prob_2 = 0.5

hidden_size_1 = 512
hidden_size_2 = 256
hidden_size_3 = 128

num_classes = 2

num_train, height, width, depth = X_train.shape

inp = Input(shape=(height, width, depth))

conv_1 = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), 
                       padding='same', activation='relu')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size_1, pool_size_1))(conv_1)
drop_1 = Dropout(drop_prob_1)(pool_1)

conv_2 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), 
                       padding='same', activation='relu')(drop_1)
pool_2 = MaxPooling2D(pool_size=(pool_size_1, pool_size_1))(conv_2)
drop_2 = Dropout(drop_prob_1)(pool_2)

conv_3 = Convolution2D(conv_depth_2, (kernel_size_3, kernel_size_3), 
                       padding='same', activation='relu')(drop_2)
pool_3 = MaxPooling2D(pool_size=(pool_size_2, pool_size_2))(conv_3)
drop_3 = Dropout(drop_prob_1)(pool_3)

conv_4 = Convolution2D(conv_depth_2, (kernel_size_4, kernel_size_4), 
                       padding='same', activation='relu')(drop_3)
pool_4 = MaxPooling2D(pool_size=(pool_size_2, pool_size_2))(conv_4)
drop_4 = Dropout(drop_prob_1)(pool_4)

flat = Flatten()(drop_4)

hidden_1 = Dense(hidden_size_1, activation='relu')(flat)
drop_4 = Dropout(drop_prob_2)(hidden_1)

hidden_2 = Dense(hidden_size_2, activation='relu')(drop_4)
drop_5 = Dropout(drop_prob_2)(hidden_2)

hidden_3 = Dense(hidden_size_3, activation='relu')(drop_5)
drop_6 = Dropout(drop_prob_2)(hidden_3)

out = Dense(num_classes, activation='softmax')(drop_6)

model = Model(inputs=inp, outputs=out)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1)

#model.evaluate(X_test, Y_test, verbose=1)

#########################

model.save('my_model.h5')

import numpy as np
import pickle
from keras.models import load_model
import cv2

X = pickle.load(open('A_X_train.pickle', 'rb'))
Y = pickle.load(open('A_Y_train.pickle', 'rb'))
model = load_model('my_model.h5')

def test_net(x, index = 0):
    
    print('\nWait...')
    
    LOCAL = 30
    HL = LOCAL//2
    
    reflected_input = cv2.copyMakeBorder(x[index],HL,HL,HL,HL,
                                         cv2.BORDER_REFLECT)
    local_patch_holder = np.zeros([1,LOCAL+1,LOCAL+1,3],np.float)
    pred_mask = np.zeros([x[index].shape[0],x[index].shape[0]],np.float)
    
    for ind1 in range(HL,np.size(reflected_input,0)-HL):
        for ind2 in range(HL,np.size(reflected_input,1)-HL):
            local_patch_holder[0] = reflected_input[ind1-HL:ind1+HL+1,
                              ind2-HL:ind2+HL+1]
            pred_mask_prob = model.predict(local_patch_holder,verbose=0)
            pred_mask[ind1-HL,ind2-HL] = pred_mask_prob[0,1]*255
            
    return pred_mask
    
##########################
pred_mask = test_net(172, X)


cv2.imshow('Output',pred_mask)
cv2.imshow('Image',X[172]) 
#cv2.imshow('Ground Truth',y[index]) 
cv2.waitKey(0)