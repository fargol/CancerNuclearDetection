import pickle
import numpy as np
import cv2
from random import shuffle

X_train = pickle.load( open( "A_X_train.pickle", "rb" ) )
Y_train = pickle.load( open( "A_Y_train.pickle", "rb" ) )
X_train = X_train.astype(np.uint8)
Y_train = Y_train.astype(np.uint8)

LOCAL = 30
GLOBAL = 32
HL = LOCAL//2
HG = GLOBAL//2

MAX_NUM_SAMP = 100

y_temp = []
local_patch_collection = np.zeros([3*MAX_NUM_SAMP*X_train.shape[0],LOCAL+1,LOCAL+1,3],dtype='uint8')
global_patch_collection = np.zeros([3*MAX_NUM_SAMP*X_train.shape[0],LOCAL+1,LOCAL+1,3],dtype='uint8')
t = 0

#YY=Y_train.copy()
#for ind in range(0,np.size(YY,0),1):
#    ff=YY[ind].copy()
#    YY[ind]*=0
#    for i in range(1,256-1,1):
#        for j in range(1,256-1,1):
#            if ff[i,j,0]==1 and np.sum(ff[i-1:i+2,j-1:j+2,0])<8:
#                YY[ind,i-1:i+2,j-1:j+2]=1
#
#vv=Y_train*YY
#YY=YY-vv

for j in range(X_train.shape[0]):
    img = Y_train[j,:,:,:]
    n_idx = np.where(img != 0)
    b_idx = np.where(img == 0)

    kernel = np.ones((5,5),np.uint8)
    
    dilation = cv2.dilate(img,kernel,iterations = 1).reshape(
            img.shape[0],img.shape[1],img.shape[2])
    border = dilation - img
    
#    erosion = cv2.erode(img,kernel,iterations = 1).reshape(
#            img.shape[0],img.shape[1],img.shape[2])
#    border = img - erosion
    bd_idx = np.where(border != 0)

    shuffled_indices_n = [i for i in range(1,n_idx[0].shape[0])]
    shuffled_indices_bd = [i for i in range(1,bd_idx[0].shape[0])]
    shuffled_indices_b = [i for i in range(1,b_idx[0].shape[0])]
    min_size = min([n_idx[0].shape[0], bd_idx[0].shape[0], b_idx[0].shape[0]])
    min_size = min([MAX_NUM_SAMP, min_size])
    shuffle(shuffled_indices_n)
    shuffle(shuffled_indices_bd)
    shuffle(shuffled_indices_b)

    img1 = X_train[j,:,:,:]
    reflect = cv2.copyMakeBorder(img1,HG,HG,HG,HG,cv2.BORDER_REFLECT)
    for i in range(min_size-1):
        
        x = n_idx[0][shuffled_indices_n[i]] + HG
        y = n_idx[1][shuffled_indices_n[i]] + HG
        local_patch_temp = reflect[x-HL:x+HL+1,y-HL:y+HL+1,:]
        local_patch_collection[t,:,:,:] = local_patch_temp
#        global_patch_temp = reflect[x-HG:x+HG+1,y-HG:y+HG+1,:]
#        global_patch_collection[t,:,:,:] = cv2.resize(global_patch_temp, (LOCAL+1,LOCAL+1))
        t += 1
        y_temp.append([0])

        x = bd_idx[0][shuffled_indices_bd[i]] + HG
        y = bd_idx[1][shuffled_indices_bd[i]] + HG
        local_patch_temp = reflect[x-HL:x+HL+1,y-HL:y+HL+1,:]
        local_patch_collection[t,:,:,:] = local_patch_temp
#        global_patch_temp = reflect[x-HG:x+HG+1,y-HG:y+HG+1,:]
#        global_patch_collection[t,:,:,:] = cv2.resize(global_patch_temp, (LOCAL+1,LOCAL+1))
        t += 1
        y_temp.append([1])

        x = b_idx[0][shuffled_indices_b[i]] + HG
        y = b_idx[1][shuffled_indices_b[i]] + HG
        local_patch_temp = reflect[x-HL:x+HL+1,y-HL:y+HL+1,:]
        local_patch_collection[t,:,:,:] = local_patch_temp
#        global_patch_temp = reflect[x-HG:x+HG+1,y-HG:y+HG+1,:]
#        global_patch_collection[t,:,:,:] = cv2.resize(global_patch_temp, (LOCAL+1,LOCAL+1))
        t += 1
        y_temp.append([0])
        
    print("STAGE: ", j, "OUT OF", X_train.shape[0])


target = np.array(y_temp)

#glbl = global_patch_collection[:t]
lcl = local_patch_collection[:t]
pickle.dump(lcl, open('local.pickle', 'wb'))
#pickle.dump(glbl, open('global.pickle', 'wb'))
pickle.dump(target, open('target.pickle', 'wb'))