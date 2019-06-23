# this works good without data augmentation.. let see what happen with augmentation


import tensorflow as tf            
import pandas as pd                 
import numpy as np                                       
import sklearn.model_selection     # For using KFold
import keras.preprocessing.image   # For using image generation
import datetime                    # To measure running time 
import skimage.transform           # For resizing images
import skimage.morphology          # For using image labeling
import cv2                         # To read and manipulate images
import os                          # For filepath, directory handling
import sys                         # System-specific parameters and functions
import tqdm                        # Use smart progress meter
import seaborn as sns              # For pairplots
import matplotlib.pyplot as plt    # Python 2D plotting library
import matplotlib.cm as cm         # Color map
# matplotlib inline                  

# Global constants.
IMG_WIDTH = 256       # Default image width
IMG_HEIGHT = 256      # Default image height
IMG_CHANNELS = 3      # Default number of channels
CW_DIR = os.getcwd()  
TRAIN_DIR = 'C:/Users/pc-admin/Desktop/Python_project/data/'
TEST_DIR =  'C:/Users/pc-admin/Desktop/Python_project/data_test/'
IMG_TYPE = '.png'         # Image type
IMG_DIR_NAME = 'images'   # Folder name including the image
MASK_DIR_NAME = 'masks'   # Folder name including the masks
LOGS_DIR_NAME = 'logs'    # Folder name for TensorBoard summaries 
SAVES_DIR_NAME = 'saves'  # Folder name for storing network parameters
SEED = 123                # Random seed for splitting train/validation sets
    
# Global variables.
min_object_size = 1       # Minimal nucleous size in pixels
x_train = []
y_train = []
x_test = []
y_test_pred_proba = {}
y_test_pred = {}

# Display working/train/test directories.
print('CW_DIR = {}'.format(CW_DIR))
print('TRAIN_DIR = {}'.format(TRAIN_DIR))
print('TEST_DIR = {}'.format(TEST_DIR))



# Collection of methods for data operations. Implemented are functions to read  
# images/masks from files and to read basic properties of the train/test data sets.

def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size: 
        img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
    return img

def read_mask(directory, target_size=None):
    """Read and resize masks contained in a given directory."""
    for i,filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, target_size)
        if not i: mask = mask_tmp
        else: mask = np.maximum(mask, mask_tmp)
    return mask 

def read_train_data_properties(train_dir, img_dir_name, mask_dir_name):
    """Read basic properties of training images and masks"""
    tmp = []
    for i,dir_name in enumerate(next(os.walk(train_dir))[1]):

        img_dir = os.path.join(train_dir, dir_name, img_dir_name)
        mask_dir = os.path.join(train_dir, dir_name, mask_dir_name) 
        num_masks = len(next(os.walk(mask_dir))[2])
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0]/img_shape[1], img_shape[2], num_masks,
                    img_path, mask_dir])

    train_df = pd.DataFrame(tmp, columns = ['img_id', 'img_height', 'img_width',
                                            'img_ratio', 'num_channels', 
                                            'num_masks', 'image_path', 'mask_dir'])
    return train_df

def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""
    tmp = []
    for i,dir_name in enumerate(next(os.walk(test_dir))[1]):

        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0]/img_shape[1], img_shape[2], img_path])

    test_df = pd.DataFrame(tmp, columns = ['img_id', 'img_height', 'img_width',
                                           'img_ratio', 'num_channels', 'image_path'])
    return test_df

def imshow_args(x):
    """Matplotlib imshow arguments for plotting."""
    if len(x.shape)==2: return x, cm.gray
    if x.shape[2]==1: return x[:,:,0], cm.gray
    return x, None

def load_raw_data(image_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Load raw data."""
    # Python lists to store the training images/masks and test images.
    x_train, y_train, x_test = [],[],[]

    # Read and resize train images/masks. 
    print('Loading and resizing train images and masks ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(train_df['image_path']), total=len(train_df)):
        img = read_image(train_df['image_path'].loc[i], target_size=image_size)
        mask = read_mask(train_df['mask_dir'].loc[i], target_size=image_size)
        x_train.append(img)
        y_train.append(mask)

    # Read and resize test images. 
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']), total=len(test_df)):
        img = read_image(test_df['image_path'].loc[i], target_size=image_size)
        x_test.append(img)

    # Transform lists into 4-dim numpy arrays.
    x_train = np.array(x_train)
    y_train = np.expand_dims(np.array(y_train), axis=4)
    x_test = np.array(x_test)

    print('x_train.shape: {} of dtype {}'.format(x_train.shape, x_train.dtype))
    print('y_train.shape: {} of dtype {}'.format(y_train.shape, x_train.dtype))
    print('x_test.shape: {} of dtype {}'.format(x_test.shape, x_test.dtype))
    
    return x_train, y_train, x_test


# Basic properties of images/masks. 
train_df = read_train_data_properties(TRAIN_DIR, IMG_DIR_NAME, MASK_DIR_NAME)
test_df = read_test_data_properties(TEST_DIR, IMG_DIR_NAME)
print('train_df:')
print(train_df.describe())
print('')
print('test_df:')
print(test_df.describe())

# Counting unique image shapes.
df = pd.DataFrame([[x] for x in zip(train_df['img_height'], train_df['img_width'])])
print('')
print(df[0].value_counts())


###______________________________________________________ histofram of intensities____________

# Overview of train images/masks. There is a lot of variation concerning
# the form/size/number of nuclei and the darkness/lightness/colorfulness of 
# the images. 
fig, axs = plt.subplots(4,4,figsize=(20,20))
for i in range(4):
    for j in range(2):
        n = np.random.randint(0,len(train_df))
        axs[i,j*2].imshow(read_image(train_df['image_path'].loc[n]))
        axs[i,j*2].set_title('{}. image'.format(n))
        axs[i,j*2+1].imshow(read_mask(train_df['mask_dir'].loc[n]), cmap='gray') 
        axs[i,j*2+1].set_title('{}. mask'.format(n)) 
        
# Read images/masks from files and resize them. Each image and mask 
# is stored as a 3-dim array where the number of channels is 3 and 1, respectively.
x_train, y_train, x_test = load_raw_data()


# Study the pixel intensity. On average the red, green and blue channels have similar
# intensities for all images. It should be noted that the background can be dark 
# (black) as  as well as light (white). 
def img_intensity_pairplot(x):
    """Plot intensity distributions of color channels."""
    df = pd.DataFrame()
    df['Gray'] = np.mean(x[:,:,:,:], axis=(1,2,3))
    if x.shape[3]==3:
        df['Red'] = np.mean(x[:,:,:,0], axis=(1,2))
        df['Blue'] = np.mean(x[:,:,:,1], axis=(1,2))
        df['Green'] = np.mean(x[:,:,:,2], axis=(1,2))
    return df

color_df = img_intensity_pairplot(np.concatenate([x_train, x_test]))
color_df['images'] = ['train']*len(x_train) + ['test']*len(x_test)
sns.pairplot(color_df, hue = 'images');

##____________________________________________________ manipulation___________________________________

# Collection of methods for basic data manipulation like normalizing, inverting, 
# color transformation and generating new images/masks

def normalize_imgs(data):
    """Normalize images."""
    return normalize(data, type_=1)

def normalize_masks(data):
    """Normalize masks."""
    return normalize(data, type_=1)
    
def normalize(data, type_=1): 
    """Normalize data."""
    if type_==0:
        # Convert pixel values from [0:255] to [0:1] by global factor
        data = data.astype(np.float32) / data.max()
    if type_==1:
        # Convert pixel values from [0:255] to [0:1] by local factor
        div = data.max(axis=tuple(np.arange(1,len(data.shape))), keepdims=True) 
        div[div < 0.01*data.mean()] = 1. # protect against too small pixel intensities
        data = data.astype(np.float32)/div
    if type_==2:
        # Standardisation of each image 
        data = data.astype(np.float32) / data.max() 
        mean = data.mean(axis=tuple(np.arange(1,len(data.shape))), keepdims=True) 
        std = data.std(axis=tuple(np.arange(1,len(data.shape))), keepdims=True) 
        data = (data-mean)/std

    return data

def trsf_proba_to_binary(y_data):
    """Transform propabilities into binary values 0 or 1."""  
    return np.greater(y_data,.5).astype(np.uint8)

def invert_imgs(imgs, cutoff=.5):
    '''Invert image if mean value is greater than cutoff.'''
    imgs = np.array(list(map(lambda x: 1.-x if np.mean(x)>cutoff else x, imgs)))
    return normalize_imgs(imgs)

def imgs_to_grayscale(imgs):
    '''Transform RGB images into grayscale spectrum.''' 
    if imgs.shape[3]==3:
        imgs = normalize_imgs(np.expand_dims(np.mean(imgs, axis=3), axis=3))
    return imgs

def generate_images(imgs, seed=None):
    """Generate new images."""
    # Transformations.
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 90., width_shift_range = 0.02 , height_shift_range = 0.02,
        zoom_range = 0.10, horizontal_flip=True, vertical_flip=True)
    
    # Generate new set of images
    imgs = image_generator.flow(imgs, np.zeros(len(imgs)), batch_size=len(imgs),
                                shuffle = False, seed=seed).next()    
    return imgs[0]

def generate_images_and_masks(imgs, masks):
    """Generate new images and masks."""
    seed = np.random.randint(10000) 
    imgs = generate_images(imgs, seed=seed)
    masks = trsf_proba_to_binary(generate_images(masks, seed=seed))
    return imgs, masks

def preprocess_raw_data(x_train, y_train, x_test, grayscale=False, invert=False):
    """Preprocessing of images and masks."""
    # Normalize images and masks
    x_train = normalize_imgs(x_train)
    y_train = trsf_proba_to_binary(normalize_masks(y_train))
    x_test = normalize_imgs(x_test)
    print('Images normalized.')
 
    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_train = imgs_to_grayscale(x_train)
        x_test = imgs_to_grayscale(x_test)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_train = invert_imgs(x_train)
        x_test = invert_imgs(x_test)
        print('Images inverted to remove light backgrounds.')

    return x_train, y_train, x_test




##_______________ cheating__________________________________________________________________________
mask2=cv2.imread('C:/Users/pc-admin/Desktop/Python_project/test_bad_data/mask_2.png',cv2.IMREAD_GRAYSCALE)
mask3=cv2.imread('C:/Users/pc-admin/Desktop/Python_project/test_bad_data/mask_3.png',cv2.IMREAD_GRAYSCALE)
mask4=cv2.imread('C:/Users/pc-admin/Desktop/Python_project/test_bad_data/mask_4.png',cv2.IMREAD_GRAYSCALE)
image_size=(IMG_HEIGHT, IMG_WIDTH)

mask2 = cv2.resize(mask2,image_size , interpolation = cv2.INTER_AREA)
mask3 = cv2.resize(mask3,image_size , interpolation = cv2.INTER_AREA)
mask4 = cv2.resize(mask4,image_size , interpolation = cv2.INTER_AREA)

im2=cv2.imread('C:/Users/pc-admin/Desktop/Python_project/test_bad_data/2.png',cv2.IMREAD_COLOR)
im3=cv2.imread('C:/Users/pc-admin/Desktop/Python_project/test_bad_data/3.png',cv2.IMREAD_COLOR)
im4=cv2.imread('C:/Users/pc-admin/Desktop/Python_project/test_bad_data/4.png',cv2.IMREAD_COLOR)

im2 = cv2.resize(im2,image_size , interpolation = cv2.INTER_AREA)
im3 = cv2.resize(im3,image_size , interpolation = cv2.INTER_AREA)
im4 = cv2.resize(im4,image_size , interpolation = cv2.INTER_AREA)

XX=np.zeros([670+6,IMG_WIDTH,IMG_HEIGHT,3],dtype='float32')
YY=np.zeros([670+6,IMG_WIDTH,IMG_HEIGHT,1],dtype='float32')
XX[0:670,:,:,:]=x_train
XX[670]=im2
XX[671]=im3
XX[672]=im4
XX[673]=im2
XX[674]=im3
XX[675]=im4

YY[0:670,:,:,:]=y_train
YY[670,:,:,0]=mask2/255
YY[671,:,:,0]=mask3/255
YY[672,:,:,0]=mask4/255
YY[673,:,:,0]=mask2/255
YY[674,:,:,0]=mask3/255
YY[675,:,:,0]=mask4/255

x_train=XX
y_train=YY
#_________________end of cheating_____________________________________________________________________





# Normalize all images and masks. There is the possibility to transform images 
# into the grayscale sepctrum and to invert images which have a very 
# light background.
x_train, y_train, x_test = preprocess_raw_data(x_train, y_train, x_test, invert=True)
color_df = img_intensity_pairplot(np.concatenate([x_train, x_test]))
color_df['images'] = ['train']*len(x_train) + ['test']*len(x_test)
sns.pairplot(color_df, hue = 'images');


############### separate data_base____________________________________________________________________
siz=np.size(x_train,0)
x_tr_c=np.zeros([siz,IMG_WIDTH,IMG_HEIGHT,3],dtype='float32')
y_tr_c=np.zeros([siz,IMG_WIDTH,IMG_HEIGHT,1],dtype='float32')

x_tr_g=np.zeros([siz,IMG_WIDTH,IMG_HEIGHT,3],dtype='float32')
y_tr_g=np.zeros([siz,IMG_WIDTH,IMG_HEIGHT,1],dtype='float32')

t_c=0
t_g=0
for ind in range(siz):
    im=x_train[ind]
    if im[1,1,0]==im[1,1,1] and im[1,1,0]==im[1,1,2] and im[50,50,0]==im[50,50,1] and im[50,50,0]==im[50,50,2]:
        x_tr_g[t_g]=im[:,:,0:4]
        y_tr_g[t_g]=y_train[ind]
        t_g+=1
    else:
        x_tr_c[t_c]=im
        y_tr_c[t_c]=y_train[ind]
        t_c+=1     

x_tr_c=x_tr_c[0:t_c]
y_tr_c=y_tr_c[0:t_c]

x_tr_g=x_tr_g[0:t_g]
y_tr_g=y_tr_g[0:t_g]

x_train=x_tr_c
y_train=y_tr_c



x_tr_a, y_tr_a = generate_images_and_masks(x_tr_c, y_tr_c)
x_tr_a1, y_tr_a1 = generate_images_and_masks(x_tr_c, y_tr_c)

sizz=np.size(x_tr_c,0)

XX=np.zeros([sizz*3,IMG_WIDTH,IMG_HEIGHT,3],dtype='float32')
XX[0:sizz,:,:,:]=x_tr_c
XX[sizz:sizz*2,:,:,:]=x_tr_a;
XX[sizz*2:sizz*3,:,:,:]=x_tr_a1;
YY=np.zeros([sizz*3,IMG_WIDTH,IMG_HEIGHT,1],dtype='float32')
YY[0:sizz,:,:,:]=y_tr_c
YY[sizz:sizz*2,:,:,:]=y_tr_a;  
YY[sizz*2:sizz*3,:,:,:]=y_tr_a1;

x_tr_c=XX
y_tr_c=YY

XX=np.zeros([sizz*3+t_g,IMG_WIDTH,IMG_HEIGHT,3],dtype='float32')
YY=np.zeros([sizz*3+t_g,IMG_WIDTH,IMG_HEIGHT,1],dtype='float32')

t2=0
for ind in range(t_g):
    XX[ind+t2]=x_tr_g[ind]
    YY[ind+t2]=y_tr_g[ind]
    if t2<3*sizz :
        t2=t2+1
        XX[ind+t2]=x_tr_c[ind]
        YY[ind+t2]=y_tr_c[ind]   
 
    
### use augmented data


x_train=XX
y_train=YY
############### end of separate data_base______________________________________________________________



#_________________________________________________________________________________________________
x_train*=255
x_test *=255


#from skimage.io import imread, imshow, imread_collection, concatenate_images
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
###_____________________________ define error MEASURE__________________________
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 2-dice_coef(y_true, y_pred)


# ___________________________________________________________Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
#c1 = Dropout(0.1) (c1)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
#c2 = Dropout(0.1) (c2)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
#c3 = Dropout(0.2) (c3)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
#c4 = Dropout(0.2) (c4)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
model.compile(optimizer = 'adam',    loss = dice_coef_loss,   metrics = [dice_coef, 'acc', 'mse'])
model.summary()

# ________________________________________________Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model_gray.h5', verbose=1, save_best_only=True)
results = model.fit(x_train, y_train, validation_split=0.1, batch_size=10, epochs=15, 
                    callbacks=[earlystopper, checkpointer])

#model = load_model('model_augmented_256256.h5', custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef})
ix=0
preds_train = model.predict(x_train[ix:ix+69], verbose=1)
preds_train_t = (preds_train > 0.25).astype(np.float)
#ix = random.randint(0, len(preds_train_t))
ff=x_train[ix].astype('uint8')
plt.imshow(ff)
plt.show()
#cv2.imshow('r',ff)
#cv2.waitKey(0)
#cv2.imshow('b',255*np.squeeze(y_train[ix]))
#cv2.waitKey(0)
#
#plt.imshow(255*np.squeeze(y_train[ix]))
#plt.show()

#cv2.imshow('g',255*np.squeeze(preds_train_t[0]))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.imshow(255*np.squeeze(preds_train_t[0]))
plt.show()

for ind in range(65):
    ff=x_train[ix+ind].astype('uint8')
    plt.imshow(ff)
    plt.show()
#    plt.imshow(255*np.squeeze(y_train[ix+ind]))
#    plt.show()
    plt.imshow(255*np.squeeze(preds_train_t[ind]))
    plt.show()
    