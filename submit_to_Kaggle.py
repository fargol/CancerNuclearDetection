import cv2
import tensorflow as tf   
import numpy as np
import pandas as pd                 
import keras.preprocessing.image   # For using image generation
import os                          # For filepath, directory handling
import sys                         # System-specific parameters and functions
import tqdm                        # Use smart progress meter
import seaborn as sns              # For pairplots
import matplotlib.pyplot as plt    # Python 2D plotting library
import matplotlib.cm as cm         # Color map
#from skimage.io import imread, imshow, imread_collection, concatenate_images
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

# Global constants.
IMG_WIDTH = 256       # Default image width
IMG_HEIGHT = 256      # Default image height
IMG_CHANNELS = 3      # Default number of channels
CW_DIR = os.getcwd()  
TRAIN_DIR = 'E:/WTERLOO/waterloo_course/DATA_MINING/project/data/nuclei/images/'
TEST_DIR =  'E:/WTERLOO/waterloo_course/DATA_MINING/project/data/nuclei/test_data/'
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
    x_test = []

    # Read and resize train images/masks. 
    

    # Read and resize test images. 
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']), total=len(test_df)):
        print(type(test_df['image_path'].loc[i]))
        img = read_image(test_df['image_path'].loc[i], target_size=image_size)
        x_test.append(img)

    # Transform lists into 4-dim numpy arrays.
    x_test = np.array(x_test)

    print('x_test.shape: {} of dtype {}'.format(x_test.shape, x_test.dtype))
    
    return  x_test


## Basic properties of images/masks. 
train_df = read_train_data_properties(TRAIN_DIR, IMG_DIR_NAME, MASK_DIR_NAME)
test_df = read_test_data_properties(TEST_DIR, IMG_DIR_NAME)
#print('train_df:')
#print(train_df.describe())
#print('')
#print('test_df:')
#print(test_df.describe())

# Counting unique image shapes.
df = pd.DataFrame([[x] for x in zip(train_df['img_height'], train_df['img_width'])])
print('')
print(df[0].value_counts())


###______________________________________________________ histofram of intensities____________

# Overview of train images/masks. There is a lot of variation concerning
# the form/size/number of nuclei and the darkness/lightness/colorfulness of 
# the images. 
#fig, axs = plt.subplots(4,4,figsize=(20,20))
#for i in range(4):
#    for j in range(2):
#        n = np.random.randint(0,len(train_df))
#        axs[i,j*2].imshow(read_image(train_df['image_path'].loc[n]))
#        axs[i,j*2].set_title('{}. image'.format(n))
#        axs[i,j*2+1].imshow(read_mask(train_df['mask_dir'].loc[n]), cmap='gray') 
#        axs[i,j*2+1].set_title('{}. mask'.format(n)) 
        
# Read images/masks from files and resize them. Each image and mask 
# is stored as a 3-dim array where the number of channels is 3 and 1, respectively
#x_test1 = load_raw_data()
        
#import imageio


TRAIN_ERROR_IDS = [
    '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80'
]
def image_ids_in(root_dir, is_train_data=False):
    ids = []
    for id in os.listdir(root_dir):
        if id in TRAIN_ERROR_IDS:
            print('Skipping ID due to bad training data:', id)
        else:
            ids.append(id)
    return ids

TEST_IMAGE_IDS = image_ids_in(TEST_DIR)

def load_images(root_dir, ids, get_masks=False):
    images = []
    masks = []
    image_sizes = []
    for id in ids:
        item_dir = root_dir + '/' + id
        image_path = item_dir + '/' + 'images' + '/' + (id + '.png')

        image = cv2.imread(str(image_path))
        image = image[:, :, :3] # remove the alpha channel as it is not used
        images.append(image)
        image_sizes.append(image.shape[:2])
        if get_masks:
            mask_sequence = []
            masks_dir = item_dir + '/' + 'masks'
            mask_paths = masks_dir.glob('*.png')
            for mask_path in mask_paths:
                mask = cv2.imread(str(mask_path),0) # 0 and 255 values
                mask = (mask > 0).astype(np.uint8) # 0 and 1 values
                mask_sequence.append(mask)
            masks.append(mask_sequence)
    if get_masks:
        return images, masks, image_sizes
    else:
        return images, image_sizes

x_test_, TEST_IMAGE_SIZES  = load_images(TEST_DIR, TEST_IMAGE_IDS, False)
x_test = np.zeros([65,IMG_WIDTH,IMG_HEIGHT,3],dtype='uint8')
c = 0
for i in x_test_:
    x_test[c] = cv2.resize(i,(IMG_WIDTH,IMG_HEIGHT))
    c = c+1


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

def preprocess_raw_data( x_test, grayscale=False, invert=False):
    """Preprocessing of images and masks."""
    # Normalize images and masks

    x_test = normalize_imgs(x_test)
    print('Images normalized.')
 
    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_test = imgs_to_grayscale(x_test)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_test = invert_imgs(x_test)
        print('Images inverted to remove light backgrounds.')

    return  x_test



    

# Normalize all images and masks. There is the possibility to transform images 
# into the grayscale sepctrum and to invert images which have a very 
# light background.
x_test = preprocess_raw_data(x_test, invert=True)






x_test *= 255

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


	#####################################################################################################################
	
model = load_model('model_augmented_256256_new.h5',custom_objects={'mean_iou': mean_iou}) ### load  model

ix=0
preds_test = model.predict(x_test[ix:ix+65], verbose=1)
preds_test = (preds_test > 0.5).astype(np.float)

model2 = load_model('model_augmented_256256_newloss_1.h5', custom_objects={'mean_iou': mean_iou}) # load new model
preds_test2 = model2.predict(x_test[ix:ix+65], verbose=1)
preds_test2 = (preds_test2 > 0.5).astype(np.float)
#
preds_test2=1-preds_test2 #combine the results
preds_test=preds_test*preds_test2
for ind in range(65):
    ff=x_test[ix+ind]
#    ff[:,:,1]=ff[:,:,1]+50*np.squeeze(preds_train_t[ind])
    ff=ff.astype('uint8')
    plt.imshow(ff)
    plt.show()
    plt.imshow(255*np.squeeze(preds_test[ind]).astype('uint8'))
    plt.show()
    

#####################################################################################################################


#
#def rle_encode(mask):
#    pixels = mask.T.flatten()
#    # We need to allow for cases where there is a '1' at either end of the sequence.
#    # We do this by padding with a zero at each end when needed.
#    use_padding = False
#    if pixels[0] or pixels[-1]:
#        use_padding = True
#        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
#        pixel_padded[1:-1] = pixels
#        pixels = pixel_padded
#    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
#    if use_padding:
#        rle = rle - 1
#    rle[1::2] = rle[1::2] - rle[:-1:2]
#    return rle
#
#
#def rle_to_string(runs):
#    return ' '.join(str(x) for x in runs)
#
#def rle_encoding(x):
#    dots = np.where(x.T.flatten() == 1)[0]
#    run_lengths = []
#    prev = -2
#    for b in dots:
#        if (b>prev+1): run_lengths.extend((b + 1, 0))
#        run_lengths[-1] += 1
#        prev = b
#    return run_lengths

#preds_test_upsampled = []
#for i in range(len(preds_test)):
#    preds_test_upsampled.append(cv2.resize(np.squeeze(preds_test[i]), 
#                                       (TEST_IMAGE_SIZES[i][1],TEST_IMAGE_SIZES[i][0]), 
#                                       mode='constant', preserve_range=True))

from skimage.morphology import label

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


from skimage.transform import resize


preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (TEST_IMAGE_SIZES[i][0],TEST_IMAGE_SIZES[i][1]), 
                                       mode='constant', preserve_range=True))

new_test_ids = []
rles = []
for n, id_ in enumerate(TEST_IMAGE_IDS):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

import pandas as pd
#Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)

#def rle_code_hossein(mask):
#    dots = np.where(mask.T.flatten()==1)[0] # .T sets Fortran order down-then-right
#    run_lengths = []
#    prev = -2
#    for b in dots:
#        if (b>prev+1): run_lengths.extend((b+1, 0))
#        run_lengths[-1] += 1
#        prev = b
#    return run_lengths
#
#csv_file = open('Submission.csv', 'w')
#csv_file.write('ImageId,EncodedPixels\n')
##
#for i in range(65):
#    mask=np.squeeze(preds_test[i])
#    mask = cv2.resize(mask, (TEST_IMAGE_SIZES[i][1],TEST_IMAGE_SIZES[i][0]))
#    mask = (mask > 0.5).astype(np.float)
##    cv2.imshow('h',mask*255)
##    cv2.waitKey(0)
#    
#    id_ = TEST_IMAGE_IDS[i]
#    #temp = rle_to_string(rle_encoding(mask))
#    temp=rle_code_hossein(mask)
#    temp = ' '.join(str(x) for x in temp)
#    temp = id_ + ',' + temp + '\n'
#    csv_file.write(temp)
#
#csv_file.close()