# coding=utf-8

from glob import glob
from scipy.misc import imresize
from skimage import color, io
import numpy as np
from sklearn.model_selection import train_test_split
import os

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
from tflearn import optimizers

###################################
### Import picture files
###################################

files_path ='tmp/input/convert PNG to JPG/New folder'
files_path1 = 'tmp/output/outputJPG/New folder'
nwa_files_path = os.path.join(files_path, '*.jpg')
wa_files_path = os.path.join(files_path1, '*.jpg')

wa_files = sorted(glob(wa_files_path))
nwa_files = sorted(glob(nwa_files_path))

n_files = len(wa_files) + len(nwa_files)
print(n_files)

size_image = 64

allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0
for f in wa_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue

for f in nwa_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue

###################################
# Prepare train & test samples
###################################

# test-train split
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# encode the Ys
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)

###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

# Input is a 64x64 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Max pooling layer
network = max_pool_2d(conv_2, 2)
# 5: Convolution layer with 64 filters
conv_3 = conv_2d(network, 64, 3, activation='relu', name='conv_3')

# 6: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 7: Fully-connected 512 node layer
network = fully_connected(network, 256, activation='relu')

# 8: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 9: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")

adam=tflearn.optimizers.Adam (learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')

network = regression(network, optimizer=adam,
                     loss='categorical_crossentropy',
                     metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model.tflearn/', max_checkpoints=3,
                    tensorboard_verbose=3, tensorboard_dir='tmp/tflearn_logs/')
###################################
# Train model for 100 epochs
###################################
model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=90,
          n_epoch=10, run_id='model', show_metric=True)

model.save('model.tflearn')
