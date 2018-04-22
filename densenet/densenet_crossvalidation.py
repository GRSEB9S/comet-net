import h5py, time, os, glob
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from keras import __version__
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Dropout, LeakyReLU, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from plot_confusion_matrix import plot_confusion_matrix

from densenet import DenseNet # Costumized DenseNet Model

file = h5py.File('/home/lnugraha/Documents/041618_data_collection/_inputs/30_px_peritumor_64.hdf5','r')
Test_set_rate = 0.3197					# (0.32) test = 94, train+valid = 200
nb_epochs  = 200; nb_batches = 20; nb_classes = 2; 	# Default Batch Value: 20

All_Data  = file['All Data'][:]
All_Label = file['All Label'][:]

file.close()

# ......... Shuffle all data along with their labels .......... #
shuffled_Data,  shuffled_Label = shuffle(All_Data,All_Label)
# ............................................................. #

total_num  = All_Data.shape[0]
test_num   = int(total_num  * Test_set_rate)
train_num  = total_num - test_num 

print("Total: {}, Test: {}, Train: {} ".format(total_num, test_num, train_num))

train_images  = shuffled_Data[0:train_num-1]
train_labels  = shuffled_Label[0:train_num-1]

########### RESERVED FOR TESTING PURPOSE ONLY ###########
test_images   = shuffled_Data[train_num-1:-1]		#
test_labels   = shuffled_Label[train_num-1:-1]		#
#########################################################

old_train_labels = train_labels
# For k-fold cross validation, do not use this label version
train_labels  = to_categorical(train_labels, num_classes = nb_classes) # Dimension: (200 X 2). For example: [0 1]

train_datagen =  ImageDataGenerator(
      preprocessing_function=None, # rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True, # vertical_flip=True,
      validation_split = 0.4
)

train_datagen.fit(train_images) # Include Data Augmentation
train_generator = train_datagen.flow(train_images, train_labels, batch_size = nb_batches)

# Using K-Fold Cross Validation
# kfold = KFold(5) # Five-Fold Cross Validation
skfolds = list(StratifiedKFold(n_splits=5,shuffle=True).split(train_images, old_train_labels))

# https://www.kaggle.com/stefanie04736/simple-keras-model-with-k-fold-cross-validation
for j, (train_idx, val_idx) in enumerate(skfolds):
   print("\n FOLD: ", j)
   train_images_cv = train_images[train_idx]
   train_labels_cv = train_labels[train_idx]

   valid_images_cv = train_images[val_idx]
   valid_labels_cv = train_labels[val_idx]

   # Declare Model Name, Callbacks
   # DATA FLOW
   # Obtain Model Design and fit_generator




