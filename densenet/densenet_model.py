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
from sklearn.model_selection import train_test_split, GroupKFold

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from plot_confusion_matrix import plot_confusion_matrix

from densenet import DenseNet

# from utils import load_data, shuffle_data, save_data, create_class_weight
# file = h5py.File('/home/lnugraha/Documents/041618_data_collection/_inputs/tumor_64.hdf5','r')
# file = h5py.File('/home/lnugraha/Documents/041618_data_collection/_inputs/30_px_both_64.hdf5','r')
file = h5py.File('/home/lnugraha/Documents/041618_data_collection/_inputs/30_px_peritumor_64.hdf5','r')
Test_set_rate = 0.32 # (0.32) test = 94, train+valid = 200
nb_epochs  = 200; nb_batches = 20; nb_classes = 2; # Default Batch Value: 20

All_Data  = file['All Data'][:]
All_Label = file['All Label'][:]

# .........
file.close()
shuffled_Data,  shuffled_Label = shuffle(All_Data,All_Label)
# .........

total_num  = All_Data.shape[0]
test_num   = int(total_num  * Test_set_rate)
train_num  = total_num - test_num 

print("Total: {}, Test: {}, Train: {} ".format(total_num, test_num, train_num))

train_images  = shuffled_Data[0:train_num-1]
train_labels  = shuffled_Label[0:train_num-1]

test_images   = shuffled_Data[train_num-1:-1]
test_labels   = shuffled_Label[train_num-1:-1]

train_labels  = to_categorical(train_labels, num_classes = nb_classes)


train_datagen =  ImageDataGenerator(
      preprocessing_function=None,
#      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
#      vertical_flip=True,
      validation_split = 0.4
)

train_datagen.fit(train_images) # Include Data Augmentation
train_generator = train_datagen.flow(train_images, train_labels, batch_size = nb_batches)

### DENSENET MODEL
#base_model = DenseNet(depth=10, weights=None, dropout_rate=0.0,nb_dense_block=3,bottleneck=True)
base_model = DenseNet(depth=10, weights=None, dropout_rate=0.4, nb_dense_block=3)
x = base_model.output
# x = Dropout(0.2)(x)
predictions = Dense(nb_classes, activation='sigmoid', name='meta_preds')(x) 				# New softmax layer  
model = Model(input = base_model.input, output = predictions)
# model.summary() 											# Added by Mar 07, 2018

"""
for layer in base_model.layers:
    layer.trainable = False
"""

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
tensorboard  = TensorBoard(log_dir= "../logs/DenseNet_10_{}_04202018".format(time.time())) 		# TensorBoard

model_name = "DenseNet_10_BEST_PERITUMOR_04202018.hdf5"
checkpointer = ModelCheckpoint(filepath=model_name, verbose=0, save_best_only=True) 			# save best model  
print("Good to Go")

#########################################################################################################################################

# Start Time Recording
t = time.time();
training = model.fit_generator(
    train_generator,
#    steps_per_epoch = 10,
    epochs = nb_epochs,
    validation_data = (train_images, train_labels),
    class_weight='auto', callbacks=[checkpointer, tensorboard], shuffle=True
)
model.save("DenseNet_10_complete_PERITUMOR_04202018.hdf5")
# Add Timer:
print("Training Time: ", time.time()-t);

################################## TEST YOUR MODEL USING DATA SETS ###################################
model_path = '/home/lnugraha/Documents/041618_data_collection/'+model_name
model      = load_model(model_path)

length = len(test_images)
preds  = model.predict(test_images)

label_values   = [] # holds all true values
predict_values = [] # holds all prediction values (for confusion matrix)
test_auc       = [] # holds all prediction values (not rounded for AUC)

# FORMAT: [control][metastasis]
for i in range(length):
    feature = preds[i]
    mets_score = feature[1]
    mets_true  = test_labels[i]

    if (mets_true != 0):
       true_ctrl = 0.0
       true_mets = 1.0
    else:
       true_ctrl = 1.0
       true_mets = 0.0
    label_values.append(true_mets)

    if (feature[0] > 0.5): 	# Remember the Precision X Recall Lecture
       pred_ctrl = 1.0		# Lower threshold for Non-Metastasis Case??? 
       pred_mets = 0.0
    else:
       pred_ctrl = 0.0
       pred_mets = 1.0
    predict_values.append(pred_mets)

########### DEFINE THE CONFUSION MATRIX ###########
cm = confusion_matrix(label_values, predict_values)
class_name = ['control','metastasis'] # sub-folders are ctrl and mets
n_classes = len(class_name)

plt.figure()
plot_confusion_matrix(cm, classes = class_name, title='Confusion Matrix')
plt.show()
