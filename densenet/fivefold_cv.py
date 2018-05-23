import h5py, time, os, glob, csv, datetime
import numpy as np
import matplotlib.pyplot as plt

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

##############################################################################################################
##############################################################################################################
def load_DN_model():
   base_model   = DenseNet(depth=10, weights=None, growth_rate = 12, dropout_rate=0.4, nb_dense_block=3, classes=64) # Default GR = 12
   output_layer = base_model.output
   predictions  = Dense(nb_classes, activation='sigmoid', name='meta_preds')(output_layer)  
   model        = Model(inputs = base_model.input, outputs = predictions)

   adam         = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

   return model

def test_DN_model(hdf5_name, test_images, test_labels, roc_name):
   path		  = '/home/lnugraha/Documents/050118_ensemble_learning/_weights/' 
   model_name     = path + hdf5_name
   model 	  = load_model(model_name)
   savefile       = roc_name

   length	  = len(test_images)
   preds	  = model.predict(test_images)

   label_values   = [] # holds all true values
   predict_values = [] # holds all prediction values (for confusion matrix)
   test_auc       = [] # holds all prediction values (not rounded for AUC)
 
   # FORMAT: [control][metastasis]
   for i in range(length):
      feature = preds[i]
      mets_score = feature[1]     # The predicted metastasis case 
      mets_true  = test_labels[i] # The true metastasis case [1 or 0] ONLY

      if (mets_true == 1.0):	  # TRUE Metastasis ONLY
         true_ctrl = 0.0
         true_mets = 1.0
      elif (mets_true != 1.0):
         true_ctrl = 1.0
         true_mets = 0.0
      label_values.append(true_mets)

      if (feature[0] > 0.5): 	# Remember the Precision X Recall Lecture
         pred_ctrl = 1.0	# Lower threshold for Non-Metastasis Case??? 
         pred_mets = 0.0
      else:
         pred_ctrl = 0.0
         pred_mets = 1.0
      predict_values.append(pred_mets)

   ############################################################################
   ### Note on Label META (test_label): 0 means control, 1 means metastasis ###
   f = open(savefile,'w');
   f.write('\t Pred META \t Lbl Prd META  \t Label META \n');
   for i in range(length):
      feature = preds[i]
      orig_mets_label = test_labels[i]
      f.write('\t {} \t {} \t {} \n'.format(feature[1], predict_values[i], orig_mets_label[0]))
   f.write('* \n') # Follows ROCKIT input format
   for i in range(length):
      feature = preds[i]
      f.write('\t {} \n'.format(1-feature[1])) # CTRL prediction score
   f.write('* \n') # Follows ROCKIT input format
   f.close()

   ###################################################
   ########### DEFINE THE CONFUSION MATRIX ###########
   cm = confusion_matrix(label_values, predict_values)
   class_name = ['control','metastasis'] # sub-folders are ctrl and mets
   n_classes = len(class_name)

#   plt.figure()
   plot_confusion_matrix(cm, classes = class_name, title='Confusion Matrix') # Spit the CM onto the screen
#   plt.show()

##############################################################################################################
################################################   MAIN CODE   ###############################################
##############################################################################################################
today         = datetime.datetime.now()
today_date    = (str(today.month)+"_"+str(today.day)+"_"+str(today.year))
file = h5py.File('/home/lnugraha/Documents/050118_ensemble_learning/_inputs/tumor_peritumor_complete_64.hdf5','r')
# file = h5py.File('/home/lnugraha/Documents/050118_ensemble_learning/_inputs/tumor_peritumor_complete_64_remake.hdf5','r')
Test_set_rate = 0.252					# (0.3197) test = 94, train & valid = 200
							# (0.1837) test = 54, train & valid = 240
							# (0.2520) test = 74, train & valid = 220 
nb_epochs  = 200; nb_batches = 22; nb_classes = 2; 	# Default Batch Value: 20

Tumor_Only  = file['All Data'][:]			# Tumor Only
All_Label   = file['All Label'][:]			# Label for ctrl [0] OR mets [1]
Peri_10     = file['Peritumor 10'][:]			# Peritumor Only 10 px
Peri_20     = file['Peritumor 20'][:]			# Peritumor Only 20 px
Peri_30     = file['Peritumor 30'][:]			# Peritumor Only 30 px

file.close()

# ............................... Shuffle all data along with their labels ................................. #
shuffled_Data,  shuffled_Label , shuffled_Peri10, shuffled_Peri20, shuffled_Peri30 = shuffle(Tumor_Only, All_Label, Peri_10, Peri_20, Peri_30)
# .......................................................................................................... #

total_num  = Tumor_Only.shape[0]
test_num   = int(total_num  * Test_set_rate)
train_num  = total_num - test_num 

print("Total: {}, Test: {}, Train: {} ".format(total_num, test_num, train_num))


##############     TRAIN IMAGES     ##############
train_both_10 = shuffled_Peri10 + shuffled_Data 	# Peritumor 10 px + Tumor ONLY
train_both_20 = shuffled_Peri20 + shuffled_Data 	# Peritumor 20 px + Tumor ONLY
train_both_30 = shuffled_Peri30 + shuffled_Data 	# Peritumor 30 px + Tumor ONLY

train_tumor   = shuffled_Data[0:train_num-1]		# Tumor ONLY 

train_img_30  = train_both_30[0:train_num-1]
train_img_20  = train_both_20[0:train_num-1]
train_img_10  = train_both_10[0:train_num-1]

train_peri_30 = shuffled_Peri30[0:train_num-1]		# Peritumor 30 px
train_peri_20 = shuffled_Peri20[0:train_num-1]		# Peritumor 20 px
train_peri_10 = shuffled_Peri10[0:train_num-1]		# Peritumor 10 px

train_labels  = shuffled_Label[0:train_num-1]		# TRAIN Labels

###############     TEST IMAGES     ###############
test_tumor    = shuffled_Data[train_num-1:-1]

test_img_30   = train_both_30[train_num-1:-1]
test_img_20   = train_both_20[train_num-1:-1]
test_img_10   = train_both_10[train_num-1:-1]

test_peri_30  = shuffled_Peri30[train_num-1:-1]
test_peri_20  = shuffled_Peri20[train_num-1:-1]
test_peri_10  = shuffled_Peri10[train_num-1:-1]

test_labels   = shuffled_Label[train_num-1:-1]		# TEST Labels

old_train_labels = train_labels
# For k-fold cross validation, do not use the below label version
train_labels  = to_categorical(train_labels, num_classes = nb_classes) # Dimension: (200 X 2). For example: [0 1]

train_datagen =  ImageDataGenerator(
      preprocessing_function=None, # rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
#      validation_split = 0.4
)
#########################################################################
########## 7 CASES: TUMOR, BOTH 10,20,30 px & PERI 10,20,30 px ##########
#########################################################################

			#########################################################################
			########## TUMOR ONLY With K-Fold Cross Validation		#########
			#########################################################################
# t       = time.time();
train_datagen.fit(train_tumor)
skfolds = list(StratifiedKFold(n_splits=5,shuffle=True).split(train_tumor, old_train_labels))

for j, (train_idx, val_idx) in enumerate(skfolds):
   print("\nFOLD: ", j)
   train_images_cv = train_tumor[train_idx]
   train_labels_cv = train_labels[train_idx]

   valid_images_cv = train_tumor[val_idx]
   valid_labels_cv = train_labels[val_idx]

   ##### Declare Model Name, Callbacks
   model        = load_DN_model();
   weight_path  = "/home/lnugraha/Documents/050118_ensemble_learning/_weights/"
   model_name   = "DN10_5CV_{}_BEST_TUMOR_{}.hdf5".format(j, today_date)
   checkpointer = ModelCheckpoint(filepath= weight_path+model_name, verbose=0, save_best_only=True)
   ##############################################################################################

   ##### DATA FLOW
   train_generator = train_datagen.flow(train_images_cv, train_labels_cv, batch_size = nb_batches)

   ##### Obtain Model Design and fit_generator
   training = model.fit_generator(
       train_generator,
       epochs = nb_epochs,
       validation_data = (train_images_cv, train_labels_cv),
       class_weight='auto', callbacks=[checkpointer], shuffle=True
   )

   model.save("/home/lnugraha/Documents/050118_ensemble_learning/_weights/DN10_5CV_{}_COMPLETE_TUMOR_{}.hdf5".format(j, today_date))
   print("TUMOR ONLY {} Fold: \n".format(j));
   test_DN_model("DN10_5CV_{}_BEST_TUMOR_{}.hdf5".format(j,today_date), test_tumor, test_labels, 'tumor_cv_{}_fold.csv'.format(j))


			#########################################################################
			########## BOTH 30 PX With K-Fold Cross Validation		#########
			#########################################################################

train_datagen.fit(train_img_30); # Include Data Augmentation
skfolds = list(StratifiedKFold(n_splits=5,shuffle=True).split(train_img_30, old_train_labels))

for j, (train_idx, val_idx) in enumerate(skfolds):
   print("\nFOLD: ", j)
   train_images_cv = train_img_30[train_idx]
   train_labels_cv = train_labels[train_idx]

   valid_images_cv = train_img_30[val_idx]
   valid_labels_cv = train_labels[val_idx]

   ##### Declare Model Name, Callbacks
   model        = load_DN_model();

   weight_path  = "/home/lnugraha/Documents/050118_ensemble_learning/_weights/"
   model_name   = "DN10_5CV_{}_BEST_BOTH_30_{}.hdf5".format(j, today_date)
   checkpointer = ModelCheckpoint(filepath= weight_path+model_name, verbose=0, save_best_only=True) 		# save best model  
   ##############################################################################################

   ##### DATA FLOW
   train_generator = train_datagen.flow(train_images_cv, train_labels_cv, batch_size = nb_batches)

   ##### Obtain Model Design and fit_generator
   training = model.fit_generator(
       train_generator,
       epochs = nb_epochs,
       validation_data = (train_images_cv, train_labels_cv),
       class_weight='auto', callbacks=[checkpointer], shuffle=True
   )

   model.save("/home/lnugraha/Documents/050118_ensemble_learning/_weights/DN10_5CV_{}_COMPLETE_BOTH_30_{}.hdf5".format(j, today_date))
   print("BOTH 30 PX {} Fold: \n".format(j));
   test_DN_model("DN10_5CV_{}_BEST_BOTH_30_{}.hdf5".format(j,today_date), test_img_30, test_labels, "both_30_cv_{}_fold.csv".format(j))

			#########################################################################
			########## BOTH 20 PX With K-Fold Cross Validation		#########
			#########################################################################

train_datagen.fit(train_img_20);
skfolds = list(StratifiedKFold(n_splits=5,shuffle=True).split(train_img_20, old_train_labels))

for j, (train_idx, val_idx) in enumerate(skfolds):
   print("\nFOLD: ", j)
   train_images_cv = train_img_20[train_idx]
   train_labels_cv = train_labels[train_idx]

   valid_images_cv = train_img_20[val_idx]
   valid_labels_cv = train_labels[val_idx]

   model        = load_DN_model();

   weight_path  = "/home/lnugraha/Documents/050118_ensemble_learning/_weights/"
   model_name   = "DN10_5CV_{}_BEST_BOTH_20_{}.hdf5".format(j, today_date)
   checkpointer = ModelCheckpoint(filepath= weight_path+model_name, verbose=0, save_best_only=True)
   ##############################################################################################

   train_generator = train_datagen.flow(train_images_cv, train_labels_cv, batch_size = nb_batches)

   ##### Obtain Model Design and fit_generator
   training = model.fit_generator(
       train_generator,
       epochs = nb_epochs,
       validation_data = (train_images_cv, train_labels_cv),
       class_weight='auto', callbacks=[checkpointer], shuffle=True
   )

   model.save("/home/lnugraha/Documents/050118_ensemble_learning/_weights/DN10_5CV_{}_COMPLETE_BOTH_20_{}.hdf5".format(j, today_date))
   print("BOTH 20 PX {} Fold: \n".format(j));
   test_DN_model("DN10_5CV_{}_BEST_BOTH_20_{}.hdf5".format(j,today_date), test_img_20, test_labels, "both_20_cv_{}_fold.csv".format(j))

			#########################################################################
			########## BOTH 10 PX With K-Fold Cross Validation		#########
			#########################################################################

train_datagen.fit(train_img_10);
skfolds = list(StratifiedKFold(n_splits=5,shuffle=True).split(train_img_10, old_train_labels))

for j, (train_idx, val_idx) in enumerate(skfolds):
   print("\nFOLD: ", j)
   train_images_cv = train_img_10[train_idx]
   train_labels_cv = train_labels[train_idx]

   valid_images_cv = train_img_10[val_idx]
   valid_labels_cv = train_labels[val_idx]

   model        = load_DN_model();

   weight_path  = "/home/lnugraha/Documents/050118_ensemble_learning/_weights/"
   model_name   = "DN10_5CV_{}_BEST_BOTH_10_{}.hdf5".format(j, today_date)
   checkpointer = ModelCheckpoint(filepath= weight_path+model_name, verbose=0, save_best_only=True)
   ##############################################################################################

   train_generator = train_datagen.flow(train_images_cv, train_labels_cv, batch_size = nb_batches)

   ##### Obtain Model Design and fit_generator
   training = model.fit_generator(
       train_generator,
       epochs = nb_epochs,
       validation_data = (train_images_cv, train_labels_cv),
       class_weight='auto', callbacks=[checkpointer], shuffle=True
   )

   model.save("/home/lnugraha/Documents/050118_ensemble_learning/_weights/DN10_5CV_{}_COMPLETE_BOTH_10_{}.hdf5".format(j, today_date))
   print("BOTH 10 PX {} Fold: \n".format(j));
   test_DN_model("DN10_5CV_{}_BEST_BOTH_10_{}.hdf5".format(j,today_date), test_img_10, test_labels, "both_10_cv_{}_fold.csv".format(j))

			#########################################################################
			########## PERI 30 PX With K-Fold Cross Validation		#########
			#########################################################################

train_datagen.fit(train_peri_30);
skfolds = list(StratifiedKFold(n_splits=5,shuffle=True).split(train_peri_30, old_train_labels))

for j, (train_idx, val_idx) in enumerate(skfolds):
   print("\nFOLD: ", j)
   train_images_cv = train_peri_30[train_idx]
   train_labels_cv = train_labels[train_idx]

   valid_images_cv = train_peri_30[val_idx]
   valid_labels_cv = train_labels[val_idx]

   ##### Declare Model Name, Callbacks
   model        = load_DN_model();

   weight_path  = "/home/lnugraha/Documents/050118_ensemble_learning/_weights/"
   model_name   = "DN10_5CV_{}_BEST_PERI_30_{}.hdf5".format(j, today_date)
   checkpointer = ModelCheckpoint(filepath= weight_path+model_name, verbose=0, save_best_only=True)  
   ##############################################################################################

   ##### DATA FLOW
   train_generator = train_datagen.flow(train_images_cv, train_labels_cv, batch_size = nb_batches)

   ##### Obtain Model Design and fit_generator
   training = model.fit_generator(
       train_generator,
       epochs = nb_epochs,
       validation_data = (train_images_cv, train_labels_cv),
       class_weight='auto', callbacks=[checkpointer], shuffle=True
   )

   model.save("/home/lnugraha/Documents/050118_ensemble_learning/_weights/DN10_5CV_{}_COMPLETE_PERI_30_{}.hdf5".format(j, today_date))
   print("PERI 30 PX {} Fold: \n".format(j));
   test_DN_model("DN10_5CV_{}_BEST_PERI_30_{}.hdf5".format(j,today_date), test_peri_30, test_labels, "peri_30_cv_{}_fold.csv".format(j))

			#########################################################################
			########## PERI 20 PX With K-Fold Cross Validation		#########
			#########################################################################

train_datagen.fit(train_peri_20);
skfolds = list(StratifiedKFold(n_splits=5,shuffle=True).split(train_peri_20, old_train_labels))

for j, (train_idx, val_idx) in enumerate(skfolds):
   print("\nFOLD: ", j)
   train_images_cv = train_peri_20[train_idx]
   train_labels_cv = train_labels[train_idx]

   valid_images_cv = train_peri_20[val_idx]
   valid_labels_cv = train_labels[val_idx]

   ##### Declare Model Name, Callbacks
   model        = load_DN_model();

   weight_path  = "/home/lnugraha/Documents/050118_ensemble_learning/_weights/"
   model_name   = "DN10_5CV_{}_BEST_PERI_20_{}.hdf5".format(j, today_date)
   checkpointer = ModelCheckpoint(filepath= weight_path+model_name, verbose=0, save_best_only=True) 		# save best model  
   ##############################################################################################
   train_generator = train_datagen.flow(train_images_cv, train_labels_cv, batch_size = nb_batches)
   ##############################################################################################
   ##### Obtain Model Design and fit_generator
   training = model.fit_generator(
       train_generator,
       epochs = nb_epochs,
       validation_data = (train_images_cv, train_labels_cv),
       class_weight='auto', callbacks=[checkpointer], shuffle=True
   )

   model.save("/home/lnugraha/Documents/050118_ensemble_learning/_weights/DN10_5CV_{}_COMPLETE_PERI_20_{}.hdf5".format(j, today_date))
   print("PERI 20 PX {} Fold: \n".format(j));
   test_DN_model("DN10_5CV_{}_BEST_PERI_20_{}.hdf5".format(j,today_date), test_peri_20, test_labels, "peri_20_cv_{}_fold.csv".format(j))

			#########################################################################
			########## PERI 10 PX With K-Fold Cross Validation		#########
			#########################################################################

train_datagen.fit(train_peri_10);
skfolds = list(StratifiedKFold(n_splits=5,shuffle=True).split(train_peri_10, old_train_labels))

for j, (train_idx, val_idx) in enumerate(skfolds):
   print("\nFOLD: ", j)
   train_images_cv = train_peri_10[train_idx]
   train_labels_cv = train_labels[train_idx]

   valid_images_cv = train_peri_10[val_idx]
   valid_labels_cv = train_labels[val_idx]

   ##### Declare Model Name, Callbacks
   model        = load_DN_model();

   weight_path  = "/home/lnugraha/Documents/050118_ensemble_learning/_weights/"
   model_name   = "DN10_5CV_{}_BEST_PERI_10_{}.hdf5".format(j, today_date)
   checkpointer = ModelCheckpoint(filepath= weight_path+model_name, verbose=0, save_best_only=True) 		# save best model  
   ##############################################################################################
   train_generator = train_datagen.flow(train_images_cv, train_labels_cv, batch_size = nb_batches)
   ##############################################################################################
   ##### Obtain Model Design and fit_generator
   training = model.fit_generator(
       train_generator,
       epochs = nb_epochs,
       validation_data = (train_images_cv, train_labels_cv),
       class_weight='auto', callbacks=[checkpointer], shuffle=True
   )

   model.save("/home/lnugraha/Documents/050118_ensemble_learning/_weights/DN10_5CV_{}_COMPLETE_PERI_10_{}.hdf5".format(j, today_date))
   print("PERI 10 PX {} Fold: \n".format(j));
   test_DN_model("DN10_5CV_{}_BEST_PERI_10_{}.hdf5".format(j,today_date), test_peri_10, test_labels, "peri_10_cv_{}_fold.csv".format(j))
