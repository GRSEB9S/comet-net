# February 25, 2018
# Desc: This code loads the trained neural net (ctrl x mets),
# tests the prediction, and performs statistical analysis (conf matrix)

import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO

import matplotlib.pyplot as plt
import scikitplot as skplt

import cv2
import os			# directories

from keras import __version__
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import load_model

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from plot_confusion_matrix import plot_confusion_matrix

# CLASSIFIER #0: VGG16 FRESH
# model_path = './_models/comet_feb_22_ep_200_ba_20_vgg16_180.model' # 
# model_path = '../_models/Mar_02_2018/cometnet_mar_02_ep_200_ba_20_vgg16_180_BEST.hdf5' # highest sensitivity & specificity

# CLASSIFIER #1: ENSEMBLE LEARNING

# CLASSIFIER #2: VGG16 CANDY
#model_path = '../_models/Mar_01_2018/candlenet_ep_200_ba_30_180_BEST.hdf5' # highest sensitivity & specificity

# CLASSIFIER #3: CANDLENET
# model_path = '../_models/candlenet_mar_02_ep_200_ba_20_180_BEAST.hdf5'
model_path = '../_models/Mar_06_2018/candlenet_load_model_ep200_ba20.model'

model = load_model(model_path)
# model = load_model('cv-tricks_fine_tuned_model.h5') # crap

image_path = './raw_tests'
#image_path = './_tests_all'
os.chdir(image_path)

test_label   = [] # holds all true values
test_predict = [] # holds all prediction values (for confusion matrix)
test_auc     = [] # holds all prediction values (not rounded for AUC)

for x in range(1, 295): # How many test images you want to have?
   # Label Your Test Data:
   if (x%2 != 0):
      true_ctrl = 0.0
      true_mets = 1.0
   else:
      true_ctrl = 1.0
      true_mets = 0.0
   test_label.append(true_mets) # true_ctrl

   number = str(x)
   img_path = '{}.jpg'.format(number)   #format(str(x))
   
   ########################################################################
   # CANDYNET USES A SIZE OF 50 X 50, BUT VGG-16 USES A SIZE OF 224 X 224 #
   ######################################################################## 
   img = image.load_img(img_path, target_size=(50, 50)) # (224,224)
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)

   preds = model.predict(x)
   feature = preds[0]
# feature has 2 dimensions: the first D tells in-site cancer, and the second element tells how close the image is to the metastasis 
# control = feature[0] AND  metastasis = feature[1]
   if (feature[0] > 0.5):
      pred_ctrl = 1.0
      pred_mets = 0.0
   else:
      pred_ctrl = 0.0
      pred_mets = 1.0

   test_predict.append(pred_mets) # pred_ctrl
   test_auc.append(feature[1])
#   check if the code delivers expected results; i.e., no deviation
#   print('Predicted {}'.format(number),': ', feature)


# print(test_predict)
# print(test_label)
########### DEFINE THE CONFUSION MATRIX ###########
# define the test label
cm = confusion_matrix(test_label, test_predict)
class_name = ['control','metastasis'] # sub-folders are ctrl and mets
n_classes = len(class_name)
print(n_classes)

plt.figure()
plot_confusion_matrix(cm, classes = class_name, title='Confusion Matrix')
#roc_curve(test_label, test_predict)
plt.show()

#skplt.metrics.plot_roc_curve(test_label[:],test_predict[:])
#plt.show()

# area = roc_auc_score(test_label,test_predict)
# print("Area Under the Curve: ", area)

# print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
   fpr[i], tpr[i], _ = roc_curve(test_label, test_predict)
   roc_auc[i] = auc(fpr[i], tpr[i])

#plt.figure()
case = 1
plt.plot(fpr[case], tpr[case], color='darkorange', lw=case, label='ROC curve (area = %0.4f)' % roc_auc[case])
plt.plot([0, 1], [0, 1], color='navy', lw=case, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()
