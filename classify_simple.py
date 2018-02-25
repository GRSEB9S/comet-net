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
import cv2
import os			# directories

from keras import __version__
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import load_model

from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

#model = VGG16(weights='imagenet')
model = load_model('comet_feb_22_ep_200_ba_20_vgg16_180.model')

image_path = '/home/lnugraha/Documents/test_dir/test_images'
os.chdir(image_path)

test_label   = [] # holds all true values
test_predict = [] # holds all prediction values

for x in range(1, 61): # How many test images you want to have?
   if (x%2 != 0):
      true_ctrl = 1.0
   else:
      true_ctrl = 0.0
   test_label.append(true_ctrl)

   number = str(x)
   img_path = '{}.jpeg'.format(number)   #format(str(x))
   img = image.load_img(img_path, target_size=(224, 224))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)

   preds = model.predict(x)
   feature = preds[0]
# feature has 2 dimensions: the first D tells in-site cancer, and the second element tells how close the image is to the metastasis 
# control = feature[0] AND  metastasis = feature[1]
   if (feature[0] > 0.5):
      pred_ctrl = 1.0
   else:
      pred_ctrl = 0.0

   test_predict.append(pred_ctrl)
#   print('Predicted: ', 100.0*preds[0])
   print('Predicted {}'.format(number),': ', feature)


# print(test_predict)
# print(test_label)
########### DEFINE THE CONFUSION MATRIX ###########
# define the test label
cm = confusion_matrix(test_label, test_predict)
class_name = ['control','metastasis']

plt.figure()
plot_confusion_matrix(cm, classes = class_name, title='Confusion Matrix')
plt.show()

# print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]
