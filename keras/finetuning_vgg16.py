###################################################################################################################
# How to run it:                                                                                                  #
# python fine-tune-vgg16.py --train_dir ... -val_dir ... -nb_epoch ... --batch_size ... --output_model ... --plot #
# train_dir_extended and val_dir_extended									  #
# Created: February 08, 2018 											  #
###################################################################################################################

import os
import sys
import glob
import time
import argparse
import matplotlib.pyplot as plt
import pylab

from keras import __version__
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

IM_WIDTH, IM_HEIGHT = 224, 224 #fixed size for InceptionV3
NB_EPOCHS = 50
BAT_SIZE = 25 #32 Try a small number of epochs (less than 12)
              # BAT_SIZE/NB_EPOCHS

FC_SIZE = 1024
# NB_VGG16_LAYERS_TO_FREEZE = 172

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

# Convert PNG format images to JPEG format images
def png_to_jpeg(self, image_data):
  return self._sess.run(self._png_to_jpeg,feed_dict={self._png_data: image_data})
# Convert BMP format images to JPEG format images
def bmp_to_jpeg(self, image_data):
  return self._sess.run(self._bmp_to_jpeg,feed_dict={self._png_data: image_data})

# Proceed to Transfer Learning Mode
#def setup_to_transfer_learn(model, base_model):
#  """Freeze all layers and compile the model"""
#  for layer in base_model.layers:
#    layer.trainable = False
#  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add a new last layer to the Convolution Network
def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet

  Args:
    base_model: keras model excluding top
    nb_classes: # of classes

  Returns:
    new keras model with last layer
  """
  x = base_model.output
#  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model

def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
# All layers are now trainable
#  for layer in model.layers[:NB_VGG16_LAYERS_TO_FREEZE]:
#     layer.trainable = False

# Select layers to be trainable
  for layer in model.layers[1:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.8), loss='categorical_crossentropy', metrics=['accuracy'])
# Version Original: lr=0.0001

# TRAINING SESSION OF OUR PRE-TRAINED NEURAL NETWORK
def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = get_nb_files(args.train_dir)
  nb_classes = len(glob.glob(args.train_dir + "/*"))
  nb_val_samples = get_nb_files(args.val_dir)
  nb_epoch = int(args.nb_epoch)
  batch_size = int(args.batch_size)

  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=None,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True
  )
  validation_datagen = ImageDataGenerator(
      preprocessing_function=None,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True
  )

  train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  validation_generator = validation_datagen.flow_from_directory(
    args.val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  # setup model
  base_model = VGG16(weights='imagenet') #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)

#  # TRANSFER LEARNING SETUP
#  setup_to_transfer_learn(model, base_model)

#  history_tl = model.fit_generator(
#    train_generator,
#    nb_epoch=nb_epoch,
#    samples_per_epoch=nb_train_samples,
#    validation_data=validation_generator,
#    nb_val_samples=nb_val_samples,
#    class_weight='auto') #class_weight='none' or 'auto'

  # FINE TUNING SETUP
  setup_to_finetune(model)

  history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')

  model.save(args.output_model_file)

  if args.plot:
    plot_training(history_ft)


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  pylab.plot(epochs, acc, 'bs-', label='Train Acc') # Training Loss
  pylab.plot(epochs, val_acc, 'r*-', label='Val Acc') # Validation Loss
  pylab.xlabel('Epoch')
  pylab.ylabel('Accuracy [%]')
  pylab.title('Training and Validation Accuracy')
  pylab.legend(loc='upper right')

  pylab.figure()

  pylab.plot(epochs, loss, 'bs-', label='Train Loss') # Training Loss
  pylab.plot(epochs, val_loss, 'r*-', label='Val Loss') # Validation Loss
  pylab.xlabel('Epoch')
  pylab.ylabel('Loss [%]')
  pylab.title('Training and Validation Loss')
  pylab.legend(loc='upper right')

  pylab.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--train_dir")
  a.add_argument("--val_dir")
  a.add_argument("--nb_epoch", default=NB_EPOCHS)
  a.add_argument("--batch_size", default=BAT_SIZE)
  a.add_argument("--output_model_file", default="inceptionv3-ft.model")
  a.add_argument("--plot", action="store_true")

  args = a.parse_args()
  if args.train_dir is None or args.val_dir is None:
    a.print_help()
    sys.exit(1)

  if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
    print("Directories Not Found!")
    sys.exit(1)

# Add Timer:
  t = time.time();
  train(args)
  print("Training Time: ", time.time()-t);
