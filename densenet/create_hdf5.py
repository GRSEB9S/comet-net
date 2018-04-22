input_image_size = 64

import h5py, os
from os import listdir
from skimage import io
from skimage.transform import resize
import numpy as np

tumor_path        = '/home/lnugraha/Documents/classic_texture_images_png/'

peritumor_10_path = '/home/lnugraha/Documents/10_px_peritumor/'
peritumor_20_path = '/home/lnugraha/Documents/20_px_peritumor/'
peritumor_30_path = '/home/lnugraha/Documents/30_px_peritumor/'

Class0 = 'ctrl/'
Class1 = 'mets/'

Class0_files_name   = listdir(tumor_path + Class0) 	  # ctrl tumor data
Class1_files_name   = listdir(tumor_path + Class1) 	  # mets tumor data

Class0_peritumor_10 = listdir(peritumor_10_path + Class0) # ctrl peritumor 10 data
Class1_peritumor_10 = listdir(peritumor_10_path + Class1) # mets peritumor 10 data

Class0_peritumor_20 = listdir(peritumor_20_path + Class0) # ctrl peritumor 20 data
Class1_peritumor_20 = listdir(peritumor_20_path + Class1) # mets peritumor 20 data

Class0_peritumor_30 = listdir(peritumor_30_path + Class0) # ctrl peritumor 30 data
Class1_peritumor_30 = listdir(peritumor_30_path + Class1) # mets peritumor 30 data

####################### Create Zero Arrays #######################################      # CTRL case
label0 = np.zeros((len(Class0_files_name),1))						# hold all labels

data0  = np.zeros((len(Class0_files_name),input_image_size,input_image_size,3))		# hold all tumor data 

peri10_0 = np.zeros((len(Class0_peritumor_10),input_image_size,input_image_size,3))	# hold peritumor10 data 
peri20_0 = np.zeros((len(Class0_peritumor_20),input_image_size,input_image_size,3))	# hold peritumor20 data 
peri30_0 = np.zeros((len(Class0_peritumor_30),input_image_size,input_image_size,3))	# hold peritumor30 data 

for i in range(len(Class0_files_name)): 						# Load Tumor ONLY ctrl case
    filename = os.path.join(tumor_path + Class0 + Class0_files_name[i])			# /home/lnugraha/Documents/classic_texture_images/ + /ctrl/ + /ctrl.1.jpg/
    camera0  = io.imread(filename)							# Load it
    camera0  = resize(camera0 ,(input_image_size,input_image_size,3), mode='reflect',preserve_range = True)
    data0[i] = camera0

for i in range(len(Class0_peritumor_10)): 						# Load PERITUMOR 10 ctrl case
    peri10_filename = os.path.join(peritumor_10_path + Class0 + Class0_files_name[i])	# /home/lnugraha/Documents/.../ + /ctrl/ + /ctrl.1.jpg/
    peri10_camera0  = io.imread(peri10_filename)					# Load it
    peri10_camera0  = resize(peri10_camera0 ,(input_image_size,input_image_size,3), mode='reflect',preserve_range = True)
    peri10_0[i]     = peri10_camera0

for i in range(len(Class0_peritumor_20)): 						# Load PERITUMOR 10 ctrl case
    peri20_filename = os.path.join(peritumor_20_path + Class0 + Class0_files_name[i])	# /home/lnugraha/Documents/.../ + /ctrl/ + /ctrl.1.jpg/
    peri20_camera0  = io.imread(peri20_filename)					# Load it
    peri20_camera0  = resize(peri20_camera0 ,(input_image_size,input_image_size,3), mode='reflect',preserve_range = True)
    peri20_0[i]     = peri20_camera0

for i in range(len(Class0_peritumor_30)): 						# Load PERITUMOR 10 ctrl case
    peri30_filename = os.path.join(peritumor_30_path + Class0 + Class0_files_name[i])	# /home/lnugraha/Documents/.../ + /ctrl/ + /ctrl.1.jpg/
    peri30_camera0  = io.imread(peri30_filename)					# Load it
    peri30_camera0  = resize(peri30_camera0 ,(input_image_size,input_image_size,3), mode='reflect',preserve_range = True)
    peri30_0[i]     = peri30_camera0

###################################################################################

label1   = np.ones((len(Class1_files_name),1))						# METS case

data1    = np.zeros((len(Class1_files_name),input_image_size,input_image_size,3))

peri10_1 = np.zeros((len(Class1_peritumor_10),input_image_size,input_image_size,3))	# hold peritumor10 data 
peri20_1 = np.zeros((len(Class1_peritumor_20),input_image_size,input_image_size,3))	# hold peritumor20 data 
peri30_1 = np.zeros((len(Class1_peritumor_30),input_image_size,input_image_size,3))	# hold peritumor30 data 


for i in range(len(Class1_files_name)):
    filename = os.path.join(tumor_path+Class1+Class1_files_name[i])
    camera1 = io.imread(filename)
    camera1 = resize(camera1 ,(input_image_size,input_image_size,3), mode='reflect',preserve_range = True)
    data1[i] = camera1

for i in range(len(Class1_peritumor_10)): 						# Load PERITUMOR 10 ctrl case
    peri10_filename = os.path.join(peritumor_10_path + Class1 + Class1_files_name[i])	# /home/lnugraha/Documents/.../ + /ctrl/ + /ctrl.1.jpg/
    peri10_camera1  = io.imread(peri10_filename)					# Load it
    peri10_camera1  = resize(peri10_camera1 ,(input_image_size,input_image_size,3), mode='reflect',preserve_range = True)
    peri10_1[i]     = peri10_camera1

for i in range(len(Class1_peritumor_20)): 						# Load PERITUMOR 10 ctrl case
    peri20_filename = os.path.join(peritumor_20_path + Class1 + Class1_files_name[i])	# /home/lnugraha/Documents/.../ + /ctrl/ + /ctrl.1.jpg/
    peri20_camera1  = io.imread(peri20_filename)					# Load it
    peri20_camera1  = resize(peri20_camera1 ,(input_image_size,input_image_size,3), mode='reflect',preserve_range = True)
    peri20_1[i]     = peri20_camera1

for i in range(len(Class1_peritumor_30)): 						# Load PERITUMOR 10 ctrl case
    peri30_filename = os.path.join(peritumor_30_path + Class1 + Class1_files_name[i])	# /home/lnugraha/Documents/.../ + /ctrl/ + /ctrl.1.jpg/
    peri30_camera1  = io.imread(peri30_filename)					# Load it
    peri30_camera1  = resize(peri30_camera1 ,(input_image_size,input_image_size,3), mode='reflect',preserve_range = True)
    peri30_1[i]     = peri30_camera1

#%%
Total_data   = np.zeros(((len(Class0_files_name)+len(Class1_files_name)),input_image_size,input_image_size,3))
Total_labe   = np.zeros(((len(Class0_files_name)+len(Class1_files_name)),1))

Total_peri10 = np.zeros(((len(Class0_peritumor_10)+len(Class1_peritumor_10)),input_image_size,input_image_size,3))
Total_peri20 = np.zeros(((len(Class0_peritumor_20)+len(Class1_peritumor_20)),input_image_size,input_image_size,3))
Total_peri30 = np.zeros(((len(Class0_peritumor_30)+len(Class1_peritumor_30)),input_image_size,input_image_size,3))

# STACK DATA VERTICALLY
Total_data   = np.vstack((data0   , data1))		# Tumor Data (64 width, 64 length, 294 height)
Total_labe   = np.vstack((label0  , label1))		# Label Data 
Total_peri10 = np.vstack((peri10_0, peri10_1))		# Peritumor 10
Total_peri20 = np.vstack((peri20_0, peri20_1))		# Peritumor 20
Total_peri30 = np.vstack((peri30_0, peri30_1))		# Peritumor 30
#%%

# Save Your hdf5 File
file = h5py.File("/home/lnugraha/Documents/041618_data_collection/_inputs/tumor_peritumor_complete_64.hdf5", "w")
file.create_dataset('All Data',     data = Total_data)
file.create_dataset('All Label',    data = Total_labe)
file.create_dataset('Peritumor 10', data = Total_peri10)
file.create_dataset('Peritumor 20', data = Total_peri20)
file.create_dataset('Peritumor 30', data = Total_peri30)
