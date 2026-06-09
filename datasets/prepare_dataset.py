import os
import numpy as np
import random
from natsort import natsorted

save_path = "/home/ponoma/workspace/Pan_Transformer/datasets/"  
root_dir = "/home/lakabuli/4tb_patrick/dataset100k26"        
dataset = 'rml'

if dataset == 'rml':
    data_dir = os.path.join(root_dir, "4x_rml")  
    target_dir = os.path.join(root_dir, "4x_undistorted_GT2RML")
elif dataset == "mirflickr":
    data_dir = "/home/ponoma/workspace/DATA/mirflickr_dataset/diffuser_images_npy"
    target_dir = "/home/ponoma/workspace/DATA/mirflickr_dataset/ground_truth_lensed_npy"

full_data_list_data = os.listdir(data_dir)
sorted_data_list_data = natsorted(full_data_list_data) # Sort the lists in numerical order

full_data_list_target = os.listdir(target_dir)
sorted_data_list_target = natsorted(full_data_list_target)

if dataset == 'mirflickr':
    train_set = sorted_data_list_data[1000:]      # just want to use 50k images for training
    val_set = sorted_data_list_data[:1000]

    train_set_target = sorted_data_list_target[1000:]      # make two separate lists if they use different file naming conventions
    val_set_target = sorted_data_list_target[:1000]
else:
    train_set = sorted_data_list_data[5000:50000]      # just want to use 50k images for training
    val_set = sorted_data_list_data[1000:5000]
    test_set = sorted_data_list_data[:1000]
    train_set_target = sorted_data_list_target[5000:50000]      # make two separate lists if they use different file naming conventions
    val_set_target = sorted_data_list_target[1000:5000]
    test_set_target = sorted_data_list_target[:1000]

train_imgs_lensless = []
train_imgs_ground = []
val_imgs_lensless = []
val_imgs_ground = []
test_imgs_lensless = []
test_imgs_ground = []
test_imgs_lensless = []
test_imgs_ground = []

# 25k images (4% of data for testing)
for i in range(len(train_set)):
    train_imgs_lensless.append(os.path.join(data_dir, train_set[i]))       # first image is missing in Mirflickr dataset
    train_imgs_ground.append(os.path.join(target_dir, train_set_target[i]))

for i in range(len(val_set)):
    val_imgs_lensless.append(os.path.join(data_dir, val_set[i]))      
    val_imgs_ground.append(os.path.join(target_dir, val_set_target[i]))

for i in range(len(test_set)):
    test_imgs_lensless.append(os.path.join(data_dir, test_set[i]))      
    test_imgs_ground.append(os.path.join(target_dir, test_set_target[i]))

np.save(save_path+'train_patterns.npy',train_imgs_lensless)     
np.save(save_path+'train_targets.npy',train_imgs_ground)
np.save(save_path+'val_patterns.npy',val_imgs_lensless)
np.save(save_path+'val_targets.npy',val_imgs_ground)
np.save(save_path+'test_patterns.npy',test_imgs_lensless)
np.save(save_path+'test_targets.npy',test_imgs_ground)