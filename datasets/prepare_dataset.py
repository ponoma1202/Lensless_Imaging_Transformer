import os
import numpy as np
import random
from natsort import natsorted

save_path = "/home/ponoma/workspace/Pan_Transformer/datasets/"  
# root_dir = "/home/lakabuli/cosmos_drive/dataset100k"        
data_dir = "/home/lakabuli/cosmos_drive/dataset100k/rml" #os.path.join(root_dir, "rml")  
target_dir = "/home/clara/4tb_patrick/dataset100k/4x_warped_ground_truth"

full_data_list_data = os.listdir(data_dir)
sorted_data_list_data = natsorted(full_data_list_data) # Sort the lists in numerical order

full_data_list_target = os.listdir(target_dir)
sorted_data_list_target = natsorted(full_data_list_target)

# need two separate lists because they use different file naming conventions
train_set = sorted_data_list_data[5000:int(len(sorted_data_list_data) * 0.5)]      # just want to use 50k images for training
val_set = sorted_data_list_data[1000:5000]

train_set_target = sorted_data_list_target[5000:50000]      # just want to use 50k images for training
val_set_target = sorted_data_list_target[1000:5000]

train_imgs_lensless = []
train_imgs_ground = []
val_imgs_lensless = []
val_imgs_ground = []

# 25k images (4% of data for testing)
for i in range(len(train_set)):
    train_imgs_lensless.append(os.path.join(data_dir, train_set[i]))       # first image is missing in Mirflickr dataset
    train_imgs_ground.append(os.path.join(target_dir, train_set_target[i]))

for i in range(len(val_set)):
    val_imgs_lensless.append(os.path.join(data_dir, val_set[i]))      
    val_imgs_ground.append(os.path.join(target_dir, val_set_target[i]))

np.save(save_path+'train_patterns.npy',train_imgs_lensless)     
np.save(save_path+'train_targets.npy',train_imgs_ground)
np.save(save_path+'val_patterns.npy',val_imgs_lensless)
np.save(save_path+'val_targets.npy',val_imgs_ground)
