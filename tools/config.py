import os

# setting for the searchable path
curr_dir = os.getcwd()
curr_dir = curr_dir.replace('tasks', '')

TRAIN_DATASET_PATH = {'images': curr_dir + '/data/train/IMAGES.hdf5',
                      'labels': curr_dir + '/data/train/LABELS.hdf5'}

TEST_DATASET_PATH = {'images': curr_dir + '/data/test/IMAGES.hdf5',
                      'labels': curr_dir + '/data/test/LABELS.hdf5'}

# variables for data augmentations:
# elastic deform probability
e_deform_probability = 0.1
# gamma correction probability
gamma_probability = 0.1
# whether to rotate the data or not
rotate_angle = True

# training phase:
# show training graph
show_save_training_stats = True

# testing phase:
# save predictions
save_figs = True
# figure dpi
fig_dpi = 1200
# figure color scheme
cmap = 'gray'  # 'gray' or 'CMRmap'
# directory for saving predictions
parent_dir = curr_dir + '/predictions/'