import os
curr_dir = os.getcwd()
curr_dir = curr_dir.replace('tasks', '')

TRAIN_DATASET_PATH = {'images': curr_dir + '/data/train/IMAGES.hdf5',
                      'labels': curr_dir + '/data/train/LABELS.hdf5'}

# provide the file path for the weight file
def load_weight_file_path(filename):
    return curr_dir + '/models/' + filename
