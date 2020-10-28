import os

# setting for the searchable path
curr_dir = os.getcwd()
curr_dir = curr_dir.replace('tasks', '')

"""
Data dimensions of the dataset in use
Assumption: all of the samples are having same size
Note: data are not down or up-sampled
"""

DATA_DIMENSION = {'height': 256,
                  'width': 256,
                  'classes': 5}

# print neural network
SHOW_MODEL = False

# directory for saving weights
weights_dir = curr_dir + '/models/'
