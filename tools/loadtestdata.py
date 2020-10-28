"""
Note: data augmentation is not set to be performed for testing data
"""

from package.tools import *

# load test data
handle_images = h5py.File(TEST_DATASET_PATH['images'], 'r')
dataset_images = handle_images['images']

handle_labels = h5py.File(TEST_DATASET_PATH['labels'], 'r')
dataset_labels = handle_labels['labels']

print('Test data are loaded')

def get_data(n_samples, dims, n_class, test_whole, start=0):
    # data dimensions
    HEIGHT = dims[0]
    WIDTH = dims[1]
    N_CLASS = n_class

    # True if test entire test data
    if test_whole:
        n_samples = len(dataset_images)
        start = 0

    images = np.zeros((n_samples, HEIGHT, WIDTH, 1), dtype=float)
    masks = np.zeros((n_samples, HEIGHT, WIDTH, N_CLASS), dtype=float)

    for n in range(n_samples):
        image = dataset_images[start + n]
        mask = dataset_labels[start + n]

        # normalize the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        images[n] = image
        masks[n] = mask

    images = np.reshape(images, (n_samples, HEIGHT, WIDTH, 1))

    return images, masks
