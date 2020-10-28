from package.tools import *
from package.utils.procdata import shuffle_lists, deform_elastic, rotate_image, correct_gamma

# load train data
handle_images = h5py.File(TRAIN_DATASET_PATH['images'], 'r')
dataset_images = handle_images['images']

handle_labels = h5py.File(TRAIN_DATASET_PATH['labels'], 'r')
dataset_labels = handle_labels['labels']

print('Train data are loaded')

def get_data(n_samples, dims, n_class, utilize_whole, train_ratio, shuffle=False, start=0):
    # data dimensions
    HEIGHT = dims[0]
    WIDTH = dims[1]
    N_CLASS = n_class

    # True while utilizing entire train dataset
    if utilize_whole:
        n_samples = len(dataset_images)
        start = 0

    # number of samples for training and remaining for the validation
    partition = int(train_ratio * n_samples)

    images = np.zeros((n_samples, HEIGHT, WIDTH, 1), dtype=float)
    masks = np.zeros((n_samples, HEIGHT, WIDTH, N_CLASS), dtype=float)

    for n in range(n_samples):
        image = dataset_images[start + n]
        mask = dataset_labels[start + n]

        image = np.reshape(image, (HEIGHT, WIDTH))

        # normalize the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        if n < partition:
            if random.random() < e_deform_probability:
                image, mask = deform_elastic(image, mask)

            if rotate_angle:
                image, mask = rotate_image(image, mask)

            if random.random() < gamma_probability:
                image = correct_gamma(image)

        # normalize the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.reshape(image, (HEIGHT, WIDTH, 1))

        images[n] = image
        masks[n] = mask

    # shuffle train data (not validation data)
    if shuffle:
        images[:partition], masks[0:partition] = shuffle_lists(images[:partition], masks[0:partition])

    images = np.reshape(images, (n_samples, HEIGHT, WIDTH, 1))
    masks = np.reshape(masks, (n_samples, HEIGHT, WIDTH, N_CLASS))

    return images, masks

