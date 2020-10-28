from package.utils import *

def shuffle_lists(list_1, list_2):  #+ if not same size
    perm = list(range(len(list_1)))
    random.shuffle(perm)
    list_1 = [list_1[index] for index in perm]
    list_2 = [list_2[index] for index in perm]

    return list_1, list_2

def deform_elastic(image, mask):
    [image, mask[:, :, 0], mask[:, :, 1], mask[:, :, 2], mask[:, :, 3], mask[:, :, 4]] = \
        elasticdeform.deform_random_grid(
            [image, mask[:, :, 0], mask[:, :, 1], mask[:, :, 2], mask[:, :, 3], mask[:, :, 4]])

    return image, mask

def rotate_image(image, mask):
    angle = random.uniform(-max_angle, max_angle)
    image = ndimage.rotate(image, angle, reshape=False)

    for l_id in range(5):
        mask[:, :, l_id] = ndimage.rotate(mask[:, :, l_id], angle, reshape=False)

    return image, mask

def correct_gamma(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    gamma = random.uniform(gamma_min, gamma_max)
    image = image ** gamma

    return image