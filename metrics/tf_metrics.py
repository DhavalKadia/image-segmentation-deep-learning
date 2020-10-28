import tensorflow as tf
from tensorflow.python.keras.backend import sum, square, pow, log

# dice similarity coefficient metric
def dice_coef(y_true, y_pred):
    smooth = 1e-7

    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = sum(y_true_f * y_pred_f)

    return 2. * (intersection+smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)

# soft dice similarity coefficient metric
def soft_dice_coef(y_true, y_pred):
    smooth = 1e-7

    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = sum(y_true_f * y_pred_f)

    return 2. * (intersection+smooth) / (sum(square(y_true_f)) + sum(square(y_pred_f)) + smooth)

# generalized dice similarity coefficient metric (based on soft dice similarity coefficient)
def gen_dice_coef(y_true, y_pred):
    smooth = 1e-7

    # get class-wise sums
    weights = sum(y_true, axis=[0, 1, 2]) + smooth
    # normalize array
    weights = weights / sum(weights)

    # invert weights (importance)
    weights = 1. / weights

    intersection = sum(weights * sum(square(y_true * y_pred), axis=[0, 1, 2]))
    positives_sum = sum(weights * (sum(square(y_true), axis=[0, 1, 2]) + sum(square(y_pred), axis=[0, 1, 2])))

    return 2. * (intersection+smooth) / (positives_sum+smooth)
