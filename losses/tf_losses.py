import tensorflow as tf
from tensorflow.python.keras.backend import sum, square, pow, log
from package.metrics.tf_metrics import dice_coef, soft_dice_coef, gen_dice_coef

# binary cross entropy loss: useful for multi-class problem
def binarycrossentropy(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss = bce(y_true, y_pred)

    return loss

# variants of loss based on dice similarity coefficient
# standard dice similarity coefficient
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# soft dice similarity coefficient
def soft_dice_coef_loss(y_true, y_pred):
    return 1 - soft_dice_coef(y_true, y_pred)

# generalized dice similarity coefficient
def gen_dice_coef_loss(y_true, y_pred):
    return 1 - gen_dice_coef(y_true, y_pred)

# hybrid loss
def sdsc_bce_loss(y_true, y_pred):
    weight = 0.8
    gamma = 0.3

    sdsc = soft_dice_coef(y_true, y_pred)
    wce = binarycrossentropy(y_true, y_pred)

    loss = weight * pow(-log(tf.clip_by_value(sdsc, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())), gamma) + (1 - weight) * wce

    return loss
