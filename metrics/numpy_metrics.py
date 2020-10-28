import numpy as np

# dice similarity coefficient metric
def dice_similarity_coef(y_true_f, y_pred_f):
    smooth = 1e-7

    intersection = np.sum(y_true_f * y_pred_f)
    return (2.*intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
