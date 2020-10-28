import matplotlib.pyplot as plt
from package.tools import *
from package.tasks import DATA_DIMENSION

# display image and five classes with their predictions and ground-truth
def disp_segmentation(image, prediction, mask, title):
    # normalize values to 0 to 255
    image *= 255
    prediction *= 255
    mask *= 255

    # convert values into integer
    image = np.array(image, np.int32)
    prediction = np.array(prediction, np.int32)
    mask = np.array(mask, np.int32)

    # number of classes for the given dataset
    N_ClASSES = 5

    plt.figure(dpi=fig_dpi)
    fig, axarr = plt.subplots(2, N_ClASSES + 1)

    fig.suptitle('Class-wise segmentation')

    # assign images to the subplot
    axarr[0][0].imshow(image, cmap=cmap)
    axarr[1][0].imshow(image, cmap=cmap)

    for l_id in range(N_ClASSES):
        axarr[0][l_id + 1].imshow(prediction[:, :, l_id], cmap=cmap)
        axarr[1][l_id + 1].imshow(mask[:, :, l_id], cmap=cmap)

    # disable axis
    axarr[0][0].axis('off')
    axarr[1][0].axis('off')

    for l_id in range(N_ClASSES):
        axarr[0][l_id + 1].axis('off')
        axarr[1][l_id + 1].axis('off')

    # assign titles
    axarr[0][0].title.set_text('image')
    axarr[0][1].title.set_text('L atrium')
    axarr[0][2].title.set_text('L ventricle')
    axarr[0][3].title.set_text('R atrium')
    axarr[0][4].title.set_text('R ventricle')
    axarr[0][5].title.set_text('aortic root')

    # define border layout
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    # save figure
    if save_figs:
        plots_saving_path = os.path.join(parent_dir + 'predictions-' + title + '.png')
        plt.savefig(plots_saving_path, dpi=fig_dpi)
        plt.show()

    plt.clf()
    plt.close()

# merge segmentation of all classes
def merged_segmentation(image, prediction, mask, title):
    # data dimension
    HEIGHT = DATA_DIMENSION['height']
    WIDTH = DATA_DIMENSION['width']

    # create color images
    merged_predictions = np.zeros((HEIGHT, WIDTH, 3))
    merged_masks = np.zeros((HEIGHT, WIDTH, 3))

    # generate color image for prediction
    merged_predictions[:, :, 0] += prediction[:, :, 0]

    merged_predictions[:, :, 1] += prediction[:, :, 1]

    merged_predictions[:, :, 2] += prediction[:, :, 2]

    merged_predictions[:, :, 0] += prediction[:, :, 3]
    merged_predictions[:, :, 1] += prediction[:, :, 3]

    merged_predictions[:, :, 1] += prediction[:, :, 4]
    merged_predictions[:, :, 2] += prediction[:, :, 4]

    # generate color ground-truth image
    merged_masks[:, :, 0] += mask[:, :, 0]

    merged_masks[:, :, 1] += mask[:, :, 1]

    merged_masks[:, :, 2] += mask[:, :, 2]

    merged_masks[:, :, 0] += mask[:, :, 3]
    merged_masks[:, :, 1] += mask[:, :, 3]

    merged_masks[:, :, 1] += mask[:, :, 4]
    merged_masks[:, :, 2] += mask[:, :, 4]

    np.seterr(invalid='ignore')

    # normalization to avoid value overflow and wrong image representation
    for n in range(3):
        merged_predictions[:, :, n] = (merged_predictions[:, :, n] - np.min(merged_predictions[:, :, n])) / (np.max(merged_predictions[:, :, n]) - np.min(merged_predictions[:, :, n]))
        merged_masks[:, :, n] = (merged_masks[:, :, n] - np.min(merged_masks[:, :, n])) / (np.max(merged_masks[:, :, n]) - np.min(merged_masks[:, :, n]))

    plt.figure(dpi=fig_dpi)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    fig.suptitle('Merged segmentation')

    # assign images to the subplot
    ax1.imshow(image, cmap=cmap)
    ax2.imshow(merged_predictions)
    ax3.imshow(merged_masks)

    # disable axis
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    # assign titles
    ax1.title.set_text('image')
    ax2.title.set_text('prediction')
    ax3.title.set_text('ground-truth')

    # define border layout
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    # save figure
    if save_figs:
        plots_saving_path = os.path.join(parent_dir + 'merge-pred-' + title + '.png')
        plt.savefig(plots_saving_path, dpi=fig_dpi)
        plt.show()

    plt.clf()
    plt.close()
