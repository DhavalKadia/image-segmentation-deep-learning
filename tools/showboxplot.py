import os
from package.tools import *
import matplotlib.pyplot as plt

# display and save the boxplot
def show_boxplot(metric_array, modelname):
    n_samples = len(metric_array)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Box plot: Dice Similarity Coefficient')
    ax.boxplot(metric_array, labels=[str(n_samples) + ' ' + 'samples'])

    plots_saving_path = os.path.join(parent_dir + 'boxplot-' + modelname[:-3] + '.png')
    plt.savefig(plots_saving_path, dpi=120)
    plt.show()
    plt.clf()
    plt.close()
