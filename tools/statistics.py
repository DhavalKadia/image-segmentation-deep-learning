from package.tools import *
import matplotlib.pyplot as plt

# display training and validation graph
def training_stats(history, model_name):
    model_acc = history.history['dice_coef']
    val_acc = history.history['val_dice_coef']

    plt.plot(model_acc, color="tomato", linewidth=2)
    plt.plot(val_acc, color="steelblue", linewidth=2)
    plt.legend(["Training", "Validation"], loc="lower right")

    plt.title("Training graph")
    plt.xlabel("Epochs")
    plt.ylabel("DSC")
    plt.grid()

    # save the graph
    if show_save_training_stats:
        plots_saving_path = os.path.join(parent_dir + model_name + '_' + 'traingraph' + '.png')
        plt.savefig(plots_saving_path)
        plt.show()

    plt.clf()
    plt.close()