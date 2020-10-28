import matplotlib.pyplot as plt
from package.tools import *

# display image
def display(image):
    plt.imshow(image, cmap=cmap)
    plt.show()