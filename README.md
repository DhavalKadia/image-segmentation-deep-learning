# Medical Image Segmentation using Deep Learning
This project trains the deep learning model based on U-Net on the given dataset.

### Requirements
All train and test data are required to be of the same size.

### Data locations
Train data are supposed to be placed inside package/data/train/
Test data are supposed to be placed inside package/data/test/
Modify image and ground-truth filenames in package/tools/config.py
Place pretrained-model.h5 into package/models/

### General configurations
package/setsystem/config/GPU_ENV: True if GPU(s) are available.
package/tasks/config/DATA_DIMENSION: Data dimension for current dataset.
Constants for data augmentation are placed in package/tools/config.py

### Basics to run the code
Open Python console and change its directory to the python file you want to run.

### Setup
Having package as a subdirectory, enter the following commands to get into the directory to begin setup.

```
import os

os.chdir(os.getcwd() + "/package/setsystem/setup/")

from package.setsystem.setup import setup
```

### Train the model
Train example:

```
import os

os.chdir(os.getcwd() + "/package/tasks/")

from package.tasks.TrainClass import TrainClass

trainObject = TrainClass()

trainObject.train()
```
or
```
trainObject2 = TrainClass(learning_rate=1e-4, train_batch_size=8, training_steps=50, epochs=1, utilize_whole=True,
                    transfer_learning=False, next_model_title='model-2')
                    
trainObject2.train()
```
or

Run train.py

### Test the model
Test example:
```
from package.tasks.TestClass import TestClass

testObject = TestClass()
```
or
```
testObject = TestClass('pretrained-model.h5')

testObject.test()
```
or

Run test.py

### Functionality
#### Training the model

* Transfer learning can be started by using a pre-trained weight file.

* The augmented training data will be fetched multiple times and trained over a certain number of epochs. If the training steps are equal to the one with several epochs, the training graph will be shown.

#### Testing the model
* Use the pre-trained weight file, and it will print the dice similarity metric of each sample.
* The class-wise and merged class segmentation results will be plotted and saved.
* The overview of the given evaluation metric for test data is plotted as the boxplot. The boxplot provides the visual representation of a number set - how numbers are distributed. It is very important when a few values are responsible for bad results (having the majority of the good results). It represents the range where the majority of the result values are placed.