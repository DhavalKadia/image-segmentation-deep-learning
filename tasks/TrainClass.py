"""

"""
# enable searchable package path
import sys
sys.path.append("..")

from package.tasks import *
from package.tools.loadtraindata import get_data
import package.networks.model as unet
from package.tools.statistics import training_stats
from package.tools.loadweights import load_weight_file_path
from package.setsystem.config import GPU_ENV

class TrainClass:
    def __init__(self, learning_rate=1e-4, train_batch_size=8, training_steps=5, epochs=10, utilize_whole=True, transfer_learning=False,
                 transfer_model_name='pretrained-model.h5', next_model_title='new-model', gpu_id=0):
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.training_steps = training_steps
        self.epochs = epochs
        # transfer learning, and use its weights file
        self.transfer_learning = transfer_learning
        # weight file name for tranfer learning
        self.transfer_model_name = transfer_model_name
        # file title for the current training
        self.next_model_title = next_model_title
        # gpu id: 0 or 1 or 2 and onwards. -1 if not available
        self.gpu_id = gpu_id
        # True if utilize entire train dataset
        self.utilize_whole = utilize_whole
        # default number of combined samples for training and validation
        self.data_batch_size = 20
        # training samples to total samples ratio
        self.train_ratio = 0.9
        # show training. set to 0 not to show
        self.verbose = 1

    # added for the future multi-gpu environment
    # more information: setsystem/config.py
    def assign_gpu(self, id):
        if GPU_ENV:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(id)

    def train(self):
        self.assign_gpu(self.gpu_id)

        # defined values in tasks/config
        HEIGHT = DATA_DIMENSION['height']
        WIDTH = DATA_DIMENSION['width']
        N_CLASS = DATA_DIMENSION['classes']

        # define neural network model
        model = unet.build_model((HEIGHT, WIDTH, 1), N_CLASS, self.learning_rate)

        # print neural network architecture
        if SHOW_MODEL:
            model.summary()

        # load weights for transfer learning
        if self.transfer_learning:
            model.load_weights(load_weight_file_path(self.transfer_model_name))

        # for outer_loop in tqdm(range(outer_training_steps)):
        for steps in range(self.training_steps):
            # fetch training data
            images, masks = get_data(self.data_batch_size, [HEIGHT, WIDTH], N_CLASS, self.utilize_whole, self.train_ratio, True)

            # partition for training and validation
            partition = int(self.train_ratio * len(images))

            # begin training
            history = model.fit(images[:partition], masks[:partition],
                                batch_size=self.train_batch_size,
                                epochs=self.epochs,
                                verbose=self.verbose,
                                validation_data=(images[partition:], masks[partition:]),
                                shuffle=True
                                )

            # useful when there are more epochs than the number of training steps
            if self.training_steps == 1:
                training_stats(history, self.next_model_title)

        # save the trained weights
        model.save_weights(weights_dir + self.next_model_title + '.h5')
        print('Saved', self.next_model_title + '.h5' + 'in package/models/')
