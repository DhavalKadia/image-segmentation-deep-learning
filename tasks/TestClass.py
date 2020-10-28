"""
Test Class for the evaluation
"""
# add searchable package path
import sys
sys.path.append("..")

from package.tasks import *
from package.tools.loadtestdata import get_data
import package.networks.model as unet
from package.tools.loadweights import load_weight_file_path
from package.metrics.numpy_metrics import dice_similarity_coef
from package.tools.displaysegmentation import disp_segmentation, merged_segmentation
from package.tools.showboxplot import show_boxplot
from package.setsystem.config import GPU_ENV

class TestClass:
    def __init__(self, test_model_name='pretrained-model.h5', test_whole=True, start_sample_index=0, data_batch_size=40, gpu_id=0):
        # transfer learning, and use its weights file
        self.test_model_name = test_model_name
        # True if test all samples in testing set
        self.test_whole = test_whole
        # starting index of samples in testing set
        self.start_sample_index = start_sample_index
        # number of testing samples
        self.data_batch_size = data_batch_size
        # gpu id: 0 or 1 or 2 and onwards. -1 if not available
        self.gpu_id = gpu_id

    # added for the future multi-gpu environment
    # more information: setsystem/config.py
    def assign_gpu(self, id):
        if GPU_ENV:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(id)

    def test(self):
        self.assign_gpu(self.gpu_id)

        # data dimensions: defined values in tasks/config
        HEIGHT = DATA_DIMENSION['height']
        WIDTH = DATA_DIMENSION['width']
        N_CLASS = DATA_DIMENSION['classes']

        # define neural network model
        model = unet.build_model((HEIGHT, WIDTH, 1), N_CLASS)
        # load weights
        model.load_weights(load_weight_file_path(self.test_model_name))

        # fetch test data
        images, masks = get_data(self.data_batch_size, [HEIGHT, WIDTH], N_CLASS, test_whole=self.test_whole)

        avg_dsc = 0
        dsc_array = []

        n_samples = len(images)

        print('sample\t\t\tDSC')

        for n in range(n_samples):
            image = np.reshape(images[n], (1, HEIGHT, WIDTH, 1))
            mask = np.reshape(masks[n], (1, HEIGHT, WIDTH, N_CLASS))

            prediction = model.predict(image)

            dsc = dice_similarity_coef(mask, prediction)
            avg_dsc += dsc
            dsc_array.append(dsc)

            print(n + 1, '\t\t\t\t', dsc)

            prediction = np.reshape(prediction, (HEIGHT, WIDTH, N_CLASS))
            mask = np.reshape(mask, (HEIGHT, WIDTH, N_CLASS))
            image = np.reshape(image, (HEIGHT, WIDTH))

            disp_segmentation(image, prediction, mask, self.test_model_name[:-3] + '-' + str(n))
            merged_segmentation(image, prediction, mask, self.test_model_name[:-3] + '-' + str(n))

        avg_dsc /= n_samples
        print('average dice similarity coefficient = ', avg_dsc)

        show_boxplot(dsc_array, self.test_model_name)
