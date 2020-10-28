import tensorflow as tf
from package.losses.tf_losses import sdsc_bce_loss
from package.metrics.tf_metrics import dice_coef

# channel growth factor: how much fast channels increase towards bottleneck layer
multi_factor = 1.5

# assignment of number of channels in different levels of U-Net
n_filters_l1 = 96
n_filters_l2 = int(multi_factor * n_filters_l1)
n_filters_l3 = int(multi_factor * n_filters_l2)
n_filters_l4 = int(multi_factor * n_filters_l3)
n_filters_l5 = int(multi_factor * n_filters_l4)
n_filters_l6 = int(multi_factor * n_filters_l5)

# Squeeze and Excitation channel division ratio: should be >= 2
se_div_factor = 2

"""
dilation rates: should be >(1, 1) for its applicability
current values does not represent dilation
"""
dilation_1 = (1, 1)
dilation_2 = (1, 1)
dilation_3 = (1, 1)

# gives tensor shape
def get_tensor_shape(tensor):
    return getattr(tensor, '_shape_val')

"""
Squeeze-and-excitation residual network
Ref: Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." In Proceedings of the IEEE conference on computer 
vision and pattern recognition, pp. 7132-7141. 2018.
"""
def SE_ResNet(x, se_div_factor):
    bias = True
    n_filters = get_tensor_shape(x)[-1]
    sq_dense_n = int(n_filters // se_div_factor)
    se_shape = (1, 1, n_filters)

    # gives average values per channel
    gap_0 = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)
    gap_0 = tf.keras.layers.Reshape(se_shape)(gap_0)
    # dense network
    dense = tf.keras.layers.Dense(sq_dense_n, activation='relu', use_bias=bias)(gap_0)
    sq = tf.keras.layers.Dense(n_filters, activation='sigmoid', use_bias=bias)(dense)
    sq = tf.keras.layers.multiply([x, sq])
    # residual path
    sq_res = tf.keras.layers.add([x, sq])

    return sq_res

# custom encoder
def encode(n_filters, input, dilation):
    conv_1 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), dilation_rate=dilation, activation='relu', padding='same',
                                    data_format='channels_last')(input)
    conv_2 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), dilation_rate=dilation, activation='relu', padding='same',
                                  data_format='channels_last')(conv_1)

    conv_add = tf.keras.layers.add([conv_1, conv_2])

    # maxpooling layer
    pool = tf.keras.layers.MaxPooling2D(strides=(2, 2))(conv_add)

    return conv_add, pool

# custom bottleneck layer
def bottleneck(n_filters, input, dilation):
    conv_1 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), dilation_rate=dilation, activation='relu', padding='same',
                                    data_format='channels_last')(input)
    conv_2 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), dilation_rate=dilation, activation='relu', padding='same',
                                    data_format='channels_last')(conv_1)

    conv_se_1 = SE_ResNet(conv_2, se_div_factor)

    conv_3 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), dilation_rate=dilation, activation='relu',
                                    padding='same',
                                    data_format='channels_last')(conv_se_1)

    conv_se_2 = SE_ResNet(conv_3, se_div_factor)

    return conv_se_2

# custom decoder
def decode(n_filters, input, concatenate_input):
    conv_trans = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), activation='relu',
                                                 padding='same', data_format='channels_last')(input)

    conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same',
                                    data_format='channels_last')(conv_trans)

    conc = tf.keras.layers.concatenate([conv, concatenate_input])

    conc_se = SE_ResNet(conc, se_div_factor)

    conv_1 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same',
                                    data_format='channels_last')(conc_se)
    conv_2 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same',
                                    data_format='channels_last')(conv_1)

    return conv_2

# output layer
def output_layer(input, n_class):
    output = tf.keras.layers.Conv2D(filters=n_class, kernel_size=(1, 1), activation='sigmoid',
                                    data_format='channels_last')(input)

    return output

"""
U-Net based neural network
Ref: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." 
In International Conference on Medical image computing and computer-assisted intervention, pp. 234-241. Springer, Cham, 2015.
"""
def build_model(input_shape, n_class, learning_rate=1e-3):
    # input
    inputs = tf.keras.layers.Input(input_shape)

    #encoder
    l1_conv, l1_pool = encode(n_filters_l1, inputs, dilation_3)
    l2_conv, l2_pool = encode(n_filters_l2, l1_pool, dilation_3)
    l3_conv, l3_pool = encode(n_filters_l3, l2_pool, dilation_2)
    l4_conv, l4_pool = encode(n_filters_l4, l3_pool, dilation_2)
    l5_conv, l5_pool = encode(n_filters_l5, l4_pool, dilation_1)

    # bottleneck layer
    bottleneck_conv = bottleneck(n_filters_l6, l5_pool, dilation_1)

    # decoder
    l5_conv_trans = decode(n_filters_l5, bottleneck_conv, l5_conv)
    l4_conv_trans = decode(n_filters_l4, l5_conv_trans, l4_conv)
    l3_conv_trans = decode(n_filters_l3, l4_conv_trans, l3_conv)
    l2_conv_trans = decode(n_filters_l2, l3_conv_trans, l2_conv)
    l1_conv_trans = decode(n_filters_l1, l2_conv_trans, l1_conv)

    # predictions
    outputs = output_layer(l1_conv_trans, n_class)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # using adam optimizer, a custom loss function and the matric dice similarity coefficient
    model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss=sdsc_bce_loss, metrics=[dice_coef])

    return model
