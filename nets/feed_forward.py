"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

def two_layers_FFN(input_shape, nclass):
    """

    :param input_shape:
    :param nclass:
    :return:
    """
    input_layer = tf.keras.Input(shape=input_shape, name='input')

    fcn1 = tf.keras.layers.Dense(nclass, use_bias=False,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.2),
                                 activation=None)(input_layer)


    return tf.keras.Model(inputs=[input_layer], outputs=[fcn1], name='two_layers_neural_network')


