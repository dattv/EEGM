""""""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf


def continuous_l12_0(t, a):
    """

    :param t:
    :param a:
    :return:
    """

    def first_cond(t):
        return -t

    def second_cond(t):
        return -1. * t ** 4 / 8. / a ** 3 + 3. / 4. / a * t ** 2 + 3. / 8. * a

    def third_cond(t):
        return t

    res = tf.where(tf.less_equal(t, -a), first_cond(t), t)
    res = tf.where(tf.greater_equal(t, a), third_cond(res), res)
    res = tf.where(tf.logical_and(tf.greater(t, -a), tf.less(t, a)), second_cond(res), res)
    return res

    # return tf.case([(tf.less_equal(t, -a), first_cond),
    #                 (tf.greater_equal(t, a), third_cond)],
    #                second_cond)

def continuous_l12_1(t, a):
    """

    :param t:
    :param a:
    :return:
    """
    return t**2


def continuous_l12_2(t, a):
    """

    :param t:
    :param a:
    :return:
    """
    return tf.sqrt(tf.abs(t))


def continuous_regularization0(t, sigma=1.e-4):
    """

    :param t:
    :param sigma:
    :return:
    """
    return 1. - tf.math.exp(-t * t / (2. * sigma * sigma))


def continuous_regularization1(t, sigma=1.e-4):
    """

    :param t:
    :param sigma:
    :return:
    """
    return 1. - sigma * sigma / (t * t + sigma * sigma)


def continuous_regularization2(t, sigma=1.e-4):
    """

    :param t:
    :param sigma:
    :return:
    """
    return 1. - tf.math.sin(t / sigma) * sigma / t


def continuous_regularization3(t, sigma=1.e-4):
    """

    :param t:
    :param sigma:
    :return:
    """
    A = tf.math.exp(t * t / (2. * sigma))
    B = tf.math.exp(- t * t / (2. * sigma))
    return (A - B) / (A + B)


def entropy_thin_fn(y_true, y_pred):
    """

        For brevity, let `x = logits`, `z = labels`.  The logistic loss is

            z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
          = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
          = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
          = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
          = (1 - z) * x + log(1 + exp(-x))
          = x - x * z + log(1 + exp(-x))

        For x < 0, to avoid overflow in exp(-x), we reformulate the above

            x - x * z + log(1 + exp(-x))
          = log(exp(x)) - x * z + log(1 + exp(-x))
          = - x * z + log(1 + exp(x))

        Hence, to ensure stability and avoid overflow, the implementation uses this
        equivalent formulation

          max(x, 0) - x * z + log(1 + exp(-abs(x)))


    :param y_true: y_true := (batch_size, one_hot)
    :param y_pred: y_pred := (batch_size, one_hot)
    :return:
    """
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    # y_true = tf.one_hot(y_true, depth=10, dtype=tf.float32)
    return tf.reduce_sum(tf.maximum(y_pred, 0) - y_pred * y_true + tf.math.log(1 + tf.math.exp(-abs(y_pred))), axis=-1)


def regularization_loss(weights, lamb, continuous_fn, sigma):
    """

    :param weights:
    :param lamb:
    :param continuous_fn:
    :param sigma:
    :return:
    """
    temp = 0.
    for weight in weights:
        temp += tf.reduce_sum(continuous_fn(weight, sigma))

    return lamb * temp


if __name__ == '__main__':
    # logits = tf.constant([1., -1., 0., 1., -1., 0., 0.])
    # labels = tf.constant([0., 0., 0., 1., 1., 1., 0.5])
    labels = tf.constant([[0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 0]], dtype=tf.float32)
    logits = tf.constant([[0.2, 0.2, 0.2, 0.2, 0.2],
                          [0.3, 0.3, 0.2, 0.1, 0.1]], dtype=tf.float32)

    print(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits).numpy())

    print(entropy_thin_fn(labels, logits).numpy())
