"""

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

class entropy_loss(object):
    """

    """
    def __init__(self,reduction=tf.keras.losses.Reduction.AUTO,
               name=None):
        """
        
        :param reduction: 
        :param name: 
        """
        self.__name__ = 'entropy_loss'
        # super(entropy_loss, self).__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """

        with tf.name_scope('entropy_error_function'):
            y_true = tf.cast(y_true, dtype=tf.float32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)
            positive_label_mask = tf.equal(y_true, 1.0)

            # probs_gt = tf.where(positive_label_mask, y_pred/y_true, (1.0 - y_pred)/(1.0 - y_true))
            probs_gt = tf.where(positive_label_mask, y_pred, (1.0 - y_pred))

            lograrithm = tf.math.log(probs_gt)

            weighted_loss = tf.where(positive_label_mask, y_true * lograrithm, (1.0 - y_true) * lograrithm)

        return tf.reduce_mean(-tf.reduce_sum(weighted_loss, axis=-1))

    def get_config(self):
        config = {
        }
        base_config = super(entropy_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


