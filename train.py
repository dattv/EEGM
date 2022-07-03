"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
from datasets.divorce import divorce_dataset
from datasets.somer_ville_happiness_survey import sumer_ville_happiness_surface_data
from datasets.spect_heart import spect_heart
from nets.feed_forward import two_layers_FFN
import tensorflow as tf
import argparse
from tqdm import trange
from visualization import visualization
from math import ceil
from entropy_loss import entropy_loss
from losser import entropy_thin_fn
import math

# calculate cross entropy for classification problem
from math import log
from numpy import mean


def custom_entropy_error_fun(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    # output = -tf.reduce_mean(y_true * tf.math.log(
    #     tf.clip_by_value(tf.math.divide_no_nan(y_pred, y_true), 1.e-10,
    #                      tf.math.divide_no_nan(y_pred, y_true))
    # ) +
    #                         (1. - y_true) * tf.math.log(
    #     tf.clip_by_value(tf.math.divide_no_nan(1. - y_pred, 1. - y_true), 1.e-10,
    #                      tf.math.divide_no_nan(1. - y_pred, 1. - y_true))
    # )
    #                         )
    divide = tf.math.divide_no_nan(
        y_pred, y_true
    )
    output = - tf.reduce_mean(
        tf.reduce_sum(
            y_true * tf.math.log(
                tf.clip_by_value(
                    divide,
                    1.e-6,
                    divide
                )
            ),
            axis=-1
        ),
        axis=0
    )

    return output


def custom_mse_fun(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # y_pred = tf.squeeze(y_pred, axis=-1)
    # y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    output = tf.losses.mean_squared_error(y_true, y_pred)
    return output


def custom_l0_entropy_err_func(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred = tf.squeeze(y_pred, axis=-1)
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    output = -tf.reduce_sum(y_true * tf.math.log(y_pred) +
                            (1. - y_true) * tf.math.log(
        1. - y_pred))

    return output


def train(args):
    """

    :return:
    """
    max_iter = args.max_iter

    custom_objective_function = entropy_thin_fn#entropy_loss()  # custom_mse_fun#custom_entropy_error_fun

    if args.data is 'divorce':
        input_data, output_data = divorce_dataset(data_file='./data/divorce/divorce.csv')
    elif args.data is 'sumer_ville_happiness_surface':
        input_data, output_data = sumer_ville_happiness_surface_data(None)
    elif args.data is 'spect_heart':
        X_train, Y_train = spect_heart('./data/spect_heart/SPECT.train')
        X_test, Y_test = spect_heart('./data/spect_heart/SPECT.test')
    else:
        input_data = None
        output_data = None
        raise Exception

    nsample = len(input_data)
    X_train = input_data[:int(nsample * 0.7), :]
    X_test = input_data[int(nsample * 0.7):, :]

    Y_train = output_data[:int(nsample * 0.7)]
    Y_test = output_data[int(nsample * 0.7):]

    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=2)
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=2)
    if args.batch_size > 0:
        batch_size = args.batch_size
    else:
        batch_size = len(Y_train)

    nepoch = ceil(max_iter / ceil(len(X_train) / batch_size))  # max_iter * batch_size // len(X_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)

    input_size = (len(X_train[0, :]))
    model = two_layers_FFN(input_shape=input_size, nclass=2)
    print(model.summary())

    if args.lr_schedule == 'constant':
        learning_rate_fn = args.lr
    else:
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=args.lr,
            decay_steps=max_iter,
            end_learning_rate=1.e-5)

    def total_loss(y_true, y_pred, model):
        """

        :param y_true:
        :param y_pred:
        :param model:
        :return:
        """
        total = custom_objective_function(y_true, y_pred)
        total = tf.reduce_mean(total)

        if args.regularization:
            total += 1.e-3 * tf.reduce_sum(model.trainable_weights)

        return total

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)
    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.Accuracy()
    val_acc_metric = tf.keras.metrics.Accuracy()
    train_errors = tf.keras.losses.mean_squared_error

    TEST_acc = []
    TRAIN_acc = []
    TEST_err = []
    TRAIN_err = []
    epochs = nepoch
    t = trange(epochs, desc='Bar desc', leave=True)
    for _ in t:
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = total_loss(y_batch_train, logits, model)

            grads = tape.gradient(loss_value, model.trainable_variables)

            # grads = [(tf.clip_by_value(grad, -1., 1.))
            #          for grad in grads]
            logits_sigmoid = tf.nn.sigmoid(logits)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits_sigmoid)

            # train_err_RMSE.update_state(tf.squeeze(y_batch_train), tf.squeeze(logits))
            TRAIN_err.append(
                np.sqrt(train_errors(tf.argmax(y_batch_train, axis=-1),
                                     tf.argmax(logits_sigmoid, axis=-1)).numpy())
                # train_err_RMSE.result()
            )
            # train_err_RMSE.reset_states()

        # print("current learning rate:", optimizer._decayed_lr(tf.float32))
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        TRAIN_acc.append(float(train_acc))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        TEST_acc.append(float(val_acc))
        val_acc_metric.reset_states()

        t.set_description(
            "train_loss: {:.4f} | train_acc: {:.4f} val_acc: {:.4f} | lr: {:.7f}".format(float(loss_value),
                                                                                         float(train_acc),
                                                                                         float(val_acc),
                                                                                         float(optimizer._decayed_lr(
                                                                                             np.float32)))
        )
        t.refresh()

    if os.path.isdir('./logs/{}'.format(args.data)) is False:
        os.makedirs('./logs/{}'.format(args.data), exist_ok=True)

    np.save('./logs/{}/{}.npy'.format(args.data, custom_objective_function.__name__), TRAIN_err)

    visualization(input_files=['./logs/{}/{}.npy'.format(args.data, custom_objective_function.__name__)]
                  )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data', type=str, default='sumer_ville_happiness_surface', help='name of dataset')
    parser.add_argument('--loss_type', type=int, default=0,
                        help='0:= MSE, 1:=entropy error function, 2:=l0 entropy error function')
    parser.add_argument('--logs', type=str, default='./logs', help='log dir where storing result *.npy file')
    parser.add_argument('--max_iter', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=-1)

    parser.add_argument('--lr', type=float, default=0.0025)
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--regularization', type=str, default=None)

    args = parser.parse_args()
    train(args)
