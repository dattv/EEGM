"""

"""
import argparse
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from datasets.breast_cancer_coimbra import breast_cancer_coimbra
from datasets.divorce import divorce_dataset
from datasets.somer_ville_happiness_survey import sumer_ville_happiness_surface_data
from datasets.sonar import sonar
from datasets.spect_heart import spect_heart
from losser import entropy_thin_fn, regularization_loss, continuous_regularization0, continuous_regularization1, \
    continuous_regularization2, continuous_regularization3
from nets.feed_forward import two_layers_FFN
from visualization import visualization


def compute_norm2(x):
    """

    :param x:
    :return:
    """
    norm = np.linalg.norm(x, axis=-1) ** 2
    return np.sum(norm)


def main(args):
    """

    :param args:
    :return:
    """
    data_folder = os.path.join(args.data_dir, args.data_name)
    if os.path.isdir(data_folder) is False:
        raise Exception("{} does not exist".format(data_folder))
    data_file = os.listdir(data_folder)
    if len(data_file) <= 0:
        raise Exception("There are not data in folder: {}".format(data_folder))

    if args.data_name == "sonar":
        data_loader = sonar
    elif args.data_name == "spect_heart":
        data_loader = spect_heart
    elif args.data_name == "BreastCAncerCoimbra":
        data_loader = breast_cancer_coimbra
    elif args.data_name == "SomervilleHappinessSurvey2015":
        data_loader = sumer_ville_happiness_surface_data
    elif args.data_name == 'divorce':
        data_loader = divorce_dataset

    else:
        raise Exception("Do not support this dataset: {}".format(args.data_name))

    if args.regularization == "l0_0":
        rg = continuous_regularization0
    elif args.regularization == "l0_1":
        rg = continuous_regularization1
    elif args.regularization == "l0_2":
        rg = continuous_regularization2
    elif args.regularization == "l0_3":
        rg = continuous_regularization3

    else:
        raise Exception("Do not support this regularization: {}".format(args.regulzarization))
    input_data, output_data = data_loader(os.path.join(data_folder, data_file[0]))

    folder_name = os.path.join(args.log_dir, args.data_name + '_' + args.regularization)
    if os.path.isdir(folder_name) is False:
        os.makedirs(folder_name, exist_ok=True)

    norm2 = compute_norm2(input_data)
    print('100% =======================>')
    print('norm2 sumer_ville_happiness_surface_data', norm2)
    print('4./((1.+1.)*norm2)', 4. / ((1. + 1.) * norm2))
    print('4./((1.+1./2.)*norm2)', 4. / ((1. + 1. / 2.) * norm2))
    print('4./((1.+1./3.)*norm2)', 4. / ((1. + 1. / 3.) * norm2))
    print('4./((1.+1./4.)*norm2)', 4. / ((1. + 1. / 4.) * norm2))
    print('4./((1.+1./5.)*norm2)', 4. / ((1. + 1. / 5.) * norm2))

    nsample = len(input_data)
    x_train = input_data[:int(nsample * 0.7), :]
    x_val = input_data[int(nsample * 0.7):, :]

    y_train = output_data[:int(nsample * 0.7)]
    y_val = output_data[int(nsample * 0.7):]

    batch_size = len(x_train)

    norm2 = compute_norm2(x_train)
    print('70% norm2 sumer_ville_happiness_surface_data', norm2, 'eta: ', 2. / norm2, 'lambda:', 0.0004 * norm2)

    # Instantiate an optimizer.
    # optimizer = keras.optimizers.SGD(learning_rate=2./norm2)
    optimizer = keras.optimizers.SGD(learning_rate=0.0027)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)

    input_size = (len(x_train[0, :]))
    model = two_layers_FFN(input_shape=input_size, nclass=2)

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(len(x_train))

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(len(x_val))

    loss_fn = entropy_thin_fn

    # Prepare the metrics.
    train_acc_metric = keras.metrics.Accuracy()
    val_acc_metric = keras.metrics.Accuracy()

    Recall = tf.keras.metrics.Recall()
    Precision = tf.keras.metrics.Precision()

    history_train_acc = []
    history_val_acc = []
    history_train_err = []
    history_val_err = []
    history_train_reg = []

    f = open(os.path.join(folder_name, 'log.txt'), 'w')

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = tf.reduce_mean(loss_fn(y, logits))

            loss_reg = regularization_loss(model.trainable_weights, lamb=0.0008, continuous_fn=rg,
                                           sigma=0.08)
            total_loss = loss_value + loss_reg

        grads = tape.gradient(total_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.uint8)
        train_acc_metric.update_state(y,
                                      tf.argmax(logits, axis=-1)
                                      )

        y = tf.cast(y, dtype=tf.float32)
        logits = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.float32)
        loss_mse_value = tf.math.sqrt(tf.reduce_mean(tf.math.square(y - logits)))

        return loss_value, loss_mse_value, loss_reg

    @tf.function
    def test_step(x, y):
        val_logits = tf.nn.sigmoid(model(x, training=False))
        y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.uint8)

        temp = tf.argmax(val_logits, axis=-1)
        val_acc_metric.update_state(y, temp)
        Precision.update_state(y, temp)
        Recall.update_state(y, temp)

    epochs = args.epoch
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            loss_value, mse, loss_reg = train_step(x_batch_train, y_batch_train)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f: %.4f: %.4f:"
                    % (step, float(loss_value), float(mse), float(loss_reg))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        history_train_acc.append(float(train_acc))
        history_train_err.append(float(mse))
        history_train_reg.append(float(loss_reg))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            # val_logits = model(x_batch_val, training=False)
            # # Update val metrics
            # val_acc_metric.update_state(y_batch_val, val_logits)
            test_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()

        precision_val = Precision.result()
        Precision.reset_states()
        recall_val = Recall.result()
        Recall.reset_states()

        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        history_val_acc.append(float(val_acc))

    f.write("Training acc over epoch: %.4f\n" % (float(train_acc),))
    f.write("Validation acc: %.4f\n" % (float(val_acc),))
    f.write("Precision: %.4f\n" % (float(precision_val),))
    f.write("Recall: %.4f" % (float(recall_val),))

    f.close()

    for x_batch_val, y_batch_val in val_dataset:
        # val_logits = model(x_batch_val, training=False)
        # # Update val metrics
        # val_acc_metric.update_state(y_batch_val, val_logits)
        test_step(x_batch_val, y_batch_val)

    # save

    np.save(os.path.join(folder_name, 'mse_loss_train.npy'), history_train_err)
    np.save(os.path.join(folder_name, train_acc_metric._name + '_train.npy'), history_train_acc)
    np.save(os.path.join(folder_name, val_acc_metric._name + '_test.npy'), history_val_acc)
    np.save(os.path.join(folder_name, 'reg_train.npy'), history_train_reg)
    model.save(os.path.join(folder_name, model.name + '.h5'), save_format='h5')

    visualization(input_files=[os.path.join(folder_name, 'mse_loss_train.npy'),
                               os.path.join(folder_name, train_acc_metric._name + '_train.npy'),
                               os.path.join(folder_name, val_acc_metric._name + '_test.npy'),
                               # os.path.join(folder_name, 'reg_train.npy')
                               ],
                  color=['r', 'b', 'g',
                         # 'c'
                         ],
                  title=os.path.split(folder_name)[-1],
                  labels=['mse_loss', 'acc_train', 'acc_test'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EEGM")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_name', type=str, default='sonar')
    parser.add_argument('--regularization', type=str, default='l0_1')
    parser.add_argument('--log_dir', type=str, default='logs_test')
    parser.add_argument('--epoch', type=int, default=5000)

    args = parser.parse_args()
    print(args)
    main(args)
