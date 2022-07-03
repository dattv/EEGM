"""

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import time
import numpy as np
import os
from losser import entropy_thin_fn, regularization_loss, continuous_regularization0, continuous_regularization1, \
    continuous_regularization2, continuous_regularization3, \
    continuous_l12_0, continuous_l12_1, continuous_l12_2
from datasets.divorce import divorce_dataset
from datasets.spect_heart import spect_heart
from datasets.somer_ville_happiness_survey import sumer_ville_happiness_surface_data
from datasets.breast_cancer_coimbra import breast_cancer_coimbra
from datasets.sonar import sonar
from nets.feed_forward import two_layers_FFN
import os
from visualization import visualization

# Prepare the training dataset.
# input_data, output_data = divorce_dataset(data_file='./data/divorce/divorce.csv')
input_data, output_data = spect_heart('./data/spect_heart/SPECT.test')
# input_data, output_data = sumer_ville_happiness_surface_data('./data/SomervilleHappinessSurvey2015/SomervilleHappinessSurvey2015.csv')
# input_data, output_data = breast_cancer_coimbra('/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/EEGM/data/BreastCAncerCoimbra/dataR2.csv')
# input_data, output_data = sonar('./data/sonar/sonar.all-data')
folder_name = "/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/EEGM/download_resulst/logs/spect_heart_reg3"

def compute_norm2(x):
    """

    :param x:
    :return:
    """
    norm = np.linalg.norm(x, axis=-1)**2
    return np.sum(norm)

norm2 = compute_norm2(input_data)
print('100% =======================>')
print('norm2 sumer_ville_happiness_surface_data', norm2)
print('4./((1.+1.)*norm2)', 4./((1.+1.)*norm2))
print('4./((1.+1./2.)*norm2)', 4./((1.+1./2.)*norm2))
print('4./((1.+1./3.)*norm2)', 4./((1.+1./3.)*norm2))
print('4./((1.+1./4.)*norm2)', 4./((1.+1./4.)*norm2))
print('4./((1.+1./5.)*norm2)', 4./((1.+1./5.)*norm2))



nsample = len(input_data)
x_train = input_data[:int(nsample * 0.7), :]
x_val = input_data[int(nsample * 0.7):, :]

y_train = output_data[:int(nsample * 0.7)]
y_val = output_data[int(nsample * 0.7):]

batch_size = len(x_train)

norm2 = compute_norm2(x_train)

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

model = tf.keras.models.load_model(
    os.path.join(folder_name, 'two_layers_neural_network.h5'))

print(model.summary())
Recall = tf.keras.metrics.Recall()
Precision = tf.keras.metrics.Precision()
Accuracy = tf.keras.metrics.Accuracy()

# @tf.function
def test_step(x, y):
    val_logits = tf.nn.sigmoid(model(x, training=False))
    y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.uint8)

    temp = tf.argmax(val_logits, axis=-1)
    Precision.update_state(y, temp)
    Recall.update_state(y, temp)
    Accuracy.update_state(y, temp)

for x_batch_val, y_batch_val in val_dataset:
    test_step(x_batch_val, y_batch_val)
# y_scores = tf.nn.sigmoid(model(x_val, training=False))[:,1]
#
# precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
# display = PrecisionRecallDisplay.from_predictions(y_val, y_scores, name="")
# _ = display.ax_.set_title("2-class Precision-Recall curve")

precision_val = Precision.result()
Precision.reset_states()
recall_val = Recall.result()
Recall.reset_states()

accuracy_val = Accuracy.result()
Accuracy.reset_states()
print("precision_val: %.4f" % (float(precision_val), ))
print("recall_val: %.4f" % (float(recall_val), ))
print('accuracy_val: %.4f' % (float(accuracy_val), ))

with open(os.path.join(folder_name, 'acc_on_val.txt'), 'w') as f:
    f.write("precision_val: %.4f \n" % (float(precision_val), ))
    f.write("recall_val: %.4f\n" % (float(recall_val), ))
    f.write('accuracy_val: %.4f' % (float(accuracy_val), ))