"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd

def divorce_dataset(data_file):
    """

    :param data_file:
    :return:
    """
    dataset = pd.read_csv(data_file, sep=";")
    dataset_np = dataset.to_numpy()

    input_data = np.asarray(dataset_np[:, :54], dtype=np.float32)
    output_data = np.expand_dims(dataset_np[:, -1], axis=1)

    max_val = np.amax(input_data)
    min_val = np.amin(input_data)
    input_data /= np.float32(max_val - min_val)
    # shufle
    ranges = np.arange(0, len(input_data))
    np.random.shuffle(ranges)

    return input_data[ranges], output_data[ranges]


