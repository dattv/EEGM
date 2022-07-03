"""

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

def spect_heart(data_file):
    """

    :param data_file:
    :return:
    """
    assert os.path.isfile(data_file), "{} not exist".format(data_file)

    with open(data_file, 'r') as f:
        data = f.readlines()

        data = np.asarray(
            [d.replace('\n', '').split(',') for d in data],
            dtype=np.int32
        )

        # shuffle
        ranges = np.arange(0, len(data))
        np.random.shuffle(ranges)
        data = data[ranges]

    return data[:, 1:], data[:, 0]
