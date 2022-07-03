"""
sonar.all-data
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


def sonar(data_file):
    """

    :param data_file:
    :return:
    """

    def transform_input(str):
        if str == 'R':
            return 0.
        else:
            return 1.

    with open(data_file, mode='r') as f:
        data = f.read().split('\n')[:-1]

    dataset_np = []
    for line in data:
        temp = line.split(',')
        input = np.asarray(list(map(float, temp[:-1])))
        output = np.asarray(list(map(transform_input, temp[-1])))
        temp = np.concatenate([input, output])
        dataset_np.append(temp)

    # dataset_np = np.asarray(dataset_np)
    dataset_np = np.vstack(dataset_np)
    input_data = dataset_np[:, :-1]

    input_data = sc.fit_transform(input_data)
    output_data = np.expand_dims(dataset_np[:, -1], axis=1)

    # shufle
    ranges = np.arange(0, len(input_data))
    np.random.shuffle(ranges)

    return input_data[ranges], output_data[ranges]


if __name__ == '__main__':
    data = sonar('../data/sonar/sonar.all-data')
    print('djkfjdl')
