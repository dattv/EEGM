"""

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

def sumer_ville_happiness_surface_data(data_file):
    """

    :param data_file:
    :return:
    """

    dataset = pd.read_csv("https://raw.githubusercontent.com/SahilSinhaLpu/Machine-Learning/master/Datasets/SomvervilleHappines.csv")

    dataset_np = dataset.to_numpy()

    input_data = dataset_np[:, 1:]

    input_data = sc.fit_transform(input_data)
    output_data = np.expand_dims(dataset_np[:, 0], axis=1)

    # shufle
    ranges = np.arange(0, len(input_data))
    np.random.shuffle(ranges)

    return input_data[ranges], output_data[ranges]

if __name__ == '__main__':
    data = sumer_ville_happiness_surface_data('/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/EEGM/data/SomervilleHappinessSurvey2015/SomervilleHappinessSurvey2015.csv')
    print('djkfjdl')