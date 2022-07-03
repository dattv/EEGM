"""

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import os


# eegm_error = np.load('./eegm_err.npy')
# sgd_error = np.load('./mse_err.npy')
# plt.plot(eegm_error, color='g', label='eegm_err')
# plt.plot(sgd_error, color='b', label='mse_err')
# plt.title('comparison between EEGM and SGD')
# plt.xlabel('step')
# plt.ylabel('mean square error')
# plt.grid()
# plt.legend()
# plt.show()

def visualization(input_files, x_tick='step', y_tick='ACC', color=None, save=True, title='history', labels=None):
    """

    :param input_files:
    :return:
    """
    if color:
        assert len(input_files) == len(color), "len of input file must be equal len of color: {} -  {}".format(
            len(input_files), len(color))
    for i, file in enumerate(input_files):
        plt.plot(np.load(file),
                 color=color[i] if color else np.random.rand(len(input_files), 3),
                 label=labels[i])
    plt.title(title)
    plt.xlabel(x_tick)
    plt.ylabel(y_tick)
    # plt.yscale("log")
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.show()
    # if len(input_files) == 1:
    #     plt.savefig(input_files[0].split('.')[0])
    # else:
    #     plt.savefig(os.path.split(input_files[0])[0])


if __name__ == '__main__':
    # visualization(['./logs/custom_entropy_error_fun.npy',
    #                './logs/custom_mse_fun.npy'], color=['r', 'b'])
    # folder_name = '/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/EEGM/download_resulst/logs_sonar/sonar_reg3'
    # visualization(input_files=[os.path.join(folder_name, 'mse_loss_train.npy'),
    #                            os.path.join(folder_name, 'accuracy' + '_train.npy'),
    #                            os.path.join(folder_name, 'accuracy' + '_test.npy')
    #                            ],
    #               color=['r', 'b', 'g'], title=os.path.split(folder_name)[-1])

    folder_name = '/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/EEGM/logs/sumer_ville_happiness_surface_data_continuous_l12_0'
    visualization(input_files=[os.path.join(folder_name, 'accuracy_train.npy'),
                               os.path.join(folder_name[:-1]+'1', 'accuracy_train' + '.npy'),
                               os.path.join(folder_name[:-1]+'2', 'accuracy_train' + '.npy')
                               ],
                  color=['r', 'b', 'g'], title='sumer_ville_happiness_surface_data_continuous_l12',
                  labels=['accuracy of SEEGML1/2', 'accuracy of EEGML2', 'accuracy of EEGML1/2'])
