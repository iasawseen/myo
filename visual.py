import auto_test
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
import math
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def low_pass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    datas = [data[:, i] for i in range(data.shape[1])]
    data_low_passes = [lfilter(b, a, data_) for data_ in datas]
    x_low_pass = np.vstack(data_low_passes).T
    return x_low_pass


file_paths = auto_test.generate_file_paths(auto_test.PROCESSED_DATA_DIR)

print(file_paths[1])
xy = auto_test.utility.read_from_file_path(file_paths[1])
# xy = auto_test.transformers.rectify_x(xy)
x, y = xy

order = 6
fs = 200       # sample rate, Hz
cutoff = 10  # desired cutoff frequency of the filter, Hz


with open('ys_list_15.pickle', 'rb') as handle:
    ys, y_preds = pickle.load(handle)

angle_ranges = {0: (0, 6000),
                3: (6000, 12000),
                6: (12000, 18000),
                9: (18000, 24000),
                12: (24000, 30000)}

for y, y_pred in zip(ys, y_preds):
    angle_index = 6
    start, end = angle_ranges[angle_index]
    y = y[start: end, angle_index]
    y_pred = y_pred[start: end, angle_index]

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    test_maxes = np.max(y, axis=0)
    test_mins = np.min(y, axis=0)
    test_ranges = test_maxes - test_mins
    y_test_norm = y / test_ranges
    test_preds_norm = y_pred / test_ranges

    nrmse = math.sqrt(mean_squared_error(y_test_norm, test_preds_norm))

    print('meta test mse: {mse:.3f}, '
          'meta test mae: {mae:.3f}, '
          'meta test nrmse {nrmse:.5f}\n'.format(mse=mse, mae=mae, nrmse=nrmse))


def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    test_maxes = np.max(y_true, axis=0)
    test_mins = np.min(y_true, axis=0)
    test_ranges = test_maxes - test_mins
    y_test_norm = y_true / test_ranges
    test_preds_norm = y_pred / test_ranges

    nrmse = math.sqrt(mean_squared_error(y_test_norm, test_preds_norm))

    return mse, mae, nrmse


def get_hists(y_trues, y_preds):
    ranges = ((0, 6000), (6000, 12000),
              (12000, 18000), (18000, 24000),
              (24000, 30000), (30000, 42000), (42000, 47000))

    metrics = {'mse': [], 'mae': [], 'nrmse': []}

    for y, y_pred in zip(y_trues, y_preds):

        for range_ in ranges:
            start, end = range_
            y_ = y[start: end, :]
            y_pred_ = y_pred[start: end, :]

            results = get_metrics(y_, y_pred_)

        # results = get_metrics(y, y_pred)

            metrics['mse'].append(results[0])
            metrics['mae'].append(results[1])
            metrics['nrmse'].append(results[2])

        # print('meta test mse: {mse:.3f}, '
        #       'meta test mae: {mae:.3f}, '
        #       'meta test nrmse {nrmse:.5f}\n'.format(mse=mse, mae=mae, nrmse=nrmse))

    return metrics['mse'], metrics['mae'], metrics['nrmse']


hists = get_hists(ys, y_preds)


# plt.hist(hists[0], bins=16)
# plt.hist(hists[1], bins=32)
# plt.hist(hists[2], bins=24)
# print(len(hists[0]))
# hists_0 = pd.Series(hists[0], name='MSE')
# sns.distplot(hists_0)
# ax = sns.kdeplot(hists_0, shade=True)
# ax.set_yticks([])
# plt.xlabel('MSE')

# plt.show()


# angle_index = 6
# index = 6
# start, end = angle_ranges[angle_index]
#
# start_offset = 1000
# end_offset = 2000
#
# plt.plot(ys[index][start + start_offset: end - end_offset, angle_index])
# plt.plot(butter_lowpass_filter(y_preds[index][start + start_offset: end - end_offset, angle_index], cutoff, fs, order))
# plt.ylabel('угол, °')
# plt.xlabel('временная шкала')
# plt.show()


index = 10
#
# # plt.plot(x[6000: 6600, :])
# plt.plot(low_pass_filter(np.abs(x[0: 10000, :]), cutoff, fs, order)[5000: 5400, 1])
# plt.plot(np.abs(x[0: 10000, :])[5000: 5400, 1])
plt.plot(x[0: 10000, :][5000: 5400, 1])

plt.xlabel('временная шкала, 200 Гц')
plt.ylabel('активация мыщц')
plt.show()
