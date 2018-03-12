import numpy as np


def average_filter(array, window, mode='same'):
    columns = []
    for j in range(array.shape[1]):
        column = array[:, j]
        column = np.convolve(column, np.ones((window,)) / window, mode=mode)
        column = np.reshape(column, newshape=(column.shape[0], 1))
        columns.append(column)
    return np.concatenate(columns, axis=1)


def average_filter_emgs_angles(xy, emgs_window=4, angles_window=4):
    x, y = xy
    x_filtered = average_filter(x, window=emgs_window)
    y_filtered = average_filter(y, window=angles_window)
    return x_filtered, y_filtered


def average_filter_angles(xy, window):
    return average_filter_emgs_angles(xy, emgs_window=1, angles_window=window)


def add_window_to_xy(xy, window):
    x, y = xy
    x_windowed = list()
    y_windowed = list()
    for i in range(0, x.shape[0] - window):
        x_indices = list(range(i, i + window))
        x_windowed.append(x[x_indices, :])
        y_windowed.append(y[x_indices[-1], :])

    return np.array(x_windowed), np.array(y_windowed)


def compact_xy(xy, ratio):
    x, y = xy
    offset = ratio
    indices_to_save = list(range(0, x.shape[0], offset))
    indices_to_delete = list(set(range(x.shape[0])) - set(indices_to_save))
    x_compacted = np.delete(x, indices_to_delete, axis=0)
    y_compacted = np.delete(y, indices_to_delete, axis=0)
    return x_compacted, y_compacted

