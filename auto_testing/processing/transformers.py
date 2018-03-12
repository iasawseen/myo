import numpy as np
import random


def average_filter(array, window, mode='same'):
    columns = []
    for j in range(array.shape[1]):
        column = array[:, j]
        column = np.convolve(column, np.ones((window,)) / window, mode=mode)
        column = np.reshape(column, newshape=(column.shape[0], 1))
        columns.append(column)
    return np.concatenate(columns, axis=1)


def rectify_x(xy):
    x, y = xy
    x_rectified = np.abs(x)
    return x_rectified, y


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


def reshape_x_for_dilated(xy):
    x, y = xy
    x_reshaped = np.reshape(x, newshape=(x.shape[0], x.shape[1], 1, x.shape[2]))
    return x_reshaped, y


def compact_iterable(iterable, ratio):
    result = []
    offset = ratio
    length = iterable[0].shape[0]
    indices_to_save = list(range(0, length, offset))
    indices_to_delete = list(set(range(length)) - set(indices_to_save))
    for el in iterable:
        result.append(np.delete(el, indices_to_delete, axis=0))
    return result


def process_iterable(iterable, func):
    results = [func(el) for el in iterable]
    return results


def merge_xys(xys):
    # xys = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
    result = [[] for _ in range(len(xys[0]))]

    for xy in xys:
        for index, element in enumerate(xy):
            result[index].append(element)

    merged = [np.concatenate(iterable) for iterable in result]

    return merged


def split_by_chunks(xy, val_test_size=0.25, chunks=20, overlapping=32):
    x, y = xy
    chunk_size = int(x.shape[0] * val_test_size / chunks)
    offset = int(x.shape[0] / chunks)
    borders = []
    x_vals = []
    y_vals = []
    x_tests = []
    y_tests = []

    for i in range(0, x.shape[0], offset):
        start_index = random.randint(i, i + offset - chunk_size)
        end_index = start_index + chunk_size
        borders.append((start_index, end_index))

    shuffled_borders = list(borders)
    random.shuffle(shuffled_borders)

    for border_pair in borders[: len(shuffled_borders) // 2]:
        x_vals.append(x[border_pair[0] + overlapping: border_pair[1] - overlapping, :])
        y_vals.append(y[border_pair[0] + overlapping: border_pair[1] - overlapping, :])

    for border_pair in borders[len(shuffled_borders) // 2:]:
        x_tests.append(x[border_pair[0] + overlapping: border_pair[1] - overlapping, :])
        y_tests.append(y[border_pair[0] + overlapping: border_pair[1] - overlapping, :])

    x_val = np.concatenate(x_vals)
    y_val = np.concatenate(y_vals)
    x_test = np.concatenate(x_tests)
    y_test = np.concatenate(y_tests)

    indices_to_delete = []

    for border_pair in borders:
        indices_to_delete.extend(list(range(border_pair[0], border_pair[1])))
    x_train = np.delete(x, indices_to_delete, axis=0)
    y_train = np.delete(y, indices_to_delete, axis=0)

    return x_train, x_val, x_test, y_train, y_val, y_test
