import pickle
import numpy as np
import os
from copy import deepcopy
from .alignment import Alignment
from ..utils.utility import flatten, read_from_file_path, write_to_file_path
from multiprocessing import Pool


def _clear_terminal_nones_from_iterables(*arrays):
    leading_end = 0
    tailing_start = 0

    length = len(arrays[0])
    assert all([len(array) == length for array in arrays]), 'arrays lengths are not equal'

    for i in range(length):
        if all(arr[i] is not None for arr in arrays):
            leading_end = i
            break

    for array in arrays:
        del array[:leading_end]

    for i in range(-1, -length, -1):
        if all(arr[i] is not None for arr in arrays):
            tailing_start = i
            break

    if tailing_start != -1:
        for array in arrays:
            del array[tailing_start + 1:]

    return arrays


def _trim_onside(fst, snd, eps, side, access_fn=lambda x, i: x[i]):
    fst = deepcopy(fst)
    snd = deepcopy(snd)

    if side == 'left':
        term_index = 0
    elif side == 'right':
        term_index = -1
    else:
        raise ValueError('Unknown trimming side')

    left_terminal = access_fn(fst, term_index)
    right_terminal = access_fn(snd, term_index)

    if abs(left_terminal - right_terminal) < eps:
        return fst, snd

    if side == 'left':
        flag = left_terminal < right_terminal
    elif side == 'right':
        flag = left_terminal > right_terminal
    else:
        raise ValueError('Unknown trimming side')

    for_trimming, untouched = (fst, snd) if flag else (snd, fst)

    if side == 'left':
        i = 0
        while abs(access_fn(for_trimming, i) - access_fn(untouched, 0)) > eps and i < len(for_trimming):
            i += 1
        del for_trimming[:i]
    elif side == 'right':
        i = len(for_trimming) - 1
        while abs(access_fn(for_trimming, i) - access_fn(untouched, -1)) > eps and i > 0:
            i -= 1
        del for_trimming[i + 1:]
    else:
        raise ValueError('Unknown trimming side')

    return fst, snd


def _trim_to_sync(fst, snd, eps, access_fn=lambda x, i: x[i]):
    fst_copy = deepcopy(fst)
    snd_copy = deepcopy(snd)

    fst_copy, snd_copy = _trim_onside(fst_copy, snd_copy, eps, side='left', access_fn=access_fn)
    fst_copy, snd_copy = _trim_onside(fst_copy, snd_copy, eps, side='right', access_fn=access_fn)
    return fst_copy, snd_copy


def _get_consecutive_none_lengths(iterable):
    none_lengths = []
    empty_start = 0
    empty_open = False

    for i, el in enumerate(iterable):
        if el is None:
            if not empty_open:
                empty_open = True
                empty_start = i
        else:
            if empty_open:
                empty_open = False
                empty_end = i
                none_lengths.append(empty_end - empty_start)

    if empty_open:
        none_lengths.append(len(iterable) - empty_start)

    return none_lengths


def _validate(iterable, max_nones_limit=10, ave_nones_limit=4):
    none_lengths = _get_consecutive_none_lengths(iterable)
    if len(none_lengths) == 0:
        return True
    return max(none_lengths) <= max_nones_limit and sum(none_lengths) / len(none_lengths) < ave_nones_limit


def _compress_angles(angles):
    if angles is None:
        return
    compressed = list()
    compressed.append(angles[0])

    for i in range(1, len(angles)):
        compressed.append(angles[i][0])
        compressed.append(angles[i][1][0])
        compressed.append(angles[i][2][0])

    return compressed


def _fill_angles_gap(angles, start, end):
    left_bound = angles[start - 1]
    right_bound = angles[end]
    length = end - start

    def get_filler(left, right, filler_length):
        assert len(left) == len(right)

        length_with_over = filler_length + 2

        linspaces = [np.linspace(left[i], right[i], length_with_over) for i in range(len(left))]
        filler = [[linspace[i] for linspace in linspaces] for i in range(length_with_over)][1:-1]

        assert filler_length == len(filler)

        return filler

    angles_filler = get_filler(left_bound, right_bound, length)
    angles[start: end] = angles_filler


def _interpolate_angles(angles):
    assert len(angles) != 0
    assert angles[0] is not None and angles[-1] is not None

    empty_start = 0
    empty_open = False

    for i, el in enumerate(angles):
        if el is None:
            if not empty_open:
                empty_open = True
                empty_start = i
        else:
            if empty_open:
                empty_open = False
                empty_end = i
                _fill_angles_gap(angles, empty_start, empty_end)

    return angles


def pre_process_single(file_path):
    emg_data, orientation_data, angles = read_from_file_path(file_path)

    while True:
        emg_length = len(emg_data)
        emg_data, angles = _trim_to_sync(emg_data, angles, 16, access_fn=lambda x, i: x[i][0])

        if emg_length == len(emg_data):
            break

    alignment = Alignment()

    emg_data_aligned_wt_angles, angles = alignment.align_fast(emg_data, angles, 10,
                                                              access_fn=lambda x, i: x[i][0], width=8)

    assert len(emg_data) == len(emg_data_aligned_wt_angles), \
        '{} has emg and angles with different lengths after alignment'.format(file_path)

    emg_data, angles = _clear_terminal_nones_from_iterables(emg_data, angles)

    assert _validate(angles, max_nones_limit=10, ave_nones_limit=2), \
        '{} did not pass validation'.format(file_path)

    emg_data = [el[1] if el is not None else el for el in emg_data]
    angles = [list(flatten(_compress_angles(el[1]))) if el is not None else el for el in angles]

    interpolated_angles = _interpolate_angles(angles)

    x = np.array(emg_data)
    y = np.array(interpolated_angles)

    parent_dir, filename = os.path.split(file_path)

    new_parent_dir = parent_dir.replace('raw', 'processed')
    if not os.path.exists(new_parent_dir):
        os.mkdir(new_parent_dir)

    new_filename = filename.replace('raw', 'processed')
    new_file_path = os.path.join(new_parent_dir, new_filename)

    write_to_file_path(new_file_path, (x, y))


def launch_pre_process(file_paths):
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(pre_process_single, file_paths)



