from copy import deepcopy


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


def trim_to_sync(fst, snd, eps, access_fn=lambda x, i: x[i]):
    fst_copy = deepcopy(fst)
    snd_copy = deepcopy(snd)

    fst_copy, snd_copy = _trim_onside(fst_copy, snd_copy, eps, side='left', access_fn=access_fn)
    fst_copy, snd_copy = _trim_onside(fst_copy, snd_copy, eps, side='right', access_fn=access_fn)
    return fst_copy, snd_copy


def pre_process():
    pass
