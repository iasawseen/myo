import collections
import numpy as np
import pickle
import os


def pipe(initial_data, funcs=None):
    if funcs is None:
        funcs = []
    result_data = initial_data
    for index, func in enumerate(funcs):
        result_data = func(result_data)

    return result_data


def flatten(iterable):
    for item in iterable:
        if not isinstance(item, collections.Iterable):
            yield item
        else:
            for sub_item in flatten(item):
                yield sub_item


def list_data_file_paths(parent_dir_path):
    file_paths = []
    for directory in os.listdir(parent_dir_path):
        dir_path = os.path.join(parent_dir_path, directory)
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            file_paths.append(file_path)
    return file_paths


def is_file_path_valid(file_path, code_index, valid_codes=None):
    _, file_name = os.path.split(file_path)
    file_path_codes = file_name.split('.')[0].split('_')[-4:]
    status = not valid_codes or file_path_codes[code_index] in valid_codes
    return status


def filter_file_paths(file_paths, filter_fn):
    filtered_file_paths = list(filter(filter_fn, file_paths))
    return filtered_file_paths


def get_encoder_for_file_paths(file_paths):
    file_coder = dict()

    def get_code(file):
        _, file_name = os.path.split(file)
        file_path_codes = file_name.split('.')[0].split('_')[-4:]
        # return '_'.join(file_path_codes[0])
        return '_'.join(file_path_codes)

    index = 0
    codes = set()
    for file_path in file_paths:
        init_len = len(codes)
        # print(file_path)
        new_code = get_code(file_path)
        # print(new_code, index)
        codes.add(new_code)
        if len(codes) > init_len:
            file_coder[new_code] = index
            index += 1

    def inner(file):
        return file_coder[get_code(file)]

    return inner


def read_from_file_path(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def read_from_file_path_with_encoder(file_path, label_encoder):
    with open(file_path, 'rb') as handle:
        x, y = pickle.load(handle)
        y_label = label_encoder(file_path)
        return x, (y, y_label)


def write_to_file_path(file_path, data):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)
