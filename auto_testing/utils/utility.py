import collections
import numpy as np
import os


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


def pipe(initial_data, *funcs):
    result_data = initial_data
    for index, func in enumerate(funcs):
        result_data = func(result_data)

    return result_data
