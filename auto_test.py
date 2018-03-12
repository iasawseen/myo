import os
from auto_testing.processing.preprocess import launch_pre_process
from auto_testing.utils.utility import list_data_file_paths, is_file_path_valid, filter_file_paths, pipe
from functools import partial


RAW_DATA_DIR = '..\\raw_data'
PROCESSED_DATA_DIR = '..\\processed_data'


def filter_dir(dir_name):
    filter_fns = (
        partial(is_file_path_valid, code_index=0, valid_codes=['00']),
        partial(is_file_path_valid, code_index=1, valid_codes=['0']),
    )
    filter_file_paths_chain = (partial(filter_file_paths, filter_fn=filter_fn)
                               for filter_fn in filter_fns)
    filtered_file_paths = pipe(list_data_file_paths(dir_name),
                               *filter_file_paths_chain)
    return filtered_file_paths


if __name__ == '__main__':
    print(filter_dir(RAW_DATA_DIR))

