import os
from auto_testing.processing.preprocess import launch_pre_process
# from auto_testing.utils.utility import list_data_file_paths, is_file_path_valid, filter_file_paths, pipe
from auto_testing.utils import utility
from auto_testing.processing import transformers
from functools import partial


RAW_DATA_DIR = '..\\raw_data'
PROCESSED_DATA_DIR = '..\\processed_data'


def filter_dir(dir_name):
    filter_fns = (
        partial(utility.is_file_path_valid, code_index=0, valid_codes=['00']),
        partial(utility.is_file_path_valid, code_index=1, valid_codes=['0']),
    )
    filter_file_paths_chain = (partial(utility.filter_file_paths, filter_fn=filter_fn)
                               for filter_fn in filter_fns)
    filtered_file_paths = utility.pipe(utility.list_data_file_paths(dir_name),
                                       filter_file_paths_chain)
    return filtered_file_paths


def transform(file_paths):
    transform_fns = (
        utility.read_from_file_path,
        partial(transformers.average_filter_angles, window=8),
        partial(transformers.add_window_to_xy, window=64),
        partial(transformers.split_by_chunks, val_test_size=0.25, chunks=20, overlapping=64),
        partial(transformers.compact_iterable, ratio=4)
    )

    funcs = (
        partial(transformers.process_iterable, func=partial(utility.pipe, funcs=transform_fns)),
        transformers.merge_xys
    )

    result = utility.pipe(file_paths, funcs=funcs)

    return result


if __name__ == '__main__':
    utility.pipe(PROCESSED_DATA_DIR, funcs=(filter_dir, transform))
