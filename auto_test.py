import os
from auto_testing.processing.preprocess import launch_pre_process
from auto_testing.utils import utility
from auto_testing.processing import transformers
from auto_testing.models import core, rnn
from functools import partial


RAW_DATA_DIR = '..\\raw_data'
PROCESSED_DATA_DIR = '..\\processed_data'


def generate_file_paths(dir_name):
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


def train_test(xys):
    train_fn = partial(core.train_model, model_cls=rnn.StandardRNN, batch_size=1024, num_epochs=64)
    test_metrics = utility.pipe(xys, funcs=(train_fn, core.test_model))
    return test_metrics


if __name__ == '__main__':
    metrics = utility.pipe(PROCESSED_DATA_DIR, funcs=(generate_file_paths, transform, train_test))
    print('metrics: {}'.format(metrics))
