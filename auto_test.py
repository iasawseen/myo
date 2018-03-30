import argparse
from auto_testing.processing import transformers, preprocess
from auto_testing.utils import utility
from auto_testing.models import cnn, core
from functools import partial

from sys import platform


RAW_DATA_DIR = '..{}input_data_raw'
PROCESSED_DATA_DIR = '..{}processed_data'


if platform == 'win32':
    RAW_DATA_DIR = RAW_DATA_DIR.format('\\')
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR.format('\\')
else:
    RAW_DATA_DIR = RAW_DATA_DIR.format('/')
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR.format('/')


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


def transform_file_paths(file_paths):
    transform_fns = (
        utility.read_from_file_path,
        # transformers.rectify_x,
        # partial(transformers.average_filter_emgs_angles, emgs_window=16, angles_window=8),
        partial(transformers.average_filter_angles, window=8),
        partial(transformers.add_window_to_xy, window=128),
        transformers.reshape_x_for_dilated,
        partial(transformers.split_by_chunks, val_test_size=0.25, chunks=10, overlapping=134),
        partial(transformers.compact_iterable, ratio=4)
    )

    funcs = (
        partial(transformers.process_iterable, func=partial(utility.pipe, funcs=transform_fns)),
        transformers.merge_xys
    )

    result = utility.pipe(file_paths, funcs=funcs)

    return result


def train_test(xys):
    train_fn = partial(core.train_model, model_cls=cnn.DilatedCNN, batch_size=1024, num_epochs=64)
    test_metrics = utility.pipe(xys, funcs=(train_fn, core.test_model))
    return test_metrics


def main(args):
    if args.pre_process:
        utility.pipe(RAW_DATA_DIR, funcs=(utility.list_data_file_paths, preprocess.launch_pre_process))
    metrics = utility.pipe(PROCESSED_DATA_DIR, funcs=(utility.list_data_file_paths, transform_file_paths, train_test))
    print('test mse: {mse}, test mae: {mae}'.format(mse=metrics['mse'], mae=metrics['mae']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing hypothesis')
    parser.add_argument('--pre_process', type=bool, default=False, help='')

    arguments = parser.parse_args()
    main(arguments)

