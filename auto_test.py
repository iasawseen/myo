import argparse
import random
import itertools
import numpy as np
import os
import math
import pickle
import lightgbm as lgb
import logging

from keras import regularizers, optimizers
from keras.layers import Input, Dense, Dropout, BatchNormalization, CuDNNGRU, Flatten
from keras.models import Model, load_model
from auto_testing.processing import transformers, preprocess
from auto_testing.utils import utility
from auto_testing.models import cnn, cnn_ad, core, rnn, gp
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sys import platform


RAW_DATA_DIR = '..{}input_data_raw'
PROCESSED_DATA_DIR = '..{}input_data_processed'

# RAW_DATA_DIR = '..{}raw_data'
# PROCESSED_DATA_DIR = '..{}processed_data'


if platform == 'win32':
    RAW_DATA_DIR = RAW_DATA_DIR.format('\\')
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR.format('\\')
else:
    RAW_DATA_DIR = RAW_DATA_DIR.format('/')
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR.format('/')


def generate_file_paths(dir_name, code_dict):

    filter_fns = (partial(utility.is_file_path_valid, code_index=code,
                          valid_codes=code_dict[code]) for code in code_dict)
    # filter_fns = (
    #     partial(utility.is_file_path_valid, code_index=0, valid_codes=['00', '01', '02']),
    #     # partial(utility.is_file_path_valid, code_index=0, valid_codes=['00', '01']),
    #     # partial(utility.is_file_path_valid, code_index=0, valid_codes=['00']),
    #     # partial(utility.is_file_path_valid, code_index=1, valid_codes=['0']),
    # )

    filter_file_paths_chain = (partial(utility.filter_file_paths, filter_fn=filter_fn)
                               for filter_fn in filter_fns)
    filtered_file_paths = utility.pipe(utility.list_data_file_paths(dir_name),
                                       filter_file_paths_chain)

    return filtered_file_paths


def transform_file_paths_train(file_paths):
    transform_fns = (
        partial(utility.read_from_file_path_with_encoder, label_encoder=utility.get_encoder_for_file_paths(file_paths)),
        transformers.rectify_x,
        transformers.low_pass_filter,
        partial(transformers.add_window_to_xy, window=128),
        transformers.reshape_x_for_dilated,
        transformers.mimic_old_y,
        transformers.shift,
        partial(transformers.split_by_chunks, val_test_size=0.20, chunks=12, overlapping=128),
        partial(transformers.compact_iterable, ratio=4)
    )

    funcs = (
        partial(transformers.process_iterable, func=partial(utility.pipe, funcs=transform_fns)),
        transformers.merge_xys
    )

    result = utility.pipe(file_paths, funcs=funcs)

    return result


def transform_file_paths_test(file_paths):
    transform_fns = (
        partial(utility.read_from_file_path_with_encoder,
                label_encoder=utility.get_encoder_for_file_paths(file_paths)),
        transformers.rectify_x,
        transformers.low_pass_filter,
        partial(transformers.add_window_to_xy, window=128),
        transformers.reshape_x_for_dilated,
        transformers.mimic_old_y,
        transformers.shift,
    )

    funcs = (
        partial(transformers.process_iterable, func=partial(utility.pipe, funcs=transform_fns)),
        transformers.merge_xys_test
    )

    result = utility.pipe(file_paths, funcs=funcs)

    return result


def transform_file_paths(file_paths):
    transform_fns = (
        utility.read_from_file_path,
        transformers.rectify_x,
        transformers.low_pass_filter,
        partial(transformers.add_window_to_xy, window=128),
        transformers.reshape_x_for_dilated,
        transformers.mimic_old_y,
        transformers.shift,
        # partial(transformers.compact_iterable, ratio=4),
        # lambda xy: (xy,)
    )

    funcs = (
        partial(transformers.process_iterable, func=partial(utility.pipe, funcs=transform_fns)),
        # transformers.merge_xys
    )

    result = utility.pipe(file_paths, funcs=funcs)

    return result


def train_test(xys, model_name, adaptation):
    model_dict = {'rnn': rnn.StandardRNN,
                  'dilated_cnn': cnn_ad.DilatedCNNAD,
                  'gp': gp.GaussianProcess}

    train_fn = partial(core.train_model, model_cls=model_dict[model_name],
                       batch_size=512, num_epochs=32, adaptation=adaptation)
    results = utility.pipe(xys, funcs=(train_fn, core.test_model))

    return results


def test_model(xy, model):
    x, y = xy

    y_preds = model.predict_on_batch(x)

    y = y[:, :-1]

    mse = mean_squared_error(y, y_preds)
    mae = mean_absolute_error(y, y_preds)

    test_maxes = np.max(y, axis=0)
    test_mins = np.min(y, axis=0)
    test_ranges = test_maxes - test_mins
    y_test_norm = y / test_ranges
    test_preds_norm = y_preds / test_ranges

    nrmse = math.sqrt(mean_squared_error(y_test_norm, test_preds_norm))

    results = {'mse': mse, 'mae': mae, 'nrmse': nrmse, 'model': model}
    return results


def train_against(filepaths, train_on, test_on, logger, model_name, adaptation):
    logger.info('train on: {}'.format(train_on))
    logger.info('test on: {}'.format(test_on))

    train_filter_fns = (
        partial(utility.is_file_path_valid, code_index=0, valid_codes=train_on),
    )
    train_filter_file_paths_chain = (partial(utility.filter_file_paths, filter_fn=filter_fn)
                                     for filter_fn in train_filter_fns)
    train_file_paths = utility.pipe(filepaths, train_filter_file_paths_chain)

    file_codes_training = ['_'.join(file_name.split('_')[-4:]) for file_name in train_file_paths]
    logger.info('training on file_names: {}'.format(', '.join(file_codes_training)))

    test_filter_fns = (
        partial(utility.is_file_path_valid, code_index=0, valid_codes=test_on),
    )

    test_filter_file_paths_chain = (partial(utility.filter_file_paths, filter_fn=filter_fn)
                                    for filter_fn in test_filter_fns)

    test_file_paths = utility.pipe(filepaths, test_filter_file_paths_chain)

    file_codes_test = ['_'.join(file_name.split('_')[-4:]) for file_name in test_file_paths]
    logger.info('testing on file_names: {}'.format(', '.join(file_codes_test)))

    results = utility.pipe(train_file_paths, funcs=(transform_file_paths_train,
                                                    partial(train_test,
                                                            model_name=model_name,
                                                            adaptation=adaptation)))
    test_msg = 'test mse: {mse:.3f}, ' \
               'test mae: {mae:.3f}, ' \
               'test nrmse: {nrmse:.5f}'.format(mse=results['mse'],
                                                mae=results['mae'],
                                                nrmse=results['nrmse'])

    logger.info(test_msg)

    model = results['model']
    oof_results = utility.pipe(test_file_paths, funcs=(transform_file_paths_test,
                                                       partial(test_model, model=model)))

    oof_msg = 'oof mse: {mse:.3f}, ' \
              'oof mae: {mae:.3f}, ' \
              'off nrmse: {nrmse:.5f}'.format(mse=oof_results['mse'],
                                              mae=oof_results['mae'],
                                              nrmse=oof_results['nrmse'])

    logger.info(oof_msg)

    results['model'].close()

    return {
        'test_mse': results['mse'],
        'test_mae': results['mae'],
        'test_rmse': results['nrmse'],
        'oof_mse': oof_results['mse'],
        'oof_mae': oof_results['mae'],
        'oof_rmse': oof_results['nrmse']
    }


def kfold(filepaths, chunks_qty, logger, model_name, adaptation):
    indices = list(range(len(filepaths)))
    random.seed(42)
    random.shuffle(indices)

    chunk_size = math.ceil(len(indices) / chunks_qty)
    chunk_indices = [indices[i * chunk_size: (i + 1) * chunk_size] for i in range(chunks_qty)]

    chunks = set(range(chunks_qty))

    test_mses = []
    test_maes = []
    test_rmses = []
    oof_mses = []
    oof_maes = []
    oof_rmses = []

    for combs_training in itertools.combinations(chunks, chunks_qty - 1):
        combs_predicting = chunks - set(combs_training)

        filepaths_oof = [filepaths[index] for chunk_index in combs_predicting
                         for index in chunk_indices[chunk_index]]

        filepaths_training = set(filepaths) - set(filepaths_oof)

        file_codes_training = ['_'.join(file_name.split('_')[-4:]) for file_name in filepaths_training]
        file_codes_oof      = ['_'.join(file_name.split('_')[-4:]) for file_name in filepaths_oof]

        logger.info('training of file_names: {}'.format(', '.join(file_codes_training)))
        logger.info('oof file_names: {}'.format(', '.join(file_codes_oof)))

        test_results = utility.pipe(filepaths_training, funcs=(transform_file_paths_train,
                                                               partial(train_test,
                                                                       model_name=model_name,
                                                                       adaptation=adaptation)))
        test_msg = 'test_mse: {mse:.3f}, ' \
                   'test_mae: {mae:.3f}, ' \
                   'test_nrmse: {nrmse:.5f}'.format(mse=test_results['mse'],
                                                    mae=test_results['mae'],
                                                    nrmse=test_results['nrmse'])
        logger.info(test_msg)

        test_mses.append(test_results['mse'])
        test_maes.append(test_results['mae'])
        test_rmses.append(test_results['nrmse'])

        model = test_results['model']

        oof_results = utility.pipe(filepaths_oof, funcs=(transform_file_paths_test,
                                                         partial(test_model, model=model)))

        off_msg = 'oof_mse: {mse:.3f}, ' \
                  'oof_mae: {mae:.3f}, ' \
                  'off_nrmse: {nrmse:.5f}'.format(mse=oof_results['mse'],
                                                  mae=oof_results['mae'],
                                                  nrmse=oof_results['nrmse'])
        logger.info(off_msg)

        oof_mses.append(oof_results['mse'])
        oof_maes.append(oof_results['mae'])
        oof_rmses.append(oof_results['nrmse'])

        test_results['model'].close()

    test_mse_avg = np.mean(np.array(test_mses))
    test_mae_avg = np.mean(np.array(test_maes))
    test_rmse_avg = np.mean(np.array(test_rmses))

    test_avg_msg = 'test_mse_avg: {mse:.3f}, ' \
                   'test_mae_avg: {mae:.3f}, ' \
                   'test_rmse_avg: {nrmse:.3f}'.format(mse=test_mse_avg,
                                                       mae=test_mae_avg,
                                                       nrmse=test_rmse_avg)
    logger.info(test_avg_msg)

    oof_mse_avg = np.mean(np.array(oof_mses))
    oof_mae_avg = np.mean(np.array(oof_maes))
    oof_rmse_avg = np.mean(np.array(oof_rmses))

    oof_avg_msg = 'oof_mse_avg: {mse:.3f}, ' \
                  'oof_mae_avg: {mae:.3f}, ' \
                  'oof_rmse_avg: {nrmse:.3f}'.format(mse=oof_mse_avg,
                                                     mae=oof_mae_avg,
                                                     nrmse=oof_rmse_avg)
    logger.info(oof_avg_msg)

    return {
        'test_mse_avg': test_mse_avg,
        'test_mae_avg': test_mae_avg,
        'test_rmse_avg': test_rmse_avg,
        'oof_mse_avg': oof_mse_avg,
        'oof_mae_avg': oof_mae_avg,
        'oof_rmse_avg': oof_rmse_avg
    }


def get_results(action_type, model_type, adaptation):
    logger = logging.getLogger('experiment_logger_{}_{}_adapt_{}'.format(action_type,
                                                                         model_type,
                                                                         adaptation))
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler('exp_all_type_{}_{}_adapt_{}_fixed_seed.log'.format(action_type, model_type, adaptation))
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    keys = ['test_mse_avg', 'test_mae_avg', 'test_rmse_avg',
            'oof_mse_avg', 'oof_mae_avg', 'oof_rmse_avg']

    code_dict = {0: ['00', '01', '02', '03', '04'], 1: [action_type]}

    person_results = {key: [] for key in keys}

    for i in range(2):
        logger.info('start iteration {}'.format(i))

        result = utility.pipe(PROCESSED_DATA_DIR, funcs=(partial(generate_file_paths, code_dict=code_dict),
                                                         partial(kfold, chunks_qty=5, logger=logger,
                                                                 model_name=model_type, adaptation=adaptation)))
        for key in result:
            person_results[key].append(result[key])

        logger.info('complete iteration {}'.format(i))

    test_mse_avg_mean = np.mean(np.array(person_results['test_mse_avg']))
    test_mae_avg_mean = np.mean(np.array(person_results['test_mae_avg']))
    test_rmse_avg_mean = np.mean(np.array(person_results['test_rmse_avg']))

    person_test_msg = 'test_mse_avg_mean: {mse:.3f}, ' \
                      'test_mae_avg_mean: {mae:.3f}, ' \
                      'test_rmse_avg_mean: {nrmse:.3f}'.format(mse=test_mse_avg_mean,
                                                               mae=test_mae_avg_mean,
                                                               nrmse=test_rmse_avg_mean)
    logger.info(person_test_msg)

    oof_mse_avg_mean = np.mean(np.array(person_results['oof_mse_avg']))
    oof_mae_avg_mean = np.mean(np.array(person_results['oof_mae_avg']))
    oof_rmse_avg_mean = np.mean(np.array(person_results['oof_rmse_avg']))

    person_oof_msg = 'oof_mse_avg_mean: {mse:.3f}, ' \
                     'oof_mae_avg_mean: {mae:.3f}, ' \
                     'oof_rmse_avg_mean: {nrmse:.3f}'.format(mse=oof_mse_avg_mean,
                                                             mae=oof_mae_avg_mean,
                                                             nrmse=oof_rmse_avg_mean)
    logger.info(person_oof_msg)


def get_results_inter_subjects(action_type, model_type, adaptation):
    logger = logging.getLogger('experiment_logger_inter_{}_{}_adapt_{}'.format(action_type,
                                                                               model_type,
                                                                               adaptation))
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler('exp_inter_sub_type_{}_{}_adapt_{}_fixed_seed.log'.format(action_type,
                                                                                       model_type,
                                                                                       adaptation))
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    subjects = ['00', '01', '02', '03', '04']
    code_dict = {0: subjects, 1: [action_type]}

    keys = ['test_mse', 'test_mae', 'test_rmse',
            'oof_mse', 'oof_mae', 'oof_rmse']

    avg_results = {key: [] for key in keys}

    for i in range(2):
        logger.info('start iteration {}'.format(i))

        person_results = {key: [] for key in keys}

        for test_on in itertools.combinations(subjects, 1):
            train_on = list(set(subjects) - set(test_on))

            result = utility.pipe(PROCESSED_DATA_DIR, funcs=(partial(generate_file_paths, code_dict=code_dict),
                                                             partial(train_against,
                                                                     train_on=train_on,
                                                                     test_on=test_on,
                                                                     logger=logger,
                                                                     model_name=model_type,
                                                                     adaptation=adaptation)))
            for key in result:
                person_results[key].append(result[key])

        test_msg = 'test_mse_avg: {mse:.3f}, ' \
                   'test_mae_avg: {mae:.3f}, ' \
                   'test_nrmse_avg: {nrmse:.5f}'.format(mse=np.mean(np.array(person_results['test_mse'])),
                                                        mae=np.mean(np.array(person_results['test_mae'])),
                                                        nrmse=np.mean(np.array(person_results['test_rmse'])))

        logger.info(test_msg)

        oof_msg = 'oof_mse_avg: {mse:.3f}, ' \
                  'oof_mae_avg: {mae:.3f}, ' \
                  'off_nrmse_avg: {nrmse:.5f}'.format(mse=np.mean(np.array(person_results['oof_mse'])),
                                                      mae=np.mean(np.array(person_results['oof_mae'])),
                                                      nrmse=np.mean(np.array(person_results['oof_rmse'])))

        logger.info(oof_msg)

        logger.info('complete iteration {}'.format(i))

        for key in keys:
            avg_results[key].append(np.mean(np.array(person_results[key])))

    test_msg = 'test_mse_avg_mean: {mse:.3f}, ' \
               'test_mae_avg_mean: {mae:.3f}, ' \
               'test_nrmse_avg_mean: {nrmse:.5f}'.format(mse=np.mean(np.array(avg_results['test_mse'])),
                                                         mae=np.mean(np.array(avg_results['test_mae'])),
                                                         nrmse=np.mean(np.array(avg_results['test_rmse'])))

    logger.info(test_msg)

    oof_msg = 'oof_mse_avg_mean: {mse:.3f}, ' \
              'oof_mae_avg_mean: {mae:.3f}, ' \
              'off_nrmse_avg_mean: {nrmse:.5f}'.format(mse=np.mean(np.array(avg_results['oof_mse'])),
                                                       mae=np.mean(np.array(avg_results['oof_mae'])),
                                                       nrmse=np.mean(np.array(avg_results['oof_rmse'])))

    logger.info(oof_msg)


def main(args):
    if args.pre_process:
        utility.pipe(RAW_DATA_DIR, funcs=(utility.list_data_file_paths, preprocess.launch_pre_process))

    # get_results(action_type='1', model_type='rnn', adaptation=False)
    # get_results(action_type='1', model_type='rnn', adaptation=False)
    # get_results(action_type='1', model_type='dilated_cnn', adaptation=False)
    # get_results(action_type='1', model_type='dilated_cnn', adaptation=True)

    # get_results_inter_subjects(action_type='0', model_type='dilated_cnn', adaptation=False)
    # get_results_inter_subjects(action_type='0', model_type='dilated_cnn', adaptation=True)
    # get_results(action_type='0', model_type='dilated_cnn', adaptation=False)
    # get_results(action_type='0', model_type='dilated_cnn', adaptation=True)

    get_results_inter_subjects(action_type='0', model_type='rnn', adaptation=False)
    get_results_inter_subjects(action_type='0', model_type='rnn', adaptation=True)
    get_results(action_type='0', model_type='rnn', adaptation=False)
    get_results(action_type='0', model_type='rnn', adaptation=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing hypothesis')
    parser.add_argument('--pre_process', type=bool, default=False, help='')

    arguments = parser.parse_args()
    main(arguments)

