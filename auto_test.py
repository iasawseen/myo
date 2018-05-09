import argparse
import random
import itertools
import numpy as np
import os
import math
import pickle
import lightgbm as lgb


from keras import regularizers, optimizers
from keras.layers import Input, Dense, Dropout, BatchNormalization, CuDNNGRU, Flatten
from keras.models import Model, load_model
from auto_testing.processing import transformers, preprocess
from auto_testing.utils import utility
from auto_testing.models import cnn, core
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sys import platform


# RAW_DATA_DIR = '..{}input_data_raw'
# PROCESSED_DATA_DIR = '..{}input_data_processed'

RAW_DATA_DIR = '..{}raw_data'
PROCESSED_DATA_DIR = '..{}processed_data'


if platform == 'win32':
    RAW_DATA_DIR = RAW_DATA_DIR.format('\\')
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR.format('\\')
else:
    RAW_DATA_DIR = RAW_DATA_DIR.format('/')
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR.format('/')


def generate_file_paths(dir_name):
    filter_fns = (
        partial(utility.is_file_path_valid, code_index=0, valid_codes=['00', '01', '02']),
        # partial(utility.is_file_path_valid, code_index=1, valid_codes=['0']),
        # partial(utility.is_file_path_valid, code_index=2, valid_codes=['1', '2', '3']),
    )
    filter_file_paths_chain = (partial(utility.filter_file_paths, filter_fn=filter_fn)
                               for filter_fn in filter_fns)
    filtered_file_paths = utility.pipe(utility.list_data_file_paths(dir_name),
                                       filter_file_paths_chain)

    return filtered_file_paths


def transform_file_paths_train(file_paths):
    transform_fns = (
        utility.read_from_file_path,
        transformers.rectify_x,
        transformers.low_pass_filter,
        partial(transformers.add_window_to_xy, window=128),
        transformers.reshape_x_for_dilated,
        # transformers.mimic_old_y,
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


def transform_file_paths(file_paths):
    transform_fns = (
        utility.read_from_file_path,
        transformers.rectify_x,
        transformers.low_pass_filter,
        partial(transformers.add_window_to_xy, window=128),
        transformers.reshape_x_for_dilated,
        # transformers.mimic_old_y,
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


def train_test(xys):
    train_fn = partial(core.train_model, model_cls=cnn.DilatedCNN,
                       batch_size=256, num_epochs=32)
    results = utility.pipe(xys, funcs=(train_fn, core.test_model))
    # train_fn = partial(core.train_rf)
    # test_metrics = utility.pipe(xys, funcs=(train_fn, core.test_rf))
    return results


def stacking(filepaths, chunks_qty=4):
    indices = list(range(len(filepaths)))
    random.shuffle(indices)
    chunk_size = len(indices) // chunks_qty
    chunk_indices = [indices[i: i + chunk_size] for i in range(chunks_qty)]

    chunks = set(range(chunks_qty))

    ys = []
    y_preds = []

    for combs_training in itertools.combinations(chunks, chunks_qty - 1):
        combs_predicting = chunks - set(combs_training)

        print(combs_training, combs_predicting)

        filepaths_predicting = [filepaths[index] for chunk_index in combs_predicting
                                for index in chunk_indices[chunk_index]]

        filepaths_training = set(filepaths) - set(filepaths_predicting)

        print('training')
        print(filepaths_training)
        print()
        print('predicting')
        print(filepaths_predicting)
        print()

        results = utility.pipe(filepaths_training, funcs=(transform_file_paths_train, train_test))

        print('\ntest mse: {mse:.3f},'
              'test mae: {mae:.3f}, '
              'test nrmse: {nrmse:.5f}\n'.format(mse=results['mse'], mae=results['mae'], nrmse=results['nrmse']))

        y_pred_ = []

        # [(x, y)] = transform_file_paths(filepaths_predicting)

        # x_chunk_size = 10000
        # for i in range(0, x.shape[0], x_chunk_size):
        #     y_pred.append(results['model'].predict(x[i: i + x_chunk_size]))

        xys = transform_file_paths(filepaths_predicting)

        y_ = []
        for (x, y) in xys:
            y_.append(y)
            y_pred_.append(results['model'].predict(x))

        y_pred = np.concatenate(y_pred_)
        y = np.concatenate(y_)
        y_preds.extend(y_pred_)
        ys.extend(y_)

        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        test_maxes = np.max(y, axis=0)
        test_mins = np.min(y, axis=0)
        test_ranges = test_maxes - test_mins
        y_test_norm = y / test_ranges
        test_preds_norm = y_pred / test_ranges

        nrmse = math.sqrt(mean_squared_error(y_test_norm, test_preds_norm))

        print('meta test mse: {mse:.3f}, '
              'meta test mae: {mae:.3f}, '
              'meta test nrmse {nrmse:.5f}\n'.format(mse=mse, mae=mae, nrmse=nrmse))

        # ys.append(y)
        # y_preds.append(y_pred)
        results['model'].close()

    print('ys_len: ', len(ys))
    print('y_preds_len: ', len(y_preds))

    with open('ys_list.pickle', 'wb') as handle:
        pickle.dump((ys, y_preds), handle)


    # with open('ys_list.pickle', 'rb') as handle:
    #     ys, y_preds = pickle.load(handle)
    #
    # indices = list(range(len(ys)))
    # random.shuffle(indices)
    #
    # x_train = []
    # x_test = []
    # y_train = []
    # y_test = []
    #
    # print('indices len', len(indices))
    #
    # for i in indices[:int(len(indices) * 0.8)]:
    #     x_train.append(y_preds[i])
    #     y_train.append(ys[i])
    #
    # for i in indices[int(len(indices) * 0.8):]:
    #     x_test.append(y_preds[i])
    #     y_test.append(ys[i])
    #
    # x_train = np.concatenate(x_train)
    # y_train = np.concatenate(y_train)
    # x_test = np.concatenate(x_test)
    # y_test = np.concatenate(y_test)
    #
    # print('shapes')
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    #
    # # ys = []
    # # y_preds = []
    #
    # # ys = np.concatenate(ys)
    # # y_preds = np.concatenate(y_preds)
    #
    # # print(ys.shape, y_preds.shape)
    #
    # # x_train, x_test, y_train, y_test = train_test_split(y_preds, ys, test_size=0.2)
    # # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)
    #
    # # def get_model(features, pred_length):
    # #         inputs = Input(shape=(features,))
    # #         layer = Dense(32, input_shape=(features,))(inputs)
    # #         layer = BatchNormalization()(layer)
    # #         layer = Dropout(0.5)(layer)
    # #         layer = Dense(32, activation='relu')(layer)
    # #         layer = BatchNormalization()(layer)
    # #         preds = Dense(pred_length, activation='linear')(layer)
    # #         model = Model(inputs=inputs, outputs=preds)
    # #         optimer = optimizers.Adam(lr=0.001)
    # #         model.compile(optimizer=optimer, loss='mse', metrics=['mae'])
    # #         return model
    #
    # # model = RandomForestRegressor(n_estimators=100,
    # #                               criterion="mse",
    # #                               max_depth=64,
    # #                               min_samples_split=100,
    # #                               min_samples_leaf=100,
    # #                               n_jobs=os.cpu_count(),
    # #                               verbose=True)
    #
    # # print('\nmeta model is learning...\n')
    #
    # # model = get_model(x_train.shape[1], y_train.shape[1])
    # # model.fit(x=x_train, y=y_train, batch_size=256, epochs=32,
    # #           shuffle=True, validation_data=(x_val, y_val))
    #
    # # model.fit(X=x_train, y=y_train)
    # # y_test_pred = model.predict(x_test)
    #
    # y_test_preds = []
    #
    # for i in range(y_train.shape[1]):
    #     lgb_train = lgb.Dataset(x_train, y_train[:, i].ravel())
    #     lgb_eval = lgb.Dataset(x_test, y_test[:, i].ravel(), reference=lgb_train)
    #
    #     params = {
    #         'task': 'train',
    #         'boosting_type': 'gbdt',
    #         'objective': 'regression',
    #         'metric': {'l2'},
    #         'max_depth': 7,
    #         'num_leaves': 32,
    #         'learning_rate': 0.005,
    #         'feature_fraction': 0.9,
    #         'bagging_fraction': 0.8,
    #         'bagging_freq': 5,
    #         'verbose': 0
    #     }
    #
    #     print('Start training...')
    #     gbm = lgb.train(params,
    #                     lgb_train,
    #                     num_boost_round=2000,
    #                     valid_sets=lgb_eval,
    #                     # verbose_eval=False,
    #                     early_stopping_rounds=20)
    #
    #     y_test_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    #     y_test_preds.append(y_test_pred.reshape((-1, 1)))
    #
    # y_test_pred = np.hstack(y_test_preds)
    #
    # with open('ys_list_meta.pickle', 'wb') as handle:
    #     print('meta len: ', len(y_test), ', ', len(y_test_pred))
    #     pickle.dump((y_test, y_test_pred), handle)
    #
    # print(y_test_pred.shape)
    # mse = mean_squared_error(y_test, y_test_pred)
    # mae = mean_absolute_error(y_test, y_test_pred)
    #
    # test_maxes = np.max(y_test, axis=0)
    # test_mins = np.min(y_test, axis=0)
    # test_ranges = test_maxes - test_mins
    # y_test_norm = y_test / test_ranges
    # test_preds_norm = y_test_pred / test_ranges
    #
    # nrmse = math.sqrt(mean_squared_error(y_test_norm, test_preds_norm))
    #
    # print('\nmeta test mse: {mse:.3f}, '
    #       'meta test mae: {mae:.3f}, '
    #       'meta test nrmse {nrmse:.5f}\n'.format(mse=mse, mae=mae, nrmse=nrmse))


def main(args):
    if args.pre_process:
        utility.pipe(RAW_DATA_DIR, funcs=(utility.list_data_file_paths, preprocess.launch_pre_process))

    utility.pipe(PROCESSED_DATA_DIR, funcs=(generate_file_paths, stacking))
    # stacking()
    # metrics = utility.pipe(PROCESSED_DATA_DIR, funcs=(generate_file_paths, transform_file_paths, train_test))

    # print('test mse: {mse}, test mae: {mae}'.format(mse=metrics['mse'], mae=metrics['mae']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing hypothesis')
    parser.add_argument('--pre_process', type=bool, default=False, help='')

    arguments = parser.parse_args()
    main(arguments)

