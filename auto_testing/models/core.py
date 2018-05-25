import os
import math
import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


class AbstractModel(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, train, val, test, batch_size, num_epochs):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def save(self, file_path):
        pass

    @abstractmethod
    def load(self, file_path):
        pass

    @abstractmethod
    def close(self):
        pass


def train_model(xys, model_cls, batch_size, num_epochs, adaptation):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = xys

    model = model_cls(x_train, y_train, adaptation)
    model.fit((x_train, y_train), (x_val, y_val), (x_test, y_test),
              batch_size=batch_size, num_epochs=num_epochs)

    # y_test_pred = model.predict(x_test)
    #
    # mse = mean_squared_error(y_test, y_test_pred)
    # mae = mean_absolute_error(y_test, y_test_pred)

    # print('inside train_model, test mse: {mse}, test mae: {mae}'.format(mse=mse, mae=mae))

    return x_test, y_test, model


def train_rf(xys):
    x_train, x_val, x_test, y_train, y_val, y_test = xys

    rf = RandomForestRegressor(n_estimators=10,
                               criterion="mse",
                               max_depth=16,
                               min_samples_split=100,
                               min_samples_leaf=100,
                               n_jobs=os.cpu_count())
    rf.fit(x_train, y_train)
    y_test_pred = rf.predict(x_test)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    print('inside train_model, test mse: {mse}, test mae: {mae}'.format(mse=mse, mae=mae))

    return x_test, y_test, rf


def test_model(pack):
    x_test, y_test, model = pack

    y_test = y_test[:, :-1]

    y_test_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    test_maxes = np.max(y_test, axis=0)
    test_mins = np.min(y_test, axis=0)
    test_ranges = test_maxes - test_mins
    y_test_norm = y_test / test_ranges
    test_preds_norm = y_test_pred / test_ranges

    nrmse = math.sqrt(mean_squared_error(y_test_norm, test_preds_norm))

    return {'mse': mse, 'mae': mae, 'nrmse': nrmse, 'model': model}


def test_rf(pack):
    x_test, y_test, model = pack

    y_test_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    return {'mse': mse, 'mae': mae}
