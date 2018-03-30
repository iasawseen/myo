from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def train_model(xys, model_cls, batch_size, num_epochs):
    x_train, x_val, x_test, y_train, y_val, y_test = xys

    model = model_cls(x_train, y_train)
    model.fit((x_train, y_train), (x_val, y_val), (x_test, y_test),
              batch_size=batch_size, num_epochs=num_epochs)

    y_test_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    print('inside train_model, test mse: {mse}, test mae: {mae}'.format(mse=mse, mae=mae))

    return x_test, y_test, model


def test_model(pack):
    x_test, y_test, model = pack

    y_test_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    return {'mse': mse, 'mae': mae}

