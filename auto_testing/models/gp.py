from .core import AbstractModel
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor


class GaussianProcess(AbstractModel):
    def __init__(self, x, y, adaptation):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=7,
                                      normalize_y=True)
        self.model = MultiOutputRegressor(gp, n_jobs=1)
        # self.model = gp

    def fit(self, train, val, test, batch_size, num_epochs):
        x_train, y_train = train

        print('x_train shape: {}'.format(x_train.shape))

        x_val, y_val = val

        y_train = y_train[:, :-1]
        y_val = y_val[:, :-1]

        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_on_batch(self, x, batch_size=512):
        return self.model.predict(x)

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass

    def close(self):
        pass
