from abc import ABCMeta, abstractmethod


class AbstractModel(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, train, val, batch_size, num_epochs):
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
