from .core import AbstractModel
from keras.layers import Input, Dense, Dropout, BatchNormalization, CuDNNGRU, Flatten
from keras.models import Model, load_model
from keras import regularizers, optimizers
from keras.callbacks import ReduceLROnPlateau


class StandardRNN(AbstractModel):
    def __init__(self, x, y):
        assert len(x.shape) == 3
        window_size = x.shape[1]
        features = x.shape[2]
        pred_length = y.shape[1]
        self.model = self.get_rnn_model(window_size, features, pred_length)

    def fit(self, train, val, batch_size, num_epochs):
        x_train, y_train = train
        x_val, y_val = val

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, verbose=True,
                                      patience=10, min_lr=0.0001)

        self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
                       shuffle=True, validation_data=(x_val, y_val), callbacks=[reduce_lr])

    def predict(self, x):
        return self.model.predict(x)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = load_model(file_path)

    @staticmethod
    def get_rnn_model(window_size, features, pred_length):
        inputs = Input(shape=(window_size, features))

        x = CuDNNGRU(512, kernel_regularizer=regularizers.l2(0.01),
                     bias_regularizer=regularizers.l2(0.01),
                     return_sequences=True,
                     input_shape=(window_size, features))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = CuDNNGRU(512, kernel_regularizer=regularizers.l2(0.01),
                     bias_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(512, activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  bias_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        preds = Dense(pred_length, activation='linear',
                      kernel_regularizer=regularizers.l2(0.01),
                      bias_regularizer=regularizers.l2(0.01))(x)
        model = Model(inputs=inputs, outputs=preds)
        optimer = optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimer, loss='mse', metrics=['mae'])

        return model
