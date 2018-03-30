import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import os

from .core import AbstractModel
from sklearn.utils import shuffle, resample
from tensorflow.contrib.layers import batch_norm
from sklearn.metrics import mean_absolute_error, mean_squared_error


from sys import platform
if not platform == 'win32':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class DilatedCNN(AbstractModel):
    def __init__(self, x, y):
        self.length = x.shape[1]
        self.features = x.shape[3]
        self.pred_length = y.shape[1]
        self.hiddens = [512, 512, 512]
        self.keep_prob = None
        self.layers = len(self.hiddens)
        self.predictions = None
        self.loss = None
        self.train_op = None
        self.mae = None
        self.sess = None
        self.data = None
        self.target = None
        self.lr = None
        self.norm = None
        self.create_placeholders()
        self.build()

    def create_placeholders(self):
        self.data = tf.placeholder(tf.float32,
                                   [None, self.length, 1, self.features])
        self.target = tf.placeholder(tf.float32, [None, self.pred_length])
        self.lr = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.norm = tf.placeholder(tf.bool)

    @staticmethod
    def get_dilated_conv(input_tensor, filter_size, strides=(1, 1), dilations=(1, 1), scope_name=None):
        with tf.name_scope(scope_name):
            conv_filter = tf.get_variable(scope_name + 'weight', filter_size,
                                          initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(scope_name + 'bias', filter_size[-1], initializer=tf.random_normal_initializer())

            logits = tf.nn.convolution(input_tensor, conv_filter, padding='SAME',
                                       strides=strides, dilation_rate=dilations)

            activations = tf.nn.selu(logits + b)
            return activations

    def add_normalizer(self, input_tensor, drop=False):
        tensor = batch_norm(input_tensor, center=True, scale=True, is_training=self.norm)
        if not drop:
            return tensor
        return tf.nn.dropout(tensor, keep_prob=self.keep_prob)

    def get_res_unit(self, input_layer, fil_size, filter_sizes=(8, 32, 32),
                     in_dilation=1, out_dilation=1, scope_name='res_unit'):
        with tf.name_scope(scope_name):
            x_in = self.get_dilated_conv(input_layer, filter_size=(fil_size, 1, filter_sizes[0], filter_sizes[1]),
                                         dilations=(in_dilation, 1), scope_name=scope_name + 'res_conv_1')
            x = tf.nn.dropout(x_in, keep_prob=self.keep_prob)
            x = self.get_dilated_conv(x, filter_size=(fil_size, 1, filter_sizes[1], filter_sizes[2]),
                                      dilations=(out_dilation, 1), scope_name=scope_name + 'res_conv_2')
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
            x_out = input_layer + x
            return x_out

    def build(self):
        filter_size = 8

        x = self.get_dilated_conv(self.data, filter_size=(filter_size, 1, 8, 32), dilations=(1, 1), scope_name='conv1')
        x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        x = self.get_dilated_conv(x, filter_size=(filter_size, 1, 32, 64), dilations=(2, 1), scope_name='conv2')
        x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        x = self.get_dilated_conv(x, filter_size=(filter_size, 1, 64, 64), dilations=(4, 1), scope_name='conv3')
        x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        x = self.get_dilated_conv(x, filter_size=(filter_size, 1, 64, 64), dilations=(8, 1), scope_name='conv4')
        x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        x = self.get_dilated_conv(x, filter_size=(filter_size, 1, 64, 64), dilations=(16, 1), scope_name='conv5')
        x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        x = self.get_dilated_conv(x, filter_size=(4, 1, 64, 64), dilations=(32, 1), scope_name='conv6')

        # x = self.get_dilated_conv(self.data, filter_size=(filter_size, 1, 8, 32),
        # dilations=(1, 1), scope_name='conv0')
        # x = self.get_res_unit(x, fil_size=filter_size, filter_sizes=(32, 32, 32),
        #                       in_dilation=1, out_dilation=1, scope_name='conv1')
        # x = self.get_res_unit(x, fil_size=filter_size, filter_sizes=(32, 32, 32),
        #                       in_dilation=2, out_dilation=2, scope_name='conv2')
        # x = self.get_res_unit(x, fil_size=filter_size, filter_sizes=(32, 32, 32),
        #                       in_dilation=4, out_dilation=4, scope_name='conv3')
        # x = self.get_res_unit(x, fil_size=filter_size, filter_sizes=(32, 32, 32),
        #                       in_dilation=8, out_dilation=8, scope_name='conv4')
        # x = self.get_res_unit(x, fil_size=filter_size, filter_sizes=(32, 32, 32),
        #                       in_dilation=16, out_dilation=16, scope_name='conv5')
        # x = self.get_res_unit(x, fil_size=filter_size, filter_sizes=(32, 32, 32),
        #                       in_dilation=32, out_dilation=32, scope_name='conv6')

        x = tf.slice(x, [0, 0, 0, 0], [tf.shape(x)[0], 6, 1, 64])
        x = tf.reshape(x, [-1, 6 * 64])
        x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        x = slim.fully_connected(x, 512, activation_fn=tf.nn.selu, scope='fc1')
        x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        self.predictions = slim.fully_connected(x, self.pred_length,
                                                activation_fn=None, scope='final')

        self.loss = tf.losses.mean_squared_error(self.target, self.predictions)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        diff = tf.abs(tf.subtract(self.target, self.predictions))
        self.mae = tf.reduce_mean(diff)

    def fit(self, train, val, test, batch_size, num_epochs):
        x_train, y_train = train
        x_val, y_val = val
        x_test, y_test = test

        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        def get_lr(epoch_index):
            if epoch_index < 64:
                return .001
            if epoch_index < 96:
                return .0007
            return .0005

        for epoch in range(num_epochs):
            x_, y_ = shuffle(x_train, y_train)

            start_time = time.time()
            loss_sum = 0
            loss_qty = 0
            for i in range(0, x_.shape[0], batch_size):
                fd = {self.data: x_[i: i + batch_size, :, :, :],
                      self.target: y_[i: i + batch_size, :],
                      self.lr: get_lr(epoch), self.keep_prob: 0.5,
                      self.norm: True}
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=fd)
                loss_sum += loss
                loss_qty += 1

            train_mse = loss_sum / loss_qty

            fd_val = {self.data: x_val,
                      self.target: y_val,
                      self.keep_prob: 1.0,
                      self.norm: False}

            val_mse, val_mae, val_preds = self.sess.run([self.loss, self.mae, self.predictions],
                                                        feed_dict=fd_val)

            mse = mean_squared_error(y_val, val_preds)
            mae = mean_absolute_error(y_val, val_preds)

            print('epoch {:4d}: train mse: {:.3f}, '
                  'val mse: {:.3f}, val mae: {:.3f}, _mse: {:.3f}, _mae: {:.3f}, '
                  'Elapsed time {:.1f} s'.format(epoch,
                                                 train_mse,
                                                 val_mse, val_mae,
                                                 mse, mae,
                                                 time.time() - start_time))

        fd_test = {self.data: x_test,
                   self.keep_prob: 1.0,
                   self.norm: False}

        y_test_pred = self.sess.run(self.predictions, feed_dict=fd_test)

        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)

        print('inside fit, test mse: {mse}, test mae: {mae}'.format(mse=mse, mae=mae))
        
    def predict(self, x):
        return self.sess.run(self.predictions, {self.data: x, self.keep_prob: 1.0, self.norm: False})

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass
