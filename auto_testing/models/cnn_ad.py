import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import os
import math

from .core import AbstractModel
from sklearn.utils import shuffle, resample
from tensorflow.contrib.layers import batch_norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from sys import platform
if not platform == 'win32':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class DilatedCNNAD(AbstractModel):
    def __init__(self, x, y, adaptation):
        y_class = y[:, -1]
        self.adaptation = adaptation
        self.length = x.shape[1]
        self.features = x.shape[3]
        self.pred_length = y.shape[1] - 1
        self.classes_num = int(np.max(y_class) - np.min(y_class)) + 1
        print('preds length:', self.pred_length)
        print('classes_num:', self.classes_num)
        self.hiddens = [512, 512, 512]
        self.keep_prob = None
        self.layers = len(self.hiddens)
        self.predictions = None
        self.reg_loss = None
        self.class_loss = None
        self.class_preds = None
        self.predictor_loss = None
        self.classifier_loss = None
        self.sum_loss = None
        self.train_regressor_op = None
        self.train_predictor_op = None
        self.train_classifier_op = None
        self.train_classifier_op_all = None
        self.train_sum = None
        self.acc = None
        self.acc_op = None
        self.mae = None
        self.sess = None
        self.data = None
        self.target = None
        self.class_target = None
        self.lr = None
        self.norm = None
        self._lambda = None
        self.ranges = None
        self.create_placeholders()
        self.build()

    def close(self):
        tf.reset_default_graph()
        self.sess.close()

    def create_placeholders(self):
        self.data = tf.placeholder(tf.float32,
                                   [None, self.length, 1, self.features])
        self.target = tf.placeholder(tf.float32, [None, self.pred_length])
        self.class_target = tf.placeholder(tf.int32, [None, self.classes_num])
        self.ranges = tf.placeholder(tf.float32, [self.pred_length])
        self.lr = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.norm = tf.placeholder(tf.bool)
        self._lambda = tf.placeholder(tf.float32, shape=())

    def get_dilated_conv(self, input_tensor, filter_size, strides=(1, 1), dilations=(1, 1), scope_name=None):
        with tf.name_scope(scope_name):
            conv_filter = tf.get_variable(scope_name + 'weight', filter_size,
                                          initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(scope_name + 'bias', filter_size[-1], initializer=tf.random_normal_initializer())

            logits = tf.nn.convolution(input_tensor, conv_filter, padding='VALID',
                                       strides=strides, dilation_rate=dilations)

            activations = tf.nn.relu(logits + b)
            return activations

    def add_normalizer(self, input_tensor, drop=False):
        tensor = batch_norm(input_tensor, center=True, scale=True, is_training=self.norm)
        if not drop:
            return tensor
        return tf.nn.dropout(tensor, keep_prob=self.keep_prob)

    def get_res_unit(self, input_layer, fil_size, filter_sizes=(8, 32, 32),
                     in_dilation=1, out_dilation=1, scope_name='res_unit'):
        with tf.name_scope(scope_name):
            x = self.get_dilated_conv(input_layer, filter_size=(fil_size, 1, filter_sizes[0], filter_sizes[1]),
                                      dilations=(in_dilation, 1), scope_name=scope_name + 'res_conv_1')
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.norm)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
            x = self.get_dilated_conv(x, filter_size=(fil_size, 1, filter_sizes[1], filter_sizes[2]),
                                      dilations=(out_dilation, 1), scope_name=scope_name + 'res_conv_2')
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.norm)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
            x_out = input_layer + x
            return x_out

    def build(self):
        filter_size = 8
        filters = 48

        feature_tensors = []

        with tf.variable_scope('predictor'):
            x = self.get_dilated_conv(self.data, filter_size=(filter_size, 1, 8, filters),
                                      dilations=(1, 1), scope_name='conv1')
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.norm)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)

            x = self.get_dilated_conv(x, filter_size=(filter_size, 1, filters, filters),
                                      dilations=(2, 1), scope_name='conv2')
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.norm)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)

            x = self.get_dilated_conv(x, filter_size=(filter_size, 1, filters, filters),
                                      dilations=(4, 1), scope_name='conv3')
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.norm)
            # feature_tensors.append(x)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)

            x = self.get_dilated_conv(x, filter_size=(filter_size, 1, filters, filters),
                                      dilations=(8, 1), scope_name='conv4')
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.norm)
            feature_tensors.append(x)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)

            x = tf.slice(x, [0, 0, 0, 0], [tf.shape(x)[0], 12, 1, filters])
            features = tf.reshape(x, [-1, 12 * filters])

            x = slim.fully_connected(features, 256, activation_fn=tf.nn.relu, scope='fc1')
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.norm)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)

            x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc2')
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.norm)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)

            self.predictions = slim.fully_connected(x, self.pred_length,
                                                    activation_fn=None, scope='final')

        self.reg_loss = tf.losses.mean_squared_error(self.target, self.predictions)

        with tf.variable_scope('classifier'):
            feature_flattens = [tf.layers.flatten(feature_tensor) for feature_tensor in feature_tensors]
            concatenated = tf.concat(feature_flattens, -1)

            inv_concat = tf.scalar_mul(-self._lambda, concatenated)
            inverse_gradient_layer = inv_concat + tf.stop_gradient(concatenated - inv_concat)

            # inv_concat = tf.scalar_mul(-self._lambda, features)
            # inverse_gradient_layer = inv_concat + tf.stop_gradient(features - inv_concat)

            x_ad = slim.fully_connected(inverse_gradient_layer, 512, activation_fn=tf.nn.relu, scope='fc1_ad')

            # x_ad = slim.fully_connected(concatenated, 512, activation_fn=tf.nn.relu, scope='fc1_ad')
            x_ad = tf.contrib.layers.batch_norm(x_ad, center=True, scale=True, is_training=self.norm)

            self.class_preds = slim.fully_connected(x_ad, self.classes_num,
                                                    activation_fn=None)

        self.acc, self.acc_op = tf.metrics.accuracy(labels=tf.argmax(self.class_target, 1),
                                                    predictions=tf.argmax(self.class_preds, 1))

        self.class_loss = tf.losses.softmax_cross_entropy(self.class_target, self.class_preds)

        self.predictor_loss = self.reg_loss + self.class_loss
        self.classifier_loss = self.class_loss

        self.sum_loss = self.reg_loss + self.class_loss

        predictor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'predictor')
        classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'classifier')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_regressor_op = tf.train.AdamOptimizer(learning_rate=self.lr).\
                minimize(self.reg_loss, var_list=predictor_vars)
            self.train_predictor_op = tf.train.AdamOptimizer(learning_rate=self.lr).\
                minimize(self.predictor_loss)
            self.train_classifier_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).\
                minimize(self.classifier_loss, var_list=classifier_vars)
            self.train_classifier_op_all = tf.train.AdamOptimizer(learning_rate=self.lr).\
                minimize(self.classifier_loss)

        diff = tf.abs(tf.subtract(self.target, self.predictions))
        self.mae = tf.reduce_mean(diff)

    def fit(self, train, val, test, batch_size, num_epochs):
        x_train, y_train = train
        x_val, y_val = val
        x_test, y_test = test

        print('class_num: ', self.classes_num)
        print(y_train[:, -1].max(), y_val[:, -1].max(), y_test[:, -1].max())
        one_hot = OneHotEncoder(self.classes_num)

        def split_y(y):
            y_class = np.array(y[:, -1], dtype=np.int32)
            y = y[:, :-1]
            y_class_one_hot = one_hot.fit_transform(y_class.reshape((-1, 1))).todense()
            return y, y_class_one_hot

        y_train, y_train_class = split_y(y_train)
        y_val, y_val_class = split_y(y_val)
        y_test, y_test_class = split_y(y_test)

        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        def get_lr(epoch_index):
            if epoch_index < 64:
                return .001
            if epoch_index < 96:
                return .0007
            return .0005

        adversarial = self.adaptation
        print('start learning, with ad: ', adversarial)
        lin_limit = 0.5
        # lambdas = np.hstack((np.linspace(0.1, lin_limit, 32), np.full(16, lin_limit)))
        # print('lambdas shape: ', lambdas.shape)
        lambdas = np.linspace(0.1, 0.5, num_epochs)
        # lambdas = np.linspace(0.1, 1.0, num_epochs)

        for epoch in range(num_epochs):
            print('lambda: {:.4f}'.format(lambdas[epoch]))
            # if self.adaptation:
            #     adversarial = False if epoch % 2 == 0 else True
            x_, y_, y_train_class_ = shuffle(x_train, y_train, y_train_class)

            start_time = time.time()
            loss_sum = 0
            acc_sum = 0
            loss_qty = 0
            acc_qty = 0

            for i in range(0, x_.shape[0], batch_size):

                x_batch = x_[i: i + batch_size, :, :, :]
                y_batch = y_[i: i + batch_size, :]
                y_batch_class = y_train_class_[i: i + batch_size, ]

                y_maxes = np.max(y_batch, axis=0)
                y_mins = np.min(y_batch, axis=0)
                y_ranges = y_maxes - y_mins

                fd = {self.data: x_batch,
                      self.target: y_batch,
                      self.class_target: y_batch_class,
                      self.ranges: y_ranges,
                      self.lr: get_lr(epoch),
                      self.keep_prob: 0.4,
                      # self.keep_prob: 0.55,
                      self.norm: 1,
                      self._lambda: lambdas[epoch]}

                if not adversarial:
                    loss, _ = self.sess.run([self.reg_loss,
                                             self.train_regressor_op], feed_dict=fd)
                    loss_sum += loss
                    loss_qty += 1
                else:
                    acc, _, loss, _, _ = self.sess.run([self.acc,
                                                        self.acc_op,
                                                        self.reg_loss,
                                                        self.train_regressor_op,
                                                        self.train_classifier_op_all], feed_dict=fd)

                    # acc, _, loss, _ = self.sess.run([self.acc,
                    #                                  self.acc_op,
                    #                                  self.reg_loss,
                    #                                  self.train_predictor_op], feed_dict=fd)

                    loss_sum += loss
                    loss_qty += 1

                    # acc, _, _ = self.sess.run([self.acc, self.acc_op,
                    #                            self.train_classifier_op], feed_dict=fd)

                    acc_sum += acc
                    acc_qty += 1

            if not adversarial:
                train_mse = loss_sum / loss_qty
                train_acc = 0
            else:
                train_mse = loss_sum / loss_qty
                train_acc = acc_sum / acc_qty

                # train_mse = 0
                # train_acc = acc_sum / acc_qty

            val_maxes = np.max(y_val, axis=0)
            val_mins = np.min(y_val, axis=0)
            val_ranges = val_maxes - val_mins

            fd_val = {self.data: x_val,
                      self.target: y_val,
                      self.class_target: y_val_class,
                      self.ranges: val_ranges,
                      self.keep_prob: 1.0,
                      self.norm: 0}

            val_mse, val_mae, val_preds = self.sess.run([self.reg_loss, self.mae, self.predictions],
                                                        feed_dict=fd_val)

            y_val_norm = y_val / val_ranges
            val_preds_norm = val_preds / val_ranges

            val_nrmse = math.sqrt(mean_squared_error(y_val_norm, val_preds_norm))

            print('epoch {:3d}: tr mse: {:.1f}, tr acc: {:.3f}, '
                  'v mse: {:.1f}, v mae: {:.1f}, v nrmse: {:.3f}, '
                  'Elapsed time {:.1f} s'.format(epoch,
                                                 train_mse,
                                                 train_acc,
                                                 val_mse, val_mae, val_nrmse,
                                                 time.time() - start_time))

    def predict(self, x):
        return self.sess.run(self.predictions, {self.data: x, self.keep_prob: 1.0, self.norm: 0})

    def predict_on_batch(self, x, batch_size=512):
        preds = []

        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i: i + batch_size, :, :, :]
            y_batch = self.sess.run(self.predictions, {self.data: x_batch,
                                                       self.keep_prob: 1.0,
                                                       self.norm: 0})
            preds.append(y_batch)

        return np.vstack(preds)

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass
