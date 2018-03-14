import numpy as np
import tensorflow as tf


with tf.Session() as sess:

    input_tensor = tf.placeholder(tf.float32, [None, 6, 1, 1])

    filter1 = np.array([1, 1, 1], dtype=np.float32).reshape((3, 1, 1, 1))

    logits = tf.nn.convolution(input_tensor, filter1, padding='SAME',
                               strides=(1, 1), dilation_rate=(2, 1))

    fd = {input_tensor: np.array([2, 4, 6, 8, 10, 12]).reshape((1, 6, 1, 1))}

    result = sess.run([logits], feed_dict=fd)

    print(result[0].reshape((-1)))
