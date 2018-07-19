# coding:utf-8
'''
created on 2018/7/19

@author:sw-git01
'''
import tensorflow as tf
import tensorflow.contrib.session_bundle.exporter as exporter
import numpy as np

x_data = np.arange(100, step=.1)
# print(x_data)
# print(x_data.shape)
# b=np.reshape(x_data,[1000,1])
# print(b.shape)
# assert 1==0
n_samples = 1000
y_data = x_data + 20 * np.sin(x_data / 10)
x_data = np.reshape(x_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

sample = 1000
learning_rate = 0.01
batch_size = 100
n_steps = 500

x = tf.placeholder(tf.float32, shape=(batch_size, 1))

y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope('test'):
    w = tf.get_variable('weights', (1, 1), initializer=tf.random_normal_initializer())
    b = tf.get_variable('bias', (1,), initializer=tf.constant_initializer(0))

    y_pred = tf.matmul(x, w) + b
    loss = tf.reduce_sum((y - y_pred) ** 2 / n_samples)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(n_steps):
        print(_)
        indices = np.random.choice(n_samples, batch_size)
        x_batch = x_data[indices]
        y_batch = y_data[indices]
        _, loss_val = sess.run([opt, loss], feed_dict={x: x_batch, y: y_batch})

    saver = tf.train.Saver()

    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'x': x}),
            'outputs': exporter.generic_signature({'y': y_pred})})
    model_exporter.export('', tf.constant(1), sess)
