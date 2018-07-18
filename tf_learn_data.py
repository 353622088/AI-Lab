# coding:utf-8 
'''
created on 2018/4/28

@author:Dxq
'''
import tensorflow as tf
import numpy as np

tt = {
    'img': ['1.jpg', '2.jpg'],
    'label': [1, 2]
}
dataset = tf.data.Dataset.from_tensor_slices(tt)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print('end')
