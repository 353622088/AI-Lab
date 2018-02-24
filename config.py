# coding:utf-8 
'''
created on 2018/2/24

@author:Dxq
'''
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string(flag_name='xx', default_value='value', docstring='help')

cfg = flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
