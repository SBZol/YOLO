#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   common.py
@Time    :   2021/03/23 14:33:06
@Author  :   Zol 
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
from activations import mish

import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    zol：目前没搞懂继承这个类重写它的目的
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)
    
    
def convolutional(input_later, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    """[封装的卷积函数]

    Args:
        input_later (tensor): 输入层]
        filters_shape (tuple/list): filter 的维度, (f, f, c)
        downsample (bool, optional): 是否为下采样的卷积. Defaults to False.
        activate (bool, optional): 是否定义激活函数. Defaults to True.
        bn (bool, optional): 是否采用BatchNormalization. Defaults to True.
        activate_type (str, optional): 激活函数的类型，'leaky' or 'mish'. Defaults to 'leaky'.

    Returns:
        Conv2D
    """
    
    if downsample:
        input_later = tf.keras.layers.ZeroPadding2D((1,0), (1,0))(input_later)
        padding = 'valid'
        stride = 2
    else:
        stride = 1
        padding = 'same'
    
    conv = tf.keras.layers.Conv2D(filters = filters_shape[-1], 
                                  kernel_size = filters_shape[0],
                                  strides = strides,
                                  padding = padding,
                                  use_bias = not bn,
                                  kernel_regularizer = tf.keras.regularizers.l2(l2=0.0005),
                                  kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer = tf.constant_initializer(0.)
                                  )(input_later)
    
    if bn:
        conv = BatchNormalization()(conv)
        
    if activate == True:
        if activate == 'leaky':
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
            
        elif activate == 'mish':
            conv = mish(conv)
            
    return conv


