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
    zol：目前没搞懂继承这个BN类重写它的目的
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convolutional(input_layer,
                  filters_shape,
                  downsample=False,
                  activate=True,
                  bn=True,
                  activate_type='leaky'):
    """[封装的卷积函数]

    Args:
        input_layer (tensor): 输入层]
        filters_shape (tuple/list): filter 的维度, (f, f, c)
        downsample (bool, optional): 是否为下采样的卷积. Defaults to False.
        activate (bool, optional): 是否定义激活函数. Defaults to True.
        bn (bool, optional): 是否采用BatchNormalization. Defaults to True.
        activate_type (str, optional): 激活函数的类型，'leaky' or 'mish'. Defaults to 'leaky'.

    Returns:
        Conv2D
    """

    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(
            ((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        stride = 2
    else:
        stride = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(
        filters=filters_shape[-1],
        kernel_size=filters_shape[0],
        strides=stride,
        padding=padding,
        use_bias=not bn,
        kernel_regularizer=tf.keras.regularizers.l2(l2=0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn:
        conv = BatchNormalization()(conv)

    if activate == True:
        if activate == 'leaky':
            conv = tf.nn.leaky_relu(conv, alpha=0.1)

        elif activate == 'mish':
            conv = mish(conv)

    return conv


def residual_block(input_layer,
                   input_channel,
                   filter_nums,
                   activate_type='leaky'):
    """残差模块

    Args:
        input_layer (tensor): 输入特征
        input_channel (uint): 输入特征的通道数
        filter_nums (tuple): 第一和第二层卷积的filter数
        activate_type (str, optional): 激活函数类型. Defaults to 'leaky'.

    Returns:
        tensor: 残差模块的输出
    """

    short_cut = input_layer

    filter_num1 = filter_nums[0]
    filter_num2 = filter_nums[1]

    conv = convolutional(input_layer,
                         filters_shape=(1, 1, input_channel, filter_num1),
                         activate_type=activate_type)
    conv = convolutional(conv,
                         filters_shape=(3, 3, filter_num1, filter_num2),
                         activate_type=activate_type)

    res_output = short_cut + conv  # 残差连接

    return res_output


def upsample(input_layer):
    """resize图像变成原本的两倍

    Args:
        input_layer (tensor): 输入图像

    Returns:
        tensor: resize后的图像
    """
    upsample_layer = tf.image.resize(
        input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2),
        method='bilinear')
    return upsample_layer


def route_group(input_layer, groups, group_id):
    """拆分张量，并返回需要的子张量列表

    Args:
        input_layer (tensor): 输入的tensor
        groups ([type]): 分组数 
        group_id ([type]): 返回的子张量坐标

    Returns:
        tensor: 拆分后指定的子张量
    """
    conv = tf.split(input_layer, groups, -1)

    sub_tensor = conv[group_id]

    return sub_tensor
