#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   activations.py
@Time    :   2021/03/23 16:12:06
@Author  :   Zol 
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import tensorflow as tf


def mish(x):
    """Mish激活函数，a smooth，self regularized， non-monotonic activa function

    Args:
        x (tensor): 输入的特征

    Returns:
        tensor: 激活值
    """

    x = x * tf.math.tanh(tf.math.softplus(x))

    return x
