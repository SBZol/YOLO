#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   backbone.py
@Time    :   2021/03/23 18:14:42
@Author  :   Zol 
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
from common import convolutional, residual_block

import tensorflow as tf


def cspdarknet53(input_data):
    """CSPDarknet53, 包含SPP模块在内

    Args:
        input_data : 网络输入
        
    Returns:
        (tensor): 返回3条支路的输出
    """

    ## head
    output = convolutional(input_layer=input_data,
                           filters_shape=(3, 3, 3, 32),
                           activate_type='mish')
    output = convolutional(output, (3, 3, 32, 64),
                           downsample=True,
                           activate_type='mish')  # downsample_0

    ## cross stage partial 1
    route = output
    route = convolutional(output,
                          filters_shape=(1, 1, 64, 64),
                          activate_type='mish')  # part 1_1

    output = convolutional(output,
                           filters_shape=(1, 1, 64, 64),
                           activate_type='mish')  # part 1_2
    output = residual_block(output,
                            input_channel=64,
                            filter_nums=(32, 64),
                            activate_type='mish')  # res_block_1
    output = convolutional(output,
                           filters_shape=(1, 1, 64, 64),
                           activate_type='mish')  # transition_1_1

    output = tf.concat([output, route], -1)  # concat_1
    output = convolutional(output,
                           filters_shape=(1, 1, 128, 64),
                           activate_type='mish')  # transition_1_2
    output = convolutional(output,
                           filters_shape=(3, 3, 64, 128),
                           downsample=True,
                           activate_type='mish')  # downsample_1

    ## cross stage partial 2
    route = output
    route = convolutional(output,
                          filters_shape=(1, 1, 128, 64),
                          activate_type='mish')  # part 2_1

    output = convolutional(output,
                           filters_shape=(1, 1, 128, 64),
                           activate_type='mish')  # part 2_2

    for _ in range(2):
        output = residual_block(input_layer=output,
                                input_channel=64,
                                filter_nums=(64, 64),
                                activate_type='mish')  # res_block_2
    output = convolutional(output,
                           filters_shape=(1, 1, 64, 64),
                           activate_type='mish')  # transition_2_1

    output = tf.concat([output, route], -1)  # concat_2
    output = convolutional(output,
                           filters_shape=(1, 1, 128, 128),
                           activate_type='mish')  # transition_2_2
    output = convolutional(output,
                           filters_shape=(3, 3, 128, 256),
                           downsample=True,
                           activate_type='mish')  # downsample_2

    ## cross stage partial 3
    route = output
    route = convolutional(output,
                          filters_shape=(1, 1, 256, 128),
                          activate_type='mish')  # part 3_1

    output = convolutional(output,
                           filters_shape=(1, 1, 256, 128),
                           activate_type='mish')  # part 3_2
    for _ in range(8):
        output = residual_block(input_layer=output,
                                input_channel=128,
                                filter_nums=(128, 128),
                                activate_type='mish')  # res_block_3
    output = convolutional(output,
                           filters_shape=(1, 1, 128, 128),
                           activate_type='mish')  # transition_3_1

    output = tf.concat([output, route], -1)  # concat_3
    output = convolutional(output,
                           filters_shape=(1, 1, 256, 256),
                           activate_type='mish')  # transition_3_2

    route_output_1 = output  # 支线输出1

    output = convolutional(output,
                           filters_shape=(3, 3, 256, 512),
                           downsample=True,
                           activate_type='mish')  # downsample_3

    ## cross stage partial 4
    route = output
    route = convolutional(output,
                          filters_shape=(1, 1, 512, 256),
                          activate_type='mish')  # part 4_1

    output = convolutional(output,
                           filters_shape=(1, 1, 512, 256),
                           activate_type='mish')  # part 4_2
    for _ in range(8):
        output = residual_block(input_layer=output,
                                input_channel=256,
                                filter_nums=(256, 256),
                                activate_type='mish')  # res_block_4
    output = convolutional(output,
                           filters_shape=(1, 1, 256, 256),
                           activate_type='mish')  # transition_4_1

    output = tf.concat([output, route], -1)  # concat_4
    output = convolutional(output,
                           filters_shape=(1, 1, 512, 512),
                           activate_type='mish')  # transition_4_2

    route_output_2 = output  # 支线输出2

    output = convolutional(output,
                           filters_shape=(3, 3, 512, 1024),
                           downsample=True,
                           activate_type='mish')  # downsample_4

    ## cross stage partial 5
    route = output
    route = convolutional(output,
                          filters_shape=(1, 1, 1024, 1024),
                          activate_type='mish')  # part 5_1

    output = convolutional(output,
                           filters_shape=(1, 1, 1024, 512),
                           activate_type='mish')  # part 5_2
    for _ in range(4):
        output = residual_block(input_layer=output,
                                input_channel=512,
                                filter_nums=(512, 512),
                                activate_type='mish')  # res_block_5
    output = convolutional(output,
                           filters_shape=(1, 1, 512, 512),
                           activate_type='mish')  # transition_5_1

    output = tf.concat([output, route], -1)  # concat_5
    output = convolutional(output,
                           filters_shape=(1, 1, 1024, 1024),
                           activate_type='mish')  # transition_5_2

    ## tail - Conv x3
    output = convolutional(output, filters_shape=(1, 1, 1024, 512))
    output = convolutional(output, filters_shape=(3, 3, 512, 1024))
    output = convolutional(output, filters_shape=(1, 1, 1024, 512))

    ## SPP - Spatial Pyramid Pooling
    max_pooling1 = tf.nn.max_pool(output, ksize=13, padding='SAME', strides=1)
    max_pooling2 = tf.nn.max_pool(output, ksize=9, padding='SAME', strides=1)
    max_pooling3 = tf.nn.max_pool(output, ksize=5, padding='SAME', strides=1)
    output = tf.concat([max_pooling1, max_pooling2, max_pooling3, output], -1)

    output = convolutional(output, filters_shape=(1, 1, 2048, 512))
    output = convolutional(output, filters_shape=(3, 3, 512, 1024))
    output = convolutional(output, filters_shape=(1, 1, 1024, 512))

    route_output_3 = output  # 支线输出2

    return route_output_1, route_output_2, route_output_3


def darknet53(input_data):
    """darknet 53

    Args:
        input_data (tendor): 

    Returns:
        (tensor): 返回3条支路的输出
    """

    output = convolutional(input_layer=input_data, filters_shape=(3, 3, 3, 32))
    output = convolutional(input_layer=output,
                           filters_shape=(3, 3, 32, 64),
                           downsample=True)  # 下采样

    output = residual_block(input_layer=output,
                            input_channel=64,
                            filter_nums=(32, 64))  # 残差模块

    output = convolutional(input_layer=output,
                           filters_shape=(3, 3, 64, 128),
                           downsample=True)

    for _ in range(2):
        output = residual_block(input_layer=output,
                                input_channel=128,
                                filter_nums=(64, 128))

    output = convolutional(input_layer=output,
                           filters_shape=(3, 3, 128, 256),
                           downsample=True)  # 下采样

    for _ in range(8):
        output = residual_block(input_layer=output,
                                input_channel=256,
                                filter_nums=(128, 256))

    route_1 = output  # 支线1
    output = convolutional(input_layer=output,
                           filters_shape=(3, 3, 256, 512),
                           downsample=True)  # 下采样

    for _ in range(8):
        output = residual_block(input_layer=output,
                                input_channel=512,
                                filter_nums=(256, 512))

    route_2 = output  # 支线2
    output = convolutional(input_layer=output,
                           filters_shape=(3, 3, 512, 1024),
                           downsample=True)  # 下采样

    for _ in range(4):
        output = residual_block(input_layer=output,
                                input_channel=1024,
                                filter_nums=(512, 1024))

    route_3 = output  # 支线3

    return route_1, route_2, route_3


if __name__ == '__main__':
    import numpy as np

    ##绘制网络图

    intput_data = tf.keras.Input(shape=(512, 512, 3))

    route_output_1, route_output_2, route_output_3 = cspdarknet53(intput_data)

    model_1 = tf.keras.Model(intput_data,
                             [route_output_1, route_output_2, route_output_3])

    tf.keras.utils.plot_model(model_1,
                              to_file='model3.png',
                              show_shapes=True,
                              show_layer_names=True,
                              rankdir='TB',
                              dpi=900,
                              expand_nested=True)
