#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   yolov4.py
@Time    :   2021/03/24 15:58:21
@Author  :   Zol 
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
from backbone import cspdarknet53
from common import convolutional, upsample

import tensorflow as tf


def yolov4(input_layer, NUM_CLASS):

    ## PANet - Path Agrregate Network
    route_1, route_2, route_3 = cspdarknet53(input_layer)

    # up
    conv = convolutional(input_layer=route_3, filters_shape=(1, 1, 512, 256))
    conv = upsample(conv)

    route_2 = convolutional(input_layer=route_2,
                            filters_shape=(1, 1, 512, 256))
    conv = tf.concat([conv, route_2], -1)

    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 512, 256))
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 256, 512))
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 512, 256))
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 256, 512))
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 256, 128))
    conv = upsample(conv)

    route_1 = convolutional(input_layer=conv, filters_shape=(1, 1, 256, 128))
    conv = tf.concat([route_1, conv], -1)

    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 256, 128))
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 128, 256))
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 256, 128))
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 128, 256))
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 256, 128))

    route_1 = conv
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 128, 256))
    
    conv_sbbox = convolutional(input_layer=conv,
                               filters_shape=(1, 1, 256, 3 * (NUM_CLASS + 5)),
                               activate=False,
                               bn=False)
    
    # down
    conv = convolutional(input_layer=route_1, filters_shape=(3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], -1)
    
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 512, 256))
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 256, 512))
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 512, 256))
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 256, 512))
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 256, 512))
    
    conv_mbbox = convolutional(input_layer=conv,
                               filters_shape=(1, 1, 512, 3 * (NUM_CLASS + 5)),
                               activate=False,
                               bn=False) 
    
    conv = convolutional(input_layer=route_2, filters_shape=(3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route_3], -1)
    
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 1024, 512))
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 512, 1024))
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 1024, 512))
    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 512, 1024))
    conv = convolutional(input_layer=conv, filters_shape=(1, 1, 1024, 512))

    conv = convolutional(input_layer=conv, filters_shape=(3, 3, 512, 1024))
    
    conv_lbbox = convolutional(input_layer=conv,
                               filters_shape=(1, 1, 1024, 3 * (NUM_CLASS + 5)),
                               activate=False,
                               bn=False)        
    
    return [conv_sbbox, conv_mbbox, conv_lbbox]