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
import utils
from backbone import cspdarknet53
from common import convolutional, upsample

import numpy as np
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
    conv = convolutional(input_layer=route_1,
                         filters_shape=(3, 3, 128, 256),
                         downsample=True)
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

    conv = convolutional(input_layer=route_2,
                         filters_shape=(3, 3, 256, 512),
                         downsample=True)
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


def decode_train(conv_output,
                 output_size,
                 NUM_CLASS,
                 STRIDES,
                 ANCHORS,
                 i=0,
                 XYSCALE=[1, 1, 1]):
    conv_output = tf.reshape(
        conv_output,
        (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (2, 2, 1, NUM_CLASS), -1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size)) # 从两数组中产生网格
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1),
                             axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0),
                      [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32) # 转换数据类型

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], -1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    
    return tf.concat([pred_xywh, pred_conf, pred_prob], -1)



def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss