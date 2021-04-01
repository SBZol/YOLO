#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/03/25 18:24:40
@Author  :   Zol 
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
from core.config import cfg

import cv2
import threading
import numpy as np
import tensorflow as tf


class myThread(threading.Thread):
    def __init__(self, threadID, name, process_func, **kwargs):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.process_func = process_func
        self.kwargs = kwargs

    def run(self):
        print("开启线程：" + self.name)
        self.process_func(**self.kwargs)
        print("退出线程：" + self.name)


def iou(bboxes1, bboxes2):
    """计算bboxes的IoU

    Args:
        bboxes1 : bounding box 1
        bboxes2 : bounding box 2

    Returns:
        [tf]: IoU
    """

    inter_area, union_area, _ = process_bboxes(bboxes1,
                                               bboxes2,
                                               need_enclose=False)
    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou


def giou(bboxes1, bboxes2):
    """计算bboxes的gIoU

    Args:
        bboxes1 : bounding box 1
        bboxes2 : bounding box 2

    Returns:
        [tf]: gIoU
    """

    inter_area, union_area, enclose_section = process_bboxes(bboxes1, bboxes2)
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]  # 计算外接矩形

    iou = tf.math.divide_no_nan(inter_area, union_area)
    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def ciou(bboxes1, bboxes2):
    """计算bboxes的cIoU

    Args:
        bboxes1 : bounding box 1
        bboxes2 : bounding box 2

    Returns:
        [tf]: cIoU
    """

    inter_area, union_area, enclose_section = process_bboxes(bboxes1, bboxes2)
    iou = tf.math.divide_no_nan(inter_area, union_area)

    # 最小外接矩形对角线长度的平方
    c_2 = enclose_section[..., 0]**2 + enclose_section[..., 1]**2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    # bboxes中心点的欧氏距离
    rho_2 = center_diagonal[..., 0]**2 + center_diagonal[..., 1]**2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v_1 = tf.math.atan(tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3]))
    v_2 = tf.math.atan(tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3]))
    v = (4 / np.pi**2) * (v_1 - v_2)**2
    # v = ((v_1 - v_2) * 2 / np.pi)**2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)
    ciou = diou - alpha * v

    return ciou


def process_bboxes(bboxes1, bboxes2, need_enclose=True):
    """根据bbboxes计算出交集、并集和外接矩形的信息

    Args:
        bboxes1 : bounding box 1
        bboxes2 : bounding box 2
        need_enclose : 是否需要计算外接矩形的区域
    Returns:
        inter_area(交集), union_area(并集), enclose_section(外接矩形，若need_enclose=False，值为None)
    """

    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        -1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        -1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    enclose_section = None
    if need_enclose:
        enclose_left_up = tf.minimum(bboxes1_coor[..., :2],
                                     bboxes2_coor[..., :2])
        enclose_right_down = tf.maximum(bboxes1_coor[..., 2:],
                                        bboxes2_coor[..., 2:])

        enclose_section = enclose_right_down - enclose_left_up

    return inter_area, union_area, enclose_section


def load_config(FLAGS):
    """读取config数据

    Args:
        FLAGS : 输入的flags数据

    Returns:
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE
    """
    STRIDES = np.array(cfg.YOLO.STRIDES)

    anchers = np.array(cfg.YOLO.ANCHORS)

    ANCHORS = np.reshape(anchers, (3, 3, 2))

    XYSCALE = cfg.YOLO.XYSCALE

    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE


def read_class_names(class_file_paht):
    """获取txt文件中的所有分类名

    Args:
        class_file_name (str): 存储类别名的txt文件  

    Returns:
        [dict]: 存储类别名的字典
    """
    names = {}
    with open(class_file_paht, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
    
    
def load_freeze_layer(model='yolov4'):
    if model == 'yolov4':
        freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']
    
    return freeze_layouts


def load_weights(model, weights_file, model_name='yolov4'):
   
    if model_name == 'yolov3':
        layer_size = 75
        output_pos = [58, 66, 74]
    else:
        layer_size = 110
        output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()
    
    
def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
            
def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)