#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2021/03/24 18:47:42
@Author  :   Zol
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import os
from easydict import EasyDict as edict

__C = edict()

# 通过导入cfg来使用: from config import cfg
cfg = __C

# YOLO option
__C.YOLO = edict()

__C.YOLO.CLASSES = os.path.join('F:\\', 'data', 'output', 'classes.txt')

__C.YOLO.ANCHOR_PER_SCALE = 3  # 每个scale的anchor数量

__C.YOLO.ANCHORS = [  # 每2个数一组表示 anchors 大小，有3个scale,
    12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
]
__C.YOLO.STRIDES = [8, 16, 32]  # 3个不同输出大小对应输入大小的缩小倍数

__C.YOLO.XYSCALE = [1.2, 1.1, 1.05]  # 用于计算pred_xy，暂时没搞懂干什么用

__C.YOLO.IOU_LOSS_THRESH = 0.5  # IOU的阈值

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = os.path.join('F:\\', 'data', 'output', 'train2017.txt')

__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.INPUT_SIZE = 416
__C.TRAIN.LR_INIT = 1e-3  # 初始的学习率
__C.TRAIN.LR_END = 1e-6  # 学习率的最大值
__C.TRAIN.WARMUP_EPOCHS = 1  # 热身epochs数
__C.TRAIN.FISRT_STAGE_EPOCHS = 20  # 预训练epochs数
__C.TRAIN.SECOND_STAGE_EPOCHS = 30  # 整体训练epochs数

# VAL options
__C.VAL = edict()

__C.VAL.ANNOT_PATH = os.path.join('F:\\', 'data', 'output', 'val2017.txt')

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = os.path.join('F:\\', 'data', 'output', 'test2017.txt')
__C.TEST.BATCH_SIZE = 4
__C.TEST.INPUT_SIZE = 416
# __C.TEST.DECTECTED_IMAGE_PATH = ""
# __C.TEST.SCORE_THRESHOLD      = 0.25
# __C.TEST.IOU_THRESHOLD        = 0.5
