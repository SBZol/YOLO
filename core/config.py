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
from easydict import EasyDict as edict

__C = edict()

# 通过导入cfg来使用: from config import cfg
cfg = __C

# YOLO option
__C.YOLO = edict()

__C.YOLO.CLASSES = "./data/classes/coco.names"

__C.YOLO.ANCHORS = [
    12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459,
    401
]
__C.YOLO.STRIDES = [8, 16, 32]

__C.YOLO.XYSCALE = [1.2, 1.1, 1.05]

__C.YOLO.ANCHOR_PER_SCALE = 3

__C.YOLO.IOU_LOSS_THRESH = 0.5   # IOU的阈值

