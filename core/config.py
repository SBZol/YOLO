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

# __C.YOLO.CLASSES = ""

# __C.YOLO.ANCHORS = [
#     12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459,
#     401
# ]
# __C.YOLO.STRIDES = [8, 16, 32]

# __C.YOLO.XYSCALE = [1.2, 1.1, 1.05]

# __C.YOLO.ANCHOR_PER_SCALE = 3

# __C.YOLO.IOU_LOSS_THRESH = 0.5   # IOU的阈值

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = os.path.join('F:\\', 'data', 'annotations',
                                    'instances_train2017.json')
__C.TRAIN.DATA_PATH = os.path.join('F:\\', 'data', 'coco2017', 'train2017')
# __C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE          = 416
# __C.TRAIN.DATA_AUG            = True
# __C.TRAIN.LR_INIT             = 1e-3
# __C.TRAIN.LR_END              = 1e-6
# __C.TRAIN.WARMUP_EPOCHS       = 2
# __C.TRAIN.FISRT_STAGE_EPOCHS    = 20
# __C.TRAIN.SECOND_STAGE_EPOCHS   = 30