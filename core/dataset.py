#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2021/03/25 20:42:32
@Author  :   Zol 
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import core.utils as utils
from core.config import cfg

import os
import cv2
import random
import numpy as np
import tensorflow as tf

# class Dataset(object):
    
#     def __init__(self, FLAGS, is_training:bool, dataset_type: str="converted_coco"):
#         self.strides, self.anchors, NUM_LASS, XYSCALE = utils.load_config(FLAGS)