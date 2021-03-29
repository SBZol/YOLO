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
from config import cfg
from utils import myThread

import os
import sys
import math
import queue
import threading
import time

import cv2
import tensorflow as tf
from pycocotools.coco import COCO


class coco_2_tfrecord:
    def __init__(
        self,
        ann_path,
        train_data_path,
        ouput_path,
    ):
        self.ann_path = ann_path
        self.train_data_path = train_data_path
        self.train_img_paths = []
        self.ouput_path = ouput_path
        self.img_bboxes = {}
        self.categories_dict = {}
        self.valid_categories_list = []

    def process_cocodata(self):

        coco = COCO(self.ann_path)
        categories = coco.loadCats(coco.getCatIds())  # 获取coco的分类和超分类

        for c in categories:
            c_id = c['id']
            c_name = c['name']
            self.categories_dict[c_id] = c_name  # 存储coco的分类id，和对应的分类名称

        image_ids = coco.getImgIds()

        train_img_names = os.listdir(self.train_data_path)

        for img_name in train_img_names:  # 查找训练集的图片是否都有对应的ID，并保存到一个列表中
            
            # int(img_name[0:-4]) 是去掉文件名的后缀和前面的0
            if int(img_name[0:-4]) in image_ids:  
                img_path = os.path.join(self.train_data_path, img_name)
                self.train_img_paths.append(img_path)

        for item in self.train_img_paths:
            boxes = []
            img_name = item.split('\\')[-1]
            imgid = int(img_name[0:-4])
            annIds = coco.getAnnIds(imgIds=imgid, iscrowd=None)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                catid = ann['category_id']
                if self.categories_dict[catid]  not in self.valid_categories_list:
                    self.valid_categories_list.append(self.categories_dict[catid])
                bbox = ann['bbox']
                xmin = int(bbox[0])
                xmax = int(bbox[0] + bbox[2])
                ymin = int(bbox[1])
                ymax = int(bbox[1] + bbox[3])
                catid = ann['category_id']
                catid_name = self.categories_dict[catid]
                valid_catid = self.valid_categories_list.index(catid_name)
                boxes.append(xmin)
                boxes.append(ymin)
                boxes.append(xmax)
                boxes.append(ymax)
                boxes.append(valid_catid)
                
            self.img_bboxes[item] = boxes
            with open(output_path, "w") as f:
                for file_name in self.img_bboxes.keys():
                    f.write(file_name + ' ')
                    for img_bbox in self.img_bboxes[file_name]:
                        f.write(str(img_bbox) + ",")
                    f.write("\n")

                

    
if __name__ == '__main__':
    ann_path = os.path.join('F:\\', 'data', 'annotations','instances_val2017.json')
    train_data_path = os.path.join('F:\\', 'data', 'coco2017', 'val2017')
    output_path = os.path.join('F:\\', 'data', 'output', 'train.txt')
    coco_process = coco_2_tfrecord(ann_path,train_data_path,output_path)
    coco_process.process_cocodata()
    
    
  
   