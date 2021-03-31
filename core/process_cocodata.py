#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   procecc_cocodata.py
@Time    :   2021/03/29 21:52:20
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


class coco_2_txt:
    def __init__(
        self,
        ann_path,
        data_path,
        ouput_path,
        classes_path='',
    ):
        """处理coco数据，保存至txt文件中

        Args:
            ann_path (str): coco ann数据路径
            data_path (str): coco 训练数据路径
            ouput_path (str): 数据信息输出路径  
            classes_path (str, optional): 类别数据输出路径. Defaults to ''.
        """
        self.ann_path = ann_path
        self.data_path = data_path
        self.train_img_paths = []
        self.ouput_path = ouput_path
        self.img_bboxes = {}
        self.categories_dict = {}
        self.valid_categories_list = []
        self.classes_path = classes_path

    def process_cocodata(self):

        coco = COCO(self.ann_path)
        categories = coco.loadCats(coco.getCatIds())  # 获取coco的分类和超分类

        for c in categories:
            c_id = c['id']
            c_name = c['name']
            self.categories_dict[c_id] = c_name  # 存储coco的分类id，和对应的分类名称

        image_ids = coco.getImgIds()

        train_img_names = os.listdir(self.data_path)

        for img_name in train_img_names:  # 查找训练集的图片是否都有对应的ID，并保存到一个列表中

            # int(img_name[0:-4]) 是去掉文件名的后缀和前面的0
            if int(img_name[0:-4]) in image_ids:
                img_path = os.path.join(self.data_path, img_name)
                self.train_img_paths.append(img_path)

        if os.path.exists(self.ouput_path):
            os.remove(self.ouput_path)
            
        with open(self.ouput_path, "w") as f:

            for item in self.train_img_paths:
                boxes = []
                img_name = item.split('\\')[-1]
                imgid = int(img_name[0:-4])
                annIds = coco.getAnnIds(imgIds=imgid, iscrowd=None)
                anns = coco.loadAnns(annIds)

                f.write(item)

                for index, ann in enumerate(anns):

                    catid = ann['category_id']
                    if self.categories_dict[
                            catid] not in self.valid_categories_list:

                        self.valid_categories_list.append(
                            self.categories_dict[catid])

                    bbox = ann['bbox']
                    xmin = int(bbox[0])
                    xmax = int(bbox[0] + bbox[2])
                    ymin = int(bbox[1])
                    ymax = int(bbox[1] + bbox[3])

                    catid_name = self.categories_dict[catid]
                    valid_catid = self.valid_categories_list.index(catid_name)
                    boxes.append(xmin)
                    boxes.append(ymin)
                    boxes.append(xmax)
                    boxes.append(ymax)
                    boxes.append(valid_catid)
                    info = ','.join([
                        str(xmin),
                        str(ymin),
                        str(xmax),
                        str(ymax),
                        str(valid_catid)
                    ])

                    if index == 0:
                        f.write(" " + info)
                    else:
                        f.write("," + info)
                f.write("\n")
                self.img_bboxes[item] = boxes

        if classes_path != '':

            if os.path.exists(self.classes_path):
                os.remove(self.classes_path)

            with open(self.classes_path, 'w') as f:
                for class_name in self.valid_categories_list:
                    f.write(class_name + '\n')


if __name__ == '__main__':
    data_set_name = 'val'
    ann_path = os.path.join('F:\\', 'data', 'annotations',
                            'instances_' + data_set_name + '2017.json')
    train_data_path = os.path.join('F:\\', 'data', 'coco2017',
                                   data_set_name + '2017')
    data_output_path = os.path.join('F:\\', 'data', 'output',
                                    data_set_name + '.txt')
    classes_path = os.path.join('F:\\', 'data', 'output', 'classes.txt')
    coco_process = coco_2_txt(ann_path, train_data_path, data_output_path,
                              classes_path)
    coco_process.process_cocodata()