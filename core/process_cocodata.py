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
        ann_paths,
        data_paths,
        output_dir_path,
    ):
        """处理coco数据，保存至txt文件中

        Args:
            ann_paths (list): coco ann数据路径 ([train,val,test])
            data_paths (list): coco 训练数据路径 ([train,val,test])
            output_dir_path (str): 数据信息输出文件夹路径  
        """

        self.train_data_path = data_paths[0]
        self.train_ann_path = ann_paths[0]

        self.val_data_path = data_paths[1]
        self.val_ann_path = ann_paths[1]

        self.test_data_path = data_paths[2]
        self.test_ann_apth = ann_paths[2]

        self.output_dir_path = output_dir_path

        self.img_bboxes = {}
        self.categories_dict = {}
        self.valid_categories_list = []
        self.classes_path = os.path.join(self.output_dir_path, 'classes.txt')

    def get_valid_classes(self):
        """获取coco数据中有效的分类，并输出txt
        """

        coco = COCO(self.val_ann_path)
        categories = coco.loadCats(coco.getCatIds())  # 获取coco的分类和超分类

        for c in categories:
            c_id = c['id']
            c_name = c['name']
            self.categories_dict[c_id] = c_name  # 存储coco的分类id，和对应的分类名称

        image_ids = coco.getImgIds()

        data_img_paths = os.listdir(self.val_data_path)

        for item in data_img_paths:
            boxes = []
            img_name = item.split('\\')[-1]
            imgid = int(img_name[0:-4])
            annIds = coco.getAnnIds(imgIds=imgid, iscrowd=None)
            anns = coco.loadAnns(annIds)

            for index, ann in enumerate(anns):
                catid = ann['category_id']
                if self.categories_dict[
                        catid] not in self.valid_categories_list:
                    self.valid_categories_list.append(
                        self.categories_dict[catid])

        if os.path.exists(self.classes_path):
            os.remove(self.classes_path)

        with open(self.classes_path, 'w') as f:
            for class_name in self.valid_categories_list:
                f.write(class_name + '\n')

    def process_cocodata(self):
        """处理coco data并输出训练所需要的txt文件
        """
        # train data 
        train_output_path = os.path.join(
            self.output_dir_path,
            os.path.split(self.train_data_path)[-1] + '.txt')
        
        self.output_data_info(
            self.train_ann_path,
            self.train_data_path,
            train_output_path
        )
        
        # val data
        val_output_path = os.path.join(
            self.output_dir_path,
            os.path.split(self.val_data_path)[-1] + '.txt')
        
        self.output_data_info(
            self.val_ann_path,
            self.val_data_path,
            val_output_path
        )
        
        # test data
        test_output_path = os.path.join(
            self.output_dir_path,
            os.path.split(self.test_data_path)[-1] + '.txt')
        
        self.output_data_info(
            self.test_ann_apth,
            self.test_data_path,
            test_output_path
        )

    def output_data_info(self, ann_path, data_path, ouput_path):
        """解析并输出数据信息文件

        Args:
            ann_path ([type]): coco数据ann的路径
            data_path ([type]): coco数据image路径
            ouput_path ([type]): 输出的txt文件路径
        """

        coco = COCO(ann_path)

        image_ids = coco.getImgIds()

        train_img_names = os.listdir(data_path)

        valid_img_paths = []  # 有效的图片路径

        for img_name in train_img_names:  # 查找训练集的图片是否都有对应的ID，并保存到一个列表中

            # int(img_name[0:-4]) 是去掉文件名的后缀和前面的0
            if int(img_name[0:-4]) in image_ids:
                img_path = os.path.join(data_path, img_name)
                valid_img_paths.append(img_path)

        if os.path.exists(ouput_path):
            os.remove(ouput_path)

        with open(ouput_path, "w") as f:

            for item in valid_img_paths:
                boxes = []
                img_name = item.split('\\')[-1]
                imgid = int(img_name[0:-4])

                annIds = coco.getAnnIds(imgIds=imgid, iscrowd=None)
                anns = coco.loadAnns(annIds)

                f.write(item)

                for index, ann in enumerate(anns):

                    catid = ann['category_id']
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


if __name__ == '__main__':
    from config import cfg
    train_data_path = cfg.TRAIN.DATA_PATH
    train_ann_path = cfg.TRAIN.ANNOT_PATH

    val_data_path = cfg.VAL.DATA_PATH
    val_ann_path = cfg.VAL.ANNOT_PATH

    test_data_path = cfg.TEST.DATA_PATH
    test_ann_apth = cfg.TEST.ANNOT_PATH

    output_dir_path = cfg.OUTPUT.DIR_PATH

    coco_process = coco_2_txt(
        [train_ann_path, val_ann_path, test_ann_apth],
        [train_data_path, val_data_path, test_data_path],
        output_dir_path,
    )

    coco_process.get_valid_classes()
    coco_process.process_cocodata()