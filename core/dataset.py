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

import os
import random
import numpy as np
from cv2 import cv2
import tensorflow as tf


class Dataset(object):
    def __init__(self,
                 input_size,
                 batch_size,
                 anchors,
                 strides,
                 annot_path,
                 num_class,
                 anchor_per_scale,
                 data_aug=True):
        """获取数据集

        Args:
            input_size (int): 网络输入大小
            batch_size (int): batch size
            anchors (np): 3个不同的scale对应的anchors
            strides (np): input和ouput的缩放倍数
            annot_path (str): 解析数据的txt文件路径
            num_class(int): 所有类别数量
            anchor_per_scale(int): 每个scale的anchor数量
            data_aug (bool): 是否需要数据增广 Defaults to True.
        """
        self.input_size = input_size

        self.batch_size = batch_size

        self.anchors = anchors

        self.strides = strides

        self.annot_path = annot_path

        self.data_aug = data_aug

        self.input_sizes = input_size

        self.output_sizes = self.input_size // self.strides

        self.num_classes = num_class

        self.anchor_per_scale = anchor_per_scale

        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()

        self.num_samples = len(self.annotations)

        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))

        self.batch_count = 0

    def load_annotations(self):
        """解析annotation.txt文件

        Returns:
            annotations: 解析后的信息数组
        """
        with open(self.annot_path, "r") as f:
            txt = f.readlines()

            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]

        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):

            batch_image = np.zeros(
                (
                    self.batch_size,
                    self.input_size,
                    self.input_size,
                    3,
                ),
                dtype=np.float32,
            )

            batch_label_sbbox = np.zeros(
                (
                    self.batch_size,
                    self.output_sizes[0],
                    self.output_sizes[0],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_mbbox = np.zeros(
                (
                    self.batch_size,
                    self.output_sizes[1],
                    self.output_sizes[1],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_lbbox = np.zeros(
                (
                    self.batch_size,
                    self.output_sizes[2],
                    self.output_sizes[2],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)

                    (
                        label_sbbox,
                        label_mbbox,
                        label_lbbox,
                        sbboxes,
                        mbboxes,
                        lbboxes,
                    ) = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                return (
                    batch_image,
                    (
                        batch_smaller_target,
                        batch_medium_target,
                        batch_larger_target,
                    ),
                )
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):

        line = annotation.split()
        img_path = line[0]

        if not os.path.exists(img_path):
            raise KeyError("%s does not exist ... " % img_path)

        image = cv2.imread(img_path)

        bboxes = np.array([list(map(int, box.split(","))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))

            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))

            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 颜色空间转换函数，BGR转换成RGB
        image, bboxes = utils.image_preprocess(
            np.copy(image),
            [self.input_size, self.input_size],
            np.copy(bboxes),
        )
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        label = [
            np.zeros((
                self.output_sizes[i],
                self.output_sizes[i],
                self.anchor_per_scale,
                5 + self.num_classes,
            )) for i in range(3)
        ]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3, ))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0

            # *** Label smooothing
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            # ***

            bbox_xywh = np.concatenate(  # 根据box左下右上角坐标算出中心点x,y和宽高w,d
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )

            bbox_xywh_scaled = (1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis])

            iou = []
            exist_positive = False
            for i in range(3):  # 3个scale,生产3个label
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))

                anchors_xywh[:, 0:2] = (  # 给每个anchors的x,y赋值
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5)

                anchors_xywh[:, 2:4] = self.anchors[i]  # 给每个anchors的w,h赋值

                iou_scale = utils.iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)  # 计算每个anchor的iou
                iou.append(iou_scale)

                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):  # 只要iou_mask中有True
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(  # 返回最大值坐标
                    np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


if __name__ == '__main__':
    pass
