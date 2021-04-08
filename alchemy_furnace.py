#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   alchemy_furnace.py
@Time    :   2021/03/31 05:38:36
@Author  :   Zol
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
from core.dataset import Dataset
from core.config import cfg
from core import utils
from core.utils import freeze_all, unfreeze_all, read_class_names
from core.yolov4 import Yolo_v4, decode_train, compute_loss

import os
import shutil
import numpy as np
import tensorflow as tf
from absl.flags import FLAGS
from absl import app, flags

flags.DEFINE_string('model', 'yolov4', 'yolov4')
flags.DEFINE_string('weights', os.path.join('.', 'yolo.weights'), 'pretrained weights')


class Alchemy_furnace:
    def __init__(self, weights_path, model_name='yolov4'):

        self.model_name = model_name

        self.log_dir = os.path.join('.', 'log')

        self.weights_path = weights_path
        self.train_annot_path = cfg.TRAIN.ANNOT_PATH  # 训练数据信息txt文件路径
        self.test_annot_path = cfg.TEST.ANNOT_PATH  # 测试数据信息txt文件路径

        self.input_size = cfg.TRAIN.INPUT_SIZE
        self.test_input_size = cfg.TEST.INPUT_SIZE

        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE  # 每个scale的anchor数量
        self.anchors = np.reshape(np.array(cfg.YOLO.ANCHORS), (3, 3, 2))  # anchor_per_scale 对应shape中的3，需更改

        self.classes_name = read_class_names(cfg.YOLO.CLASSES)  # 所有分类名
        self.num_class = len(self.classes_name)  # 分类名数量

        self.strides = np.array(cfg.YOLO.STRIDES)  # 3个不同输出大小对应输入大小的缩小倍数
        self.xy_scale = cfg.YOLO.XYSCALE  # 用于计算pred_xy，暂时没搞懂干什么用

        self.train_batch_size = cfg.TRAIN.BATCH_SIZE
        self.test_batch_size = cfg.TEST.BATCH_SIZE

        self.train_set = Dataset(self.input_size,
                                 self.train_batch_size,
                                 self.anchors,
                                 self.strides,
                                 self.train_annot_path,
                                 self.num_class,
                                 self.anchor_per_scale,
                                 data_aug=True)
        self.test_set = Dataset(self.test_input_size,
                                self.test_batch_size,
                                self.anchors,
                                self.strides,
                                self.test_annot_path,
                                self.num_class,
                                self.anchor_per_scale,
                                data_aug=False)

        self.warmup_epochs = cfg.TRAIN.WARMUP_EPOCHS
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS

        self.steps_per_epoch = self.train_set.num_batchs
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch  # 预热学习率的steps数
        self.total_steps = (self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_epoch

        self.is_freeeze = False
        self.freeze_layers = ['conv2d_93', 'conv2d_101', 'conv2d_109']

        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH

        self.optimizer = tf.keras.optimizers.Adam()

    def train(self):
        """开始炼丹
        """

        # 输入层
        input_layer = tf.keras.layers.Input([self.input_size, self.input_size, 3])

        feature_maps = Yolo_v4(input_layer, self.num_class)

        bbox_tensors = []
        for i, feature_map in enumerate(feature_maps):  # 处理3个feature得到最后的输出分类层
            if i == 0:
                bbox_tensor = decode_train(feature_map, cfg.TRAIN.INPUT_SIZE // 8, self.num_class, self.strides,
                                           self.anchors, i, self.xy_scale)
            elif i == 1:
                bbox_tensor = decode_train(feature_map, cfg.TRAIN.INPUT_SIZE // 16, self.num_class, self.strides,
                                           self.anchors, i, self.xy_scale)
            else:
                bbox_tensor = decode_train(feature_map, cfg.TRAIN.INPUT_SIZE // 32, self.num_class, self.strides,
                                           self.anchors, i, self.xy_scale)
            bbox_tensors.append(feature_map)
            bbox_tensors.append(bbox_tensor)

        model = tf.keras.Model(input_layer, bbox_tensors)

        tf.keras.utils.plot_model(model,
                                  to_file='modelV4.png',
                                  show_shapes=True,
                                  show_layer_names=True,
                                  rankdir='TB',
                                  dpi=900,
                                  expand_nested=True)

        model.summary()  # 输出参数Param计算过程

        if self.weights_path is None:
            print("Training from scratch")

        else:
            if self.weights_path.split(".")[len(self.weights_path.split(".")) - 1] == "weights":
                utils.load_weights(model, self.weights_path, self.model_name)
            else:
                model.load_weights(FLAGS.weights)
            print('Restoring weights from: %s ... ' % FLAGS.weights)

        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

        writer = tf.summary.create_file_writer(self.log_dir)

        total_epochs = self.first_stage_epochs + self.second_stage_epochs

        for epoch in range(total_epochs):
            if epoch < self.first_stage_epochs:
                if not self.is_freeeze:
                    is_freeeze = True
                    for name in self.freeze_layers:
                        freeze = model.get_layer(name)
                        freeze_all(freeze)

            elif epoch >= self.first_stage_epochs:
                if is_freeeze:
                    is_freeeze = False
                    for name in self.freeze_layers:
                        freeze = model.get_layer(name)
                        unfreeze_all(freeze)

            for image_data, target in self.train_set:
                self._train_step(image_data, target, model, writer)

            for image_data, target in self.test_set:
                self._test_step(image_data, target, model)

            model.save_weights(os.path.join('.', 'yolov4'))

    def _train_step(self, image_data, target, model, writer):
        """训练步骤

        Args:
            image_data : 图片数据
            target : label和bboxes的信息
            model : 模型
            writer : tf.summary.create_file_writer
        """
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            for i in range(len(self.freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred,
                                          conv,
                                          target[i][0],
                                          target[i][1],
                                          strides=self.strides,
                                          num_class=self.num_class,
                                          iou_loss_thresh=self.iou_loss_thresh,
                                          i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)

            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" %
                     (self.global_steps, self.total_steps, self.optimizer.lr.numpy(), giou_loss, conf_loss, prob_loss,
                      total_loss))

            # 更新学习率
            self.global_steps.assign_add(1)

            if self.global_steps < self.warmup_steps:  # 先以小的学习率热身，慢慢随着setp的增大而增大
                lr = self.global_steps / self.warmup_steps * cfg.TRAIN.LR_INIT

            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * ((1 + tf.cos(
                    (self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi)))

            self.optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", self.optimizer.lr, step=self.global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=self.global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=self.global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=self.global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=self.global_steps)
            writer.flush()

    def _test_step(self, image_data, target, model):
        """测试步骤

        Args:
            image_data : 图片数据
            target : label和bboxes的信息
            model : 模型
        """
        with tf.GradientTape() as _:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(self.freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred,
                                          conv,
                                          target[i][0],
                                          target[i][1],
                                          strides=self.strides,
                                          num_class=self.num_class,
                                          iou_loss_thresh=self.iou_loss_thresh,
                                          i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" %
                     (self.global_steps, giou_loss, conf_loss, prob_loss, total_loss))


def main(_argv):
    """主函数

    Args:
        _argv : 参数
    """
    pysical_devices = tf.config.experimental.list_physical_devices("GPU")

    if len(pysical_devices) > 0:
        tf.config.experimental.set_memory_growth(pysical_devices[0], True)  # 内存按需分配

    alchemy_furnace = Alchemy_furnace(FLAGS.weights, FLAGS.model)
    alchemy_furnace.train()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
