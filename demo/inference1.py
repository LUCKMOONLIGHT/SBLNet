import argparse
import os.path as osp
import mmcv
import os
import cv2
import math
import copy
import torch
import numpy as np
from PIL import Image, ImageDraw
from mmdet.apis import init_detector, inference_detector
from mmdet.core import multiclass_nms
from draw_box_in_img import draw_boxes_with_label_and_scores

ODAI_LABEL_MAP = {
        'back-ground': 0,
        'plane': 1,
        'baseball-diamond': 2,
        'bridge': 3,
        'ground-track-field': 4,
        'small-vehicle': 5,
        'large-vehicle': 6,
        'ship': 7,
        'tennis-court': 8,
        'basketball-court': 9,
        'storage-tank': 10,
        'soccer-ball-field': 11,
        'roundabout': 12,
        'harbor': 13,
        'swimming-pool': 14,
        'helicopter': 15,
    }

def get_label_name_map():
    reverse_dict = {}
    for name, label in ODAI_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict



def osp(savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

if __name__ == '__main__':
    from mmdet.apis import init_detector, inference_detector
    import mmcv
    from PIL import Image

    config_file = '/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/configs/myconfig/test_libra_faster_rcnn_r50_fpn_1x_sp.py'
    checkpoint_file = '/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/work_dirs/libra_faster_rcnn_r50_fpn_1x_sp_trainval/v1_epoch_40.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    # img = '/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/data/test_sp/images/P0063__1__0___0.png'  # or img = mmcv.imread(img), which will only load it once
    img = '/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/data/test_sp/images/P0063__1__2304___256.png'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    bboxes = np.vstack(result)
    labels = [  # 0-15
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    if len(bboxes) > 0:

        outimg = Image.open(img).convert('RGB')
        outimg = np.array(outimg)
        image = draw_boxes_with_label_and_scores(outimg, bboxes[:, :4], bboxes[:, 4], labels, 0)
        image.save('P0063__1__2304___256_show' + '.png')

        CLASS_DOTA = ODAI_LABEL_MAP.keys()
        LABEl_NAME_MAP = get_label_name_map()
        write_handle_r = {}

        for sub_class in CLASS_DOTA:
            if sub_class == 'back-ground':
                continue
            write_handle_r[sub_class] = open(('Task2_%s.txt' % sub_class), 'a+')

        boxes = []

        for rect in bboxes[:, :4]:
            boxes.append([rect[0], rect[1], rect[2], rect[3]])

        rboxes = np.array(boxes, dtype=np.float32)

        for i, rbox in enumerate(rboxes):
            command = '%s %.5f %.5f %.5f %.5f %.5f\n' % (img.split('/')[-1][:-4], bboxes[i, 4], rbox[0],
                                                         rbox[1], rbox[2], rbox[3])

            write_handle_r[LABEl_NAME_MAP[int(labels[i])+1]].write(command)

        for sub_class in CLASS_DOTA:
            if sub_class == 'back-ground':
                continue
            write_handle_r[sub_class].close()
