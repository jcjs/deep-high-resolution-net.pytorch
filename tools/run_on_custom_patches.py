# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
# from core.loss import JointsMSELoss
# from core.function import validate
from core import inference
# from utils.utils import create_logger

# import dataset
import models

import cv2
import mmcv
from mmdet.apis import inference_detector
# from mmdet.datasets import to_tensor

import json
import glob
import numpy as np

# "keypoints": [
#             "nose","left_eye","right_eye","left_ear","right_ear",
#             "left_shoulder","right_shoulder","left_elbow","right_elbow",
#             "left_wrist","right_wrist","left_hip","right_hip",
#             "left_knee","right_knee","left_ankle","right_ankle"
#         ],


def read_bboxes(jspath, input_path, expand=2.0):
    '''
    :param jspath:
    :param input_path:
    :param expand:
    :return:
    '''
    with open(jspath, 'r') as f:
        json_data = json.load(f)

    image_size = json_data['image_size']
    results_dict = json_data['results']
    bboxes = list()
    img_fnames = list()
    lefttop_points = list()

    for img_fname in results_dict.keys():
        img = mmcv.imread('{}/{}'.format(input_path, img_fname))
        for r in results_dict[img_fname]:
            lt, rb = r['bbox']['lt'], r['bbox']['rb']  # left top, right bottom corners
            lt = [int(dim*(1-expand/100)) for dim in lt]  # apply expansion
            rb = [int(dim*(1+expand/100)) for dim in rb]

            bbox = img[lt[1]:rb[1], lt[0]:rb[0]]  # crop
            bboxes.append(bbox)
            img_fnames.append(img_fname)
            lefttop_points.append(lt)
            # mmcv.imshow(bbox)  # debug

    return dict(img_fnames=img_fnames, image_size=image_size, bboxes=bboxes, lefttop_points=lefttop_points)


def calc_avg_aspect_ratio(bboxes):
    '''
    :param bboxes:
    :return: the averaged height/width ratio
    '''
    return sum(bbox.shape[0] for bbox in bboxes) / sum(bbox.shape[1] for bbox in bboxes)


def plot_result(img, coords_list, scores):
    '''
    :param img:
    :param coords_list:
    :param scores:
    :return:
    '''
    for idx, coords in enumerate(coords_list):
        # Debug visualization
        cv2.circle(img, coords, 4, (0, 255, 0), -1)
        cv2.putText(img, str(idx)+': '+str(scores[0][idx][0])[:5], (coords[0]+5, coords[1]),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_final_results(outputs, bboxes_data, input_path):
    '''
    :param outputs:
    :param bboxes_data:
    :param input_path:
    :return:
    '''
    for idx, output in enumerate(list(outputs)):
        if isinstance(output, tuple):
            heatmaps, _ = output
        else:
            heatmaps, _ = output, None

        heatmaps = heatmaps.clone().cpu().numpy()
        heatmap_shape = (heatmaps.shape[2], heatmaps.shape[3])

        keypoints, scores = inference.get_max_preds(heatmaps)

        bboxes = bboxes_data['bboxes']
        bbox_img = mmcv.imread(bboxes[idx])
        bbox_shape = bbox_img.shape[:2]

        img_fname = bboxes_data['img_fnames'][idx]
        full_img = mmcv.imread('{}/{}'.format(input_path, img_fname))
        full_img_shape = full_img.shape[:2]

        lefttop_point = bboxes_data['lefttop_points'][idx]

        coords_list = list()
        for idx, kp in enumerate(keypoints[0]):
            bbox_coords = (int(kp[0] * bbox_shape[0]/heatmap_shape[0]), int(kp[1] * bbox_shape[1]/heatmap_shape[1]))
            full_img_coords = (bbox_coords[0] + lefttop_point[0], bbox_coords[1] + lefttop_point[1])
            coords_list.append(full_img_coords)

        plot_result(full_img, coords_list, scores)  # debug


def main():
    # Params
    config_file = '../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
    cfg_mmcv = mmcv.Config.fromfile('../lib/config/mmcv_config.py')
    checkpoint = '../lib/models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth'
    input_path = './input/images'
    json_path = './input/json/20190521073448_detection_bboxes.json'

    # Simplifying bullshit arg parsing (not touching get_pose_net() method)
    args = argparse.ArgumentParser().parse_args()
    args.cfg = config_file
    args.opts, args.modelDir, args.logDir, args.dataDir = [], None, None, None
    update_config(cfg, args)
    ##

    bboxes_data = read_bboxes(json_path, input_path)
    bboxes = bboxes_data['bboxes']

    model_input_width = cfg.MODEL.IMAGE_SIZE[1]
    height_scale = calc_avg_aspect_ratio(bboxes)
    cfg_mmcv.data.test.img_scale = (model_input_width, int(height_scale*model_input_width))

    model = models.pose_hrnet.get_pose_net(cfg, is_train=False)
    model.load_state_dict(torch.load(checkpoint), strict=False)

    if len(bboxes) > 1:
        outputs = inference_detector(model, bboxes, cfg_mmcv)
    elif len(bboxes) == 1:
        outputs = [inference_detector(model, bboxes, cfg_mmcv)]
    else:
        raise Exception('There are no bboxes!')

    save_final_results(outputs, bboxes_data, input_path)



if __name__ == '__main__':
    main()
    # read_bboxes('./input/json/20190521073448_detection_bboxes.json', './input/images')
