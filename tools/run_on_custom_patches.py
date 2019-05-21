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



def read_bboxes(jspath, imgpath, expand=1.5):
    '''
    :param jspath:
    :param imgpath:
    :return:
    '''
    with open(jspath, 'r') as f:
        json_data = json.load(f)
    # print(json_data, type(json_data))

    image_size = json_data['image_size']
    results_dict = json_data['results']
    bboxes = list()

    for img_fname in results_dict.keys():
        img = mmcv.imread('{}/{}'.format(imgpath, img_fname))
        for r in results_dict[img_fname]:
            lt, rb = r['bbox']['lt'], r['bbox']['rb'] # left top, right bottom corners
            lt = [int(dim*(1-expand/100)) for dim in lt]
            rb = [int(dim*(1+expand/100)) for dim in rb]
            bbox = img[lt[1]:rb[1], lt[0]:rb[0]]
            bboxes.append(bbox)
            # mmcv.imshow(bbox)

    return dict(image_size=image_size, bboxes=bboxes)


def calc_avg_aspect_ratio(bboxes):
    pass


def main():
    # Params
    config_file = '../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
    checkpoint = '../lib/models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth'
    input_path = './input/images'
    json_path = './input/json/20190521073448_detection_bboxes.json'

    # Simplifying bullshit arg parsing (not touching get_pose_net() method)
    args = argparse.ArgumentParser().parse_args()
    args.cfg, args.opts, args.modelDir, args.logDir, args.dataDir = config_file, [], None, None, None
    update_config(cfg, args)
    ##

    bboxes_data = read_bboxes(json_path, input_path)
    bboxes = bboxes_data['bboxes']

    image_size = bboxes_data['image_size']
    full_img_width = image_size[1]
    full_img_height = image_size[0]

    # We will resize input image to match model_img_width, keeping aspect ratio
    # model_img_width = cfg.MODEL.IMAGE_SIZE[1]
    # input_img_width = input_img.shape[1]
    # input_img_height = input_img.shape[0]
    # wh_ratio = input_img_width / input_img_height
    # model_size = (model_img_width, int(model_img_width / wh_ratio))

    cfg_mmcv = mmcv.Config.fromfile('../lib/config/mmcv_config.py')
    cfg_mmcv.data.test.img_scale = (cfg.MODEL.IMAGE_SIZE[1], int(2.75*cfg.MODEL.IMAGE_SIZE[1]))

    model = models.pose_hrnet.get_pose_net(cfg, is_train=False)
    model.load_state_dict(torch.load(checkpoint), strict=False)

    # if os.path.isdir(input_path):
    #     img_fnames = glob.glob('{}/*.jpg'.format(input_path))
    #     outputs = inference_detector(model, img_fnames, cfg_mmcv)
    # elif os.path.isfile(input_path):
    #     img_fnames = [input_path]
    #     outputs = [inference_detector(model, input_path, cfg_mmcv)]
    # else:
    #     raise Exception('Provided image path is not a file or directory.')

    if len(bboxes) > 1:
        outputs = inference_detector(model, bboxes, cfg_mmcv)
    elif len(bboxes) == 1:
        outputs = [inference_detector(model, bboxes, cfg_mmcv)]
    else:
        raise Exception('There are no bboxes!')


    # img_sizes = [mmcv.imread(img).shape for img in img_fnames]

    # Forward pass
    # output = [inference_detector(model, input_img, cfg_mmcv)]
    # batch_heatmaps = output.clone().cpu().numpy()


    for idx, output in enumerate(list(outputs)):
        if isinstance(output, tuple):
            heatmaps, _ = output
        else:
            heatmaps, _ = output, None

        input_img = mmcv.imread(bboxes[idx])
        input_img_width = input_img.shape[1]
        input_img_height = input_img.shape[0]

        heatmaps = heatmaps.clone().cpu().numpy()
        coords, scores = inference.get_max_preds(heatmaps)
        print(scores)

        # Transform back coords, matching input image
        hm_width = heatmaps.shape[3]
        hm_height = heatmaps.shape[2]

        for idx, kp in enumerate(coords[0]):
            tf_coords = (int(kp[0]*input_img_height/hm_height), int(kp[1]*input_img_width/hm_width))

            # Debug visualization
            print(idx, kp, tf_coords)
            cv2.circle(input_img, tf_coords, 4, (0, 255, 0), -1)
            cv2.putText(input_img, str(idx) + ': ' + str(scores[0][idx][0])[:5], (tf_coords[0]+5, tf_coords[1]),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', input_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ##


if __name__ == '__main__':
    main()
    # read_bboxes('./input/json/20190521073448_detection_bboxes.json', './input/images')
