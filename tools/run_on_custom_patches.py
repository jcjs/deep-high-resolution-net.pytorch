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
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from core import inference
from utils.utils import create_logger

import dataset
import models

import cv2
import mmcv
from mmdet.apis import inference_detector
from mmdet.datasets import to_tensor

import numpy as np

# "keypoints": [
#             "nose","left_eye","right_eye","left_ear","right_ear",
#             "left_shoulder","right_shoulder","left_elbow","right_elbow",
#             "left_wrist","right_wrist","left_hip","right_hip",
#             "left_knee","right_knee","left_ankle","right_ankle"
#         ],


def main():
    # Params
    config_file = '../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
    checkpoint = '../lib/models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth'
    # checkpoint = '../lib/models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth'
    input_img_path = 'test_images/body5.jpg'

    # Simplifying bullshit arg parsing (not touching get_pose_net() method)
    args = argparse.ArgumentParser().parse_args()
    args.cfg, args.opts, args.modelDir, args.logDir, args.dataDir = config_file, [], None, None, None
    update_config(cfg, args)
    ##

    model = models.pose_hrnet.get_pose_net(cfg, is_train=False)
    model.load_state_dict(torch.load(checkpoint), strict=False)

    cfg_mmcv = mmcv.Config.fromfile('../lib/config/mmcv_config.py')

    img = mmcv.imread(input_img_path)

    model_img_width = cfg.MODEL.IMAGE_SIZE[1]
    model_img_height = cfg.MODEL.IMAGE_SIZE[0]

    im_width = img.shape[1]
    im_height = img.shape[0]

    wh_ratio = im_width / im_height
    hw_ratio = im_height / im_width

    model_size = (model_img_width, int(model_img_width / wh_ratio))

    # Resize width to model_img_width, keeping aspect ratio
    cfg_mmcv.data.test.img_scale = model_size

    output = inference_detector(model, img, cfg_mmcv)
    batch_heatmaps = output.clone().cpu().numpy()
    hm_width = batch_heatmaps.shape[3]
    hm_height = batch_heatmaps.shape[2]

    img = cv2.resize(img, model_size, interpolation=cv2.INTER_LANCZOS4)
    im_height = img.shape[0]
    im_width = img.shape[1]

    coords, scores = inference.get_max_preds(batch_heatmaps)

    # Debug visualization
    print(scores)
    for idx, kp in enumerate(coords[0]):
        tf_coords = (int(kp[0]*im_height/hm_height), int(kp[1]*im_width/hm_width))
        print(idx, kp, tf_coords)
        cv2.circle(img, tf_coords, 4, (0, 255, 0), -1)
        cv2.putText(img, str(idx) + ': ' + str(scores[0][idx][0])[:5], (tf_coords[0]+5, tf_coords[1]),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ##


if __name__ == '__main__':
    main()
