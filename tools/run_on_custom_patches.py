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

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default='../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.cfg = '../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
    # args.cfg = '../experiments/coco/resnet/res152_384x288_d256x3_adam_lr1e-3.yaml'
    # args.cfg = '../experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml'
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    cfg_mmcv = mmcv.Config.fromfile('../lib/config/mmcv_config.py')
    # cfg_mmcv.model.pretrained = None

    # model = model.to('cuda:5').eval()
    #
    img = mmcv.imread('test_images/body2.jpg')

    im_height, im_width = img.shape[0], img.shape[1]
    wh_ratio = im_width / im_height
    hw_ratio = im_height / im_width
    model_img_width, model_img_height = 384, 288
    model_size = (model_img_width, int(model_img_width / wh_ratio))

    # img = cv2.resize(img, (int(model_img_height / hw_ratio), model_img_height),
    #                  interpolation=cv2.INTER_CUBIC if im_height < model_img_height else cv2.INTER_AREA)

    # Resize width to model_img_width, keeping aspect ratio

    # Resize width to model_img_width, keeping aspect ratio
    cfg_mmcv.data.test.img_scale = model_size

    output = inference_detector(model, img, cfg_mmcv)
    batch_heatmaps = output.clone().cpu().numpy()

    hm_width, hm_height = batch_heatmaps.shape[3], batch_heatmaps.shape[2]

    # Assumption: input image is already a BBOX: [top left x position, top left y position, width, height]
    coords, scores = inference.get_max_preds(batch_heatmaps)
    bbox = (0, 0, im_width, im_height)
    # coords, scores = inference.get_final_preds(batch_heatmaps, bbox)

    img = cv2.resize(img, model_size, interpolation=cv2.INTER_LANCZOS4)
    im_height, im_width = img.shape[0], img.shape[1]

    print(scores)
    # Visualization
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


if __name__ == '__main__':
    main()
