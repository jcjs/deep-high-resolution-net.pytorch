# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str)

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


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def main():
    args = parse_args()
    args.cfg = '../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
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

    #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    cfg_mmcv = mmcv.Config.fromfile('../lib/config/mmcv_dummy.py')
    # cfg_mmcv.model.pretrained = None

    # model = model.to('cuda:5').eval()
    #
    img = mmcv.imread('test_images/body4.jpg')

    im_height = img.shape[0]
    im_width = img.shape[1]
    wh_ratio = im_width / im_height
    hw_ratio = im_height / im_width

    model_img_height = 288
    model_img_width = 384

    # img = cv2.resize(img, (int(model_img_height / hw_ratio), model_img_height),
    #                  interpolation=cv2.INTER_CUBIC if im_height < model_img_height else cv2.INTER_AREA)

    img = cv2.resize(img, (model_img_width, int(model_img_width / wh_ratio)),
                     interpolation=cv2.INTER_CUBIC if im_width < model_img_width else cv2.INTER_AREA)

    # img = cv2.copyMakeBorder(img, int((1333-im_height)/2), int((1333-im_height)/2), int((800-im_width)/2),
    #                          int((800-im_width)/2), cv2.BORDER_CONSTANT, value=[0, 0, 0])

    im_height = img.shape[0]
    im_width = img.shape[1]

    output = inference_detector(model, img, cfg_mmcv)
    batch_heatmaps = output.clone().cpu().numpy()

    hm_height = batch_heatmaps.shape[2]
    hm_width = batch_heatmaps.shape[3]

    preds = get_max_preds(batch_heatmaps)

    print(preds, preds[1])

    # Draw points
    #print(img.shape)
    # img = img[0].permute(1, 2, 0).cpu().numpy()

    for idx, kp in enumerate(preds[0][0]):
        coords = (int(kp[0]*im_height/hm_height), int(kp[1]*im_width/hm_width))
        cv2.circle(img, coords, 4, (0, 255, 0), -1)
        cv2.putText(img, str(preds[1][0][idx][0])[:5], (coords[0]+5, coords[1]), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 255),
                    1, cv2.LINE_AA)


    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # define loss function (criterion) and optimizer
    # criterion = JointsMSELoss(
    #     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    # ).cuda()
    #
    # # Data loading code
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    # valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    #     shuffle=False,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=True
    # )
    #
    # # evaluate on validation set
    # validate(cfg, valid_loader, valid_dataset, model, criterion,
    #          final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
