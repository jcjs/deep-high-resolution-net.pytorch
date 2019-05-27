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

import time
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


def save_data(img, img_fname, coords_list, scores, output_path):
    '''
    :param img:
    :param coords_list:
    :param scores:
    :return:
    '''

    new_path = '{}/{}_poses.jpg'.format(output_path, os.path.splitext(os.path.basename(img_fname))[0])

    if  os.path.exists(new_path):  # accumulate instances
        img = mmcv.imread(new_path)

    for idx, coords in enumerate(coords_list):
        if idx in [1, 2, 3, 4]: continue

        cv2.circle(img, coords, 4, (0, 255, 255), -1)
        cv2.putText(img, str(idx)+': '+str(scores[0][idx][0])[:3], (coords[0]+5, coords[1]),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    line_color = (255, 255, 0)
    line_thickness = 2
    # cv2.line(img, coords_list[0], coords_list[5], line_color, line_thickness)
    # cv2.line(img, coords_list[0], coords_list[6], line_color, line_thickness)

    # cv2.line(img, coords_list[5], coords_list[6], line_color, line_thickness)
    # cv2.line(img, coords_list[5], coords_list[7], line_color, line_thickness)
    # cv2.line(img, coords_list[5], coords_list[11], line_color, line_thickness)
    # cv2.line(img, coords_list[6], coords_list[12], line_color, line_thickness)
    # cv2.line(img, coords_list[6], coords_list[8], line_color, line_thickness)
    # cv2.line(img, coords_list[7], coords_list[9], line_color, line_thickness)
    # cv2.line(img, coords_list[8], coords_list[10], line_color, line_thickness)
    # cv2.line(img, coords_list[11], coords_list[12], line_color, line_thickness)
    # cv2.line(img, coords_list[11], coords_list[13], line_color, line_thickness)
    # cv2.line(img, coords_list[12], coords_list[14], line_color, line_thickness)
    # cv2.line(img, coords_list[13], coords_list[15], line_color, line_thickness)
    # cv2.line(img, coords_list[14], coords_list[16], line_color, line_thickness)


    mmcv.imwrite(img, new_path)

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def forward_and_parse(outputs, bboxes_data, input_path, output_path, save_images):
    '''
    :param outputs:
    :param bboxes_data:
    :param input_path:
    :return:
    '''
    prev_fname = None
    result_dict = dict()
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

        basefname = bboxes_data['img_fnames'][idx]
        img_fname = '{}/{}'.format(input_path, basefname)
        full_img = mmcv.imread(img_fname)

        lefttop_point = bboxes_data['lefttop_points'][idx]
        coords_list = list()

        for idx, kp in enumerate(keypoints[0]):
            bbox_coords = (int(kp[0] * bbox_shape[0]/heatmap_shape[0]), int(kp[1] * bbox_shape[1]/heatmap_shape[1]))
            full_img_coords = (bbox_coords[0] + lefttop_point[0], bbox_coords[1] + lefttop_point[1])
            coords_list.append(full_img_coords)

        if basefname not in list(result_dict):
            result_dict[basefname] = dict(keypoints=coords_list, scores=[str(s[0])[:5] for s in scores[0]])
        else:
            result_dict[basefname]['keypoints'] += coords_list
            result_dict[basefname]['scores'] += [str(s[0])[:5] for s in scores[0]]

        if save_images:
            save_data(full_img, img_fname, coords_list, scores, output_path)  # debug

    # Save JSON file
    with open('{}_keypoints.json'.format(time.strftime("%Y%m%d%H%M%S")), 'w') as out_file:
        json.dump(result_dict, out_file)


def main():
    '''
    Usage example:  python3 run_on_custom_patches.py -i ./input/images -js ./input/json/20190521073448_detection_bboxes.json -o ./pose_results
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="Path to directory containing images OR path to a single image",
                        required=True)
    parser.add_argument("-js", "--input_json", help="Path to JSON containing detections",
                        required=True)
    parser.add_argument("-o", "--output_path", help="Desired output dir",
                        required=True)
    parser.add_argument("-s", "--save_images", help="True/False",
                        default=False, required=False)
    args = parser.parse_args()

    # Params
    config_file = '../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
    cfg_mmcv = mmcv.Config.fromfile('../lib/config/mmcv_config.py')
    checkpoint = '../lib/models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth'

    input_path = args.input_path
    json_path = args.input_json
    output_path = args.output_path
    save_images = eval(args.save_images)

    # Trying to simplify bullshit cfg parsing without modifying get_pose_net() method
    cfg.cfg = config_file
    cfg.opts, cfg.modelDir, cfg.logDir, cfg.dataDir = [], '', '', ''
    update_config(cfg, cfg)  # lol
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

    forward_and_parse(outputs, bboxes_data, input_path, output_path, save_images)



if __name__ == '__main__':
    main()
