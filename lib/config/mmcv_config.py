# dataset settings
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

# img_scale = (384, 288)
img_scale = (1333, 800)

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        # type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        img_scale=img_scale,
        resize_keep_ratio=True,
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=False))
