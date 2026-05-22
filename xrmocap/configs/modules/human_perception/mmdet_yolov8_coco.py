# YOLOv8 configuration for MMDetection
# This config is compatible with mmdet 3.x
# Modified to detect only person class (class_id=0 in COCO)

# model settings
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=512,
        deepen_factor=1.0,
        widen_factor=1.0,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=1.0,
        widen_factor=1.0,
        in_channels=[256, 512, 512],
        out_channels=[256, 512, 512],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=1,  # Only detect person class
            in_channels=[256, 512, 512],
            widen_factor=1.0,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32]),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=1.5 / 4)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=1,  # Only detect person class
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# pipeline settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-320, -320)),
    dict(
        type='YOLOv5MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Mosaic',
                img_scale=(640, 640),
                pad_val=114.0,
                pre_transform=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True)
                ]),
            dict(
                type='mmdet.RandomAffine',
                scaling_ratio_range=(0.5, 1.5),
                border=(-320, -320)),
        ]),
    dict(type='mmdet.YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    dict(type='mmdet.Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataloader settings
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False)

test_evaluator = val_evaluator
