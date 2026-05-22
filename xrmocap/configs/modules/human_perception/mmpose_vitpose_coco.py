# ViTPose configuration for MMPose
# ViTPose-B configuration with COCO dataset

# Model architecture
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3),
    neck=dict(
        type='FeatureMapProcessor',
        select_index=0,
        concat=True),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=17,  # COCO keypoints
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=None,
        conv_kernel_sizes=None,
        has_final_layer=True,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2)),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True))

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# Data pipeline
data_mode = 'topdown'
dataset_info = 'configs/_base_/datasets/coco.py'

# Pipeline
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='GenerateTarget', encoder='MSRAHeatmap'),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]

test_pipeline = val_pipeline

# Data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')

test_evaluator = val_evaluator

# Codec settings for keypoint encoding/decoding
codec = dict(
    type='MSRAHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2)
