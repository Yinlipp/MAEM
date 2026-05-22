type = 'MultiViewMultiPersonTopDownEstimator'
bbox_thr = 0.6
work_dir = './temp'
verbose = False
logger = None
pred_kps3d_convention = 'coco'

bbox_detector = dict(
    type='MMdetDetector',
    mmdet_kwargs=dict(
        checkpoint='weight/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
        config='/home/y_li/workspace7/Anaconda3/envs/xcap/lib/python3.8/site-packages/mmdet/.mim/configs/yolox/yolox_x_8x8_300e_coco.py',
        device='cuda'),
    batch_size=10)

kps2d_estimator = dict(
    type='MMposeTopDownEstimator',
    mmpose_kwargs=dict(
        checkpoint='weight/hrnet_w48_coco_wholebody' +
        '_384x288_dark-f5726563_20200918.pth',
        config='configs/modules/human_perception/mmpose_hrnet_w48_' +
        'coco_wholebody_384x288_dark_plus.py',
        device='cuda'),
    bbox_thr=bbox_thr)

associator = dict(
    type='MvposeAssociator',
    triangulator=dict(
        type='AniposelibTriangulator',
        camera_parameters=[],
        logger=logger,
    ),
    affinity_estimator=dict(
        type='AppearanceAffinityEstimator',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./weight/mvpose/resnet50_reid_camstyle-98d61e41_20220921.pth'
        )
    ),
    point_selector=dict(
        type='HybridKps2dSelector',
        triangulator=dict(
            type='AniposelibTriangulator', camera_parameters=[],
            logger=logger),
        verbose=verbose,
        ignore_kps_name=['left_eye', 'right_eye', 'left_ear', 'right_ear'],
        convention=pred_kps3d_convention),
    multi_way_matching=dict(
        type='MultiWayMatching',
        use_dual_stochastic_SVT=True,
        lambda_SVT=50,
        alpha_SVT=0.5,
        n_cam_min=3,
    ),
    kalman_tracking=dict(type='KalmanTracking', n_cam_min=3, logger=logger),
    identity_tracking=dict(
        type='KeypointsDistanceTracking',
        # type='Perception2dTracking',
        tracking_distance=1.5,
        tracking_kps3d_convention=pred_kps3d_convention,
        tracking_kps3d_name=[
            'left_shoulder', 'right_shoulder', 'left_hip_extra',
            'right_hip_extra'
        ]),
    checkpoint_path='./weight/mvpose/' +
    'resnet50_reid_camstyle-98d61e41_20220921.pth',
    best_distance=90,
    interval=5,
    bbox_thr=bbox_thr,
    device='cuda',
    logger=logger,
)

smplify = None

triangulator = dict(
    type='AniposelibTriangulator',
    camera_parameters=[],
    logger=logger,
)
point_selectors = [
    dict(
        type='ReprojectionErrorPointSelector',
        target_camera_number=2,  # Lowered from 3 to 2 to handle cases with fewer valid views
        triangulator=dict(
            type='AniposelibTriangulator', camera_parameters=[],
            logger=logger),
        verbose=verbose,
        logger=logger,
    )
]

kps3d_optimizers = [
    dict(type='TrajectoryOptimizer', verbose=verbose, logger=logger),
    dict(type='NanInterpolation', verbose=verbose, logger=logger),
    # SMPLShapeAwareOptimizer is optional.
    # dict(
    #     type='SMPLShapeAwareOptimizer',
    #     smplify=smplify,
    #     body_model=smplify['body_model'],
    #     projector=dict(type='PytorchProjector', camera_parameters=[]),
    #     iteration=1,
    #     refine_threshold=1,
    #     kps2d_conf_threshold=0.97,
    #     use_percep2d_optimizer=False,
    #     verbose=verbose,
    #     logger=logger),
    # After SMPL shape-aware optimizer, the keypoints are not very stable,
    # so trajectory optimization is added.
    dict(type='TrajectoryOptimizer', verbose=verbose, logger=logger),
    dict(type='NanInterpolation', verbose=verbose, logger=logger),
]
