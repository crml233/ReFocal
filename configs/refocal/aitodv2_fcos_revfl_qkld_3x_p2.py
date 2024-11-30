'''

p2:
2024-05-05 18:25:43,995 - mmdet - INFO - Epoch(val) [36][14018] bbox_mAP: 0.2640, bbox_mAP_50: 0.5910, bbox_mAP_75: 0.1930, bbox_mAP_vt: 0.1160, bbox_mAP_t: 0.2670, bbox_mAP_s: 0.3020, bbox_mAP_m: 0.3730, bbox_oLRP: -1.0000, bbox_oLRP_Localisation: -1.0000, bbox_oLRP_false_positive: -1.0000, bbox_oLRP_false_negative: -1.0000, bbox_mAP_copypaste: 0.264 -1.000 0.591 0.193 0.116 0.267

'''

_base_ = [
    '../_base_/datasets/aitodv2_detection.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='Li_LLEFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0, #1, for test fpn feat 
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='ReFL_FCOSHead',
        norm_cfg=None,
        # output_pred = '/home/czj/mmdet-rfla/vis_tools/vis_feature/det_result_json/aitodv2_revari_det.json',
        if_qkld = True,
        qkld = 0.2,
        if_3x = True,
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        norm_on_bbox=True,
        center_sampling=True,
        refl = True,
        loss_cls=dict(
            type='ReFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='DIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000))
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.01/2, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=10000,
    warmup_ratio=1.0 / 3,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

checkpoint_config = dict(interval=36)