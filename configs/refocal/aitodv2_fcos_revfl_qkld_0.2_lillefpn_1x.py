'''
0.1
2024-04-09 09:33:49,030 - mmdet - INFO - Epoch(val) [12][14018] bbox_mAP: 0.1750, bbox_mAP_50: 0.4300, bbox_mAP_75: 0.1120, bbox_mAP_vt: 0.0400, bbox_mAP_t: 0.1600, bbox_mAP_s: 0.2360, bbox_mAP_m: 0.3240, bbox_oLRP: -1.0000, bbox_oLRP_Localisation: -1.0000, bbox_oLRP_false_positive: -1.0000, bbox_oLRP_false_negative: -1.0000, bbox_mAP_copypaste: 0.175 -1.000 0.430 0.112 0.040 0.160'
14018] bbox_mAP: 0.1720, bbox_mAP_50: 0.4320, bbox_mAP_75: 0.1060, bbox_mAP_vt: 0.0380, bbox_mAP_t: 0.1560, bbox_mAP_s: 0.2310, bbox_mAP_m: 0.3200, bbox_oLRP: -1.0000, bbox_oLRP_Localisation: -1.0000, bbox_oLRP_false_positive: -1.0000, bbox_oLRP_false_negative: -1.0000, bbox_mAP_copypaste: 0.172 -1.000 0.432 0.106 0.038 0.156
0.1+lifpn
 bbox_mAP: 0.1770, bbox_mAP_50: 0.4460, bbox_mAP_75: 0.1050, bbox_mAP_vt: 0.0540, bbox_mAP_t: 0.1670, bbox_mAP_s: 0.2340, bbox_mAP_m: 0.3240, bbox_oLRP: -1.0000, bbox_oLRP_Localisation: -1.0000, bbox_oLRP_false_positive: -1.0000, bbox_oLRP_false_negative: -1.0000, bbox_mAP_copypaste: 0.177 -1.000 0.446 0.105 0.054 0.167

0.2+lifpn

Epoch(val) [12][14018] bbox_mAP: 0.1820, bbox_mAP_50: 0.4480, bbox_mAP_75: 0.1200, bbox_mAP_vt: 0.0510, bbox_mAP_t: 0.1650, bbox_mAP_s: 0.2470, bbox_mAP_m: 0.3310, bbox_oLRP: -1.0000, bbox_oLRP_Localisation: -1.0000, bbox_oLRP_false_positive: -1.0000, bbox_oLRP_false_negative: -1.0000, bbox_mAP_copypaste: 0.182 -1.000 0.448 0.120 0.051 0.165

'''

_base_ = [
    '../_base_/datasets/aitodv2_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
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
        start_level=1, #1, for test fpn feat 
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='ReFL_FCOSHead',
        norm_cfg=None,
        # output_pred = '/home/czj/mmdet-rfla/vis_tools/vis_feature/det_result_json/aitodv2_qkldllefpn_det.json',
        if_qkld = True,
        qkld = 0.2,
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
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
        max_per_img=3000)
    )
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
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)