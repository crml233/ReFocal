'''
warm up:2000
OrderedDict([('bbox_mAP', 0.127), ('bbox_mAP_50', 0.296), ('bbox_mAP_75', 0.092), ('bbox_mAP_vt', 0.021), ('bbox_mAP_t', 0.102), ('bbox_mAP_s', 0.194), ('bbox_mAP_m', 0.334), ('bbox_oLRP', -1.0), ('bbox_oLRP_Localisation', -1.0), ('bbox_oLRP_false_positive', -1.0), ('bbox_oLRP_false_negative', -1.0), ('bbox_mAP_copypaste', '0.127 -1.000 0.296 0.092 0.021 0.102')])

warmup:3000
OrderedDict([('bbox_mAP', 0.151), ('bbox_mAP_50', 0.358), ('bbox_mAP_75', 0.105), ('bbox_mAP_vt', 0.024), ('bbox_mAP_t', 0.126), ('bbox_mAP_s', 0.215), ('bbox_mAP_m', 0.358), ('bbox_oLRP', -1.0), ('bbox_oLRP_Localisation', -1.0), ('bbox_oLRP_false_positive', -1.0), ('bbox_oLRP_false_negative', -1.0), ('bbox_mAP_copypaste', '0.151 -1.000 0.358 0.105 0.024 0.126')])


'''



_base_ = [
    '../_base_/datasets/aitodv2_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='ATSS',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

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