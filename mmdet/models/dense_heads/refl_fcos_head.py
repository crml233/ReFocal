import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean, build_assigner, bbox_overlaps
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmdet.core.bbox.iou_calculators import build_iou_calculator
import json

INF = 1e8


@HEADS.register_module()
class ReFL_FCOSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 if_3x = False,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=True,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 refl=True,
                 if_qkld = False,
                 if_qiou = False,
                 if_qsiwd = False,
                 qsiwd = None,
                 qiou = None,
                 qkld = None,
                 iou_calculator = dict(type='BboxDistanceMetric'),
                 only_am=False,
                 only_as=False,
                 all_soft = False,
                 if_centerbbox = False,
                 if_reverse = False,
                 if_gt_wise = False,
                 output_pred = None,
                 loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.refl = refl
        self.if_3x = if_3x
        self.output_pred = output_pred
        self.if_reverse = if_reverse
        self.if_gt_wise = if_gt_wise
        self.only_am = only_am
        self.only_as = only_as
        self.all_soft = all_soft
        self.if_centerbbox = if_centerbbox
        self.if_qkld = if_qkld
        self.if_qiou = if_qiou
        self.if_qsiwd = if_qsiwd
        self.qsiwd = qsiwd
        self.qiou = qiou
        self.qkld = qkld
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.iou_calculator = build_iou_calculator(iou_calculator)
        # self.assigner = build_assigner(self.train_cfg.assigner)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.count = -1
        self.thr = [0,0,0.1]

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()

    def forward(self, feats, show_clsfeat = False):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides, show_clsfeat=show_clsfeat)

    def forward_single(self, x, scale, stride, show_clsfeat):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        
        if show_clsfeat:
            return  cls_feat
        else:
            return cls_score, bbox_pred

    # Proposed: Quality Weighted Loss (QWL) which uses RFD as prior and IoU as posterior, in a focal way
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) 
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        
        labels, bbox_targets, pos_preds_gt, pos_num_gt = self.get_targets(all_level_points, cls_scores, bbox_preds, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and bbox_preds_refine
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3,
                              1).reshape(-1,
                                         self.cls_out_channels).contiguous()
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        flatten_pos_preds_gt = torch.cat(pos_preds_gt)
        flatten_pos_num_gt = torch.cat(pos_num_gt)

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.where(
            ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]
        neg_inds = (flatten_labels >= bg_class_ind).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        num_neg = torch.tensor(
            len(neg_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_neg = max(reduce_mean(num_neg), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        flatten_cls_scores_sig = flatten_cls_scores.sigmoid() # normalize into (0,1)
        pos_cls_score = flatten_cls_scores_sig[pos_inds, pos_labels]
        self.count = self.count + 1

        ###################### RE-FOCAL LOSS
        reverse_focal_rate = flatten_pos_preds_gt
        pos_num_gt_inds = flatten_pos_num_gt.nonzero() # get the index of positive samples
        flatten_pos_pos_num_gt = flatten_pos_num_gt[pos_num_gt_inds]
        anchor_value = torch.mean(flatten_pos_pos_num_gt)
        pos_alpha_t = anchor_value/torch.clamp(flatten_pos_pos_num_gt, min=1.0, max=anchor_value)
        margin_factor = torch.zeros_like(flatten_labels).float()
        margin_factor[pos_num_gt_inds] = pos_alpha_t
        #######################

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_points = flatten_points[pos_inds]

            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            iou_targets_ini = bbox_overlaps(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                is_aligned=True).clamp(min=1e-6)
            
            thr = [0,0,0.1]

            # for analysis
            
            ave_iou = torch.mean(iou_targets_ini)
            #max_iou = torch.max(iou_targets_ini)
            std_iou = torch.std(iou_targets_ini)
            thr_iou = (ave_iou - 2*std_iou).clamp(min=0.0)
            #thr_iou = 0.1

            ave_cls = torch.mean(pos_cls_score)
            std_cls = torch.std(pos_cls_score)

            thr_cls = 0.0# (ave_cls - 2*std_cls).clamp(min=0.0)
            '''
            thr[0] = thr_iou
            thr[1] = thr_cls
            thr[2] = ave_iou
            
            if self.count % 100 == 0:
                iou_statis = open("/home/xc/mmdet-swd/mmdetection/mmdet/models/dense_heads/v004.05.10_iou_statis.txt", "a")
                context_iou = "iterations: " + str(self.count) + ", ave_iou: " + str(ave_iou) + ", std_iou: " + str(std_iou) + "\n"
                iou_statis.writelines(context_iou)
                cls_statis = open("/home/xc/mmdet-swd/mmdetection/mmdet/models/dense_heads/v004.05.10_cls_statis.txt", "a")
                context_cls = "iterations: " + str(self.count) + ", ave_cls: " + str(ave_cls) + ", std_cls: " + str(std_cls) + "\n"
                cls_statis.writelines(context_cls)
            '''

            bbox_weights_ini = iou_targets_ini.clone().detach()
            bbox_weights_ini = bbox_weights_ini.clamp(min=0.02)
            iou_targets_ini_avg_per_gpu = reduce_mean(
                bbox_weights_ini.sum()).item()
            bbox_avg_factor_ini = max(iou_targets_ini_avg_per_gpu, 1.0)
            
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                weight=None,
                avg_factor=num_pos)
            
            # build IoU-aware cls_score targets  
            if self.refl:
                pos_ious = bbox_weights_ini.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                if self.if_3x:
                    countnum = 10000
                else:
                    countnum = 3000
                if self.count < countnum:
                    if self.all_soft:
                        pos_targets = pos_ious
                    else:
                        pos_targets = torch.ones_like(pos_ious)
                else:
                    pos_targets = pos_ious
                cls_iou_targets[pos_inds, pos_labels] = pos_targets #pos_ious
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            #loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.refl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)

        if self.refl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                reverse_focal_rate=reverse_focal_rate.detach(), 
                margin_factor=margin_factor.detach(),
                thr =thr_cls,
                only_am = self.only_am,
                only_as =self.only_as,
                
                avg_factor=num_pos)
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=None,
                avg_factor=num_pos)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)
       

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list, mlvl_points,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        
        for img_id, _ in enumerate(img_metas):
            if self.output_pred:
                for i in range(result_list[0][0].size(0)):                    
                    pred = {}
                    pred['image_id'] = img_metas[img_id]['ori_filename']
                    pred['bbox'] = result_list[0][0][i,:4].cpu().numpy().tolist()
                    pred['score'] = result_list[0][0][i,4].cpu().numpy().tolist()
                    pred['category_id'] = result_list[0][1][i].cpu().numpy().tolist()
                    with open(self.output_pred,'a+') as file: #追加读写
                        json.dump(pred, file) # each line denotes a predicted item
                        file.write('\n')    
        
        
        
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, points in zip(
                cls_scores, bbox_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            # Always keep topk op for dynamic input in onnx
            if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
                                       or scores.shape[-2] > nms_pre_tensor):
                from torch import _shape_as_tensor
                # keep shape as tensor and get k
                num_anchor = _shape_as_tensor(scores)[-2].to(device)
                nms_pre = torch.where(nms_pre_tensor < num_anchor,
                                      nms_pre_tensor, num_anchor)

                max_scores, _ = scores.max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Set max number of box to be feed into nms in deployment
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            batch_mlvl_scores, _ = (
                    batch_mlvl_scores.expand_as(batch_mlvl_scores)
            ).max(-1)
            _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
            batch_inds = torch.arange(batch_mlvl_scores.shape[0]).view(
                -1, 1).expand_as(topk_inds)
            batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
            batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]

        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results
    
    
    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification  targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        num_imgs = len(gt_bboxes_list)

        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        lvl_scores = []
        lvl_bboxes = []
        concat_cls_scores_list = []
        concat_bbox_preds_list = []

        for i in range(len(cls_scores_list)):
            reshaped_scores = cls_scores_list[i].detach().reshape(num_imgs,self.num_classes,-1)
            reshaped_bboxes = bbox_preds_list[i].detach().reshape(num_imgs,4,-1)
            lvl_scores.append(reshaped_scores)
            lvl_bboxes.append(reshaped_bboxes)
        cat_lvl_scores = torch.cat(lvl_scores, dim=-1)
        cat_lvl_bboxes = torch.cat(lvl_bboxes, dim=-1)

        for j in range(num_imgs):
            concat_cls_scores_list.append(cat_lvl_scores[j,...])
            concat_bbox_preds_list.append(cat_lvl_bboxes[j,...])

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, pos_preds_gt_list, pos_num_gt_list  = multi_apply(
            self._get_target_single,
            concat_cls_scores_list,
            concat_bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        pos_preds_gt_list = [pos_preds_gt.split(num_points, 0) for pos_preds_gt in pos_preds_gt_list]
        pos_num_gt_list = [pos_num_gt.split(num_points, 0) for pos_num_gt in pos_num_gt_list] 

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_preds_gt = []
        concat_lvl_num_gt = []

        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

            pos_preds_gt = torch.cat([pos_preds_gt[i] for pos_preds_gt in pos_preds_gt_list])
            concat_lvl_preds_gt.append(pos_preds_gt)
            pos_num_gt = torch.cat([pos_num_gt[i] for pos_num_gt in pos_num_gt_list])
            concat_lvl_num_gt.append(pos_num_gt)            

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_preds_gt,  concat_lvl_num_gt

    def _get_target_single(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        ########################
        ## get the targets of the relative focal loss
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero().reshape(-1)
        num_gts = gt_labels.size(0)
        assigned_gt_inds = min_area_inds[pos_inds] # assigned gt inds
        pos_labels = labels[pos_inds]
        cls_sigmoid = cls_scores.permute(1, 0).sigmoid()
        reshaped_bbox_preds = bbox_preds.permute(1, 0).reshape(-1, 4)
        pos_bbox_preds = reshaped_bbox_preds[pos_inds]
        pos_cls_score = cls_sigmoid[pos_inds, pos_labels]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_points = points[pos_inds,:]
        pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
        pos_decoded_target_preds = distance2bbox(pos_points,pos_bbox_targets)
        
        
        
       
        
        cls_iou = pos_cls_score 
        
       
       
        if self.if_reverse:
            reverse_cls_iou = 1/(cls_iou.clamp(min=1e-6))
        else:
            reverse_cls_iou = cls_iou #+1e-6
            
            
        if self.if_qkld:
            kld_pred = self.iou_calculator(
                pos_decoded_bbox_preds.detach(),
                pos_decoded_target_preds.detach(),
                mode = 'kl',
                is_aligned=True).detach()
            kld_pred = torch.diag(kld_pred)
            reverse_cls_iou = (reverse_cls_iou**self.qkld)*(kld_pred**(1-self.qkld)) #qkld = 0.2
        
        if self.if_qiou:
            iou_pred =bbox_overlaps(pos_decoded_bbox_preds.detach(), pos_decoded_target_preds.detach(), is_aligned=True).detach()
            reverse_cls_iou = (reverse_cls_iou**self.qiou)*(iou_pred**(1-self.qiou))
        
        if self.if_qsiwd:
            siwd_pred = self.iou_calculator(
                pos_decoded_bbox_preds.detach(),
                pos_decoded_target_preds.detach(),
                mode = 'siwd',
                is_aligned=True).detach()
            siwd_pred = torch.diag(siwd_pred)
            reverse_cls_iou = (reverse_cls_iou**self.qsiwd)*(siwd_pred**(1-self.qsiwd))
            
            
            
            
        pos_preds_gt = torch.zeros_like(labels).float()
        pos_num_gt = torch.zeros_like(labels).float()
        
        
        # r01.01.04
        if self.if_gt_wise:
            for i in range(num_gts):
                if assigned_gt_inds.size(0)>0:
                    gt_i_pos_inds = (i==assigned_gt_inds) 
                    gt_i_pos_num = torch.sum(gt_i_pos_inds).float() #number of positive samples for the corresponding gt
                    gt_i_cls_iou = reverse_cls_iou[gt_i_pos_inds].float() #
                    if gt_i_cls_iou.size(0)>1:
                        gt_i_min = torch.min(gt_i_cls_iou)
                        gt_i_max = torch.max(gt_i_cls_iou)
                        recalibrated_gt_i_cls_iou = (gt_i_cls_iou-0.0)/(gt_i_max-0.0)
                        recalibrated_gt_i_cls_iou = torch.clamp(recalibrated_gt_i_cls_iou, min=1e-6, max=1.0)
                        gt_i_sum = torch.sum(recalibrated_gt_i_cls_iou)
                        # print(recalibrated_gt_i_cls_iou.shape) #4,9,2,2,3
                        # print(recalibrated_gt_i_cls_iou)#[0.9890, 1.0000, 0.9896, 0.9951],[0.9830, 0.9887, 0.9811, 0.9923, 0.9942, 0.9901, 1.0000, 0.9897, 0.9846]
                        # # print(gt_i_sum)#

                        pos_preds_gt[pos_inds[gt_i_pos_inds]] = gt_i_sum 
                        pos_num_gt[pos_inds[gt_i_pos_inds]] = gt_i_pos_num 
                        # print(pos_preds_gt[pos_inds[gt_i_pos_inds]])#([3.9736, 3.9736, 3.9736, 3.9736],
                        # print(pos_num_gt[pos_inds[gt_i_pos_inds]])#[4., 4., 4., 4.],[9., 9., 9., 9., 9., 9., 9., 9., 9.]
                    else:
                        pos_preds_gt[pos_inds[gt_i_pos_inds]] = torch.ones_like(gt_i_cls_iou).float() 
                        pos_num_gt[pos_inds[gt_i_pos_inds]] = torch.ones_like(gt_i_cls_iou).float() 
                        
      
            pos_preds_gt = pos_preds_gt[pos_inds]/torch.sum(pos_preds_gt[pos_inds])
        
        else:#
            for i in range(num_gts):
                if assigned_gt_inds.size(0)>0:
                    gt_i_pos_inds = (i==assigned_gt_inds)
                    gt_i_pos_num = torch.sum(gt_i_pos_inds).float() # number of positive samples for each gt
                    gt_i_cls_iou = reverse_cls_iou[gt_i_pos_inds].float()#list of score for each positive sample
                    if gt_i_cls_iou.size(0)>1: # if there are more than one positive samples for each gt
                        gt_i_min = torch.min(gt_i_cls_iou)
                        gt_i_max = torch.max(gt_i_cls_iou)
                        recalibrated_gt_i_cls_iou = (gt_i_cls_iou-0.0)/(gt_i_max-0.0) #normalize the score into (0,1)
                        recalibrated_gt_i_cls_iou = torch.clamp(recalibrated_gt_i_cls_iou, min=1e-6, max=1.0)

                        pos_preds_gt[pos_inds[gt_i_pos_inds]] = recalibrated_gt_i_cls_iou 
                        pos_num_gt[pos_inds[gt_i_pos_inds]] = gt_i_pos_num 
                    else:
                        pos_preds_gt[pos_inds[gt_i_pos_inds]] = torch.ones_like(gt_i_cls_iou).float() 
                        pos_num_gt[pos_inds[gt_i_pos_inds]] = torch.ones_like(gt_i_cls_iou).float() 


        return labels, bbox_targets, pos_preds_gt, pos_num_gt 

