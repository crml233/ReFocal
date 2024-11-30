import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_relative_focal_loss(pred,
                          target,
                          reverse_focal_rate,
                          margin_factor,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    
    ########## relative focal loss
    # alpha_t * alpha_s 
    pos_inds = reverse_focal_rate.nonzero()
    pos_loss = loss[pos_inds]
    pos_reverse_focal_rate = reverse_focal_rate[pos_inds]
    pos_alpha_s = pos_reverse_focal_rate * torch.sum(pos_loss)/torch.sum(pos_reverse_focal_rate * pos_loss)
    pos_alpha_t = margin_factor[pos_inds]
    pos_weight = pos_alpha_t * pos_alpha_s
    weight = torch.ones_like(loss)
    weight[pos_inds] = pos_weight

    # filter out low quality samples
    '''
    noisy_thr = 0.1
    pos_pred, _ = torch.max(pred_sigmoid[pos_inds], dim=1)
    num_pos = pos_loss.size(0)
    num_noisy = int(num_pos * noisy_thr)
    pos_pred_sorted, pos_pred_inds_sorted = torch.sort(pos_pred)
    noisy_pred_inds = pos_pred_inds_sorted[:num_noisy] # last 10%
    weight[pos_inds[noisy_pred_inds]] = 0.0
    '''
    
    ##########

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_relative_focal_loss(pred,
                       target,
                       reverse_focal_rate,
                       margin_factor,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target, gamma, alpha, None,
                               'none')

    ########## relative focal loss
    # alpha_t * alpha_s 
    pred_sigmoid = pred.sigmoid()
    num_classes = pred.size(1)
    #pos_inds = reverse_focal_rate.nonzero().squeeze(-1)
    pos_inds = ((target>=0) & (target<num_classes)).nonzero().squeeze(-1)

    pos_target = target[pos_inds]
    pos_loss = loss[pos_inds,pos_target]
    pos_reverse_focal_rate = reverse_focal_rate[pos_inds]
    pos_alpha_s = pos_reverse_focal_rate * torch.sum(pos_loss)/torch.sum(pos_reverse_focal_rate * pos_loss)
    pos_alpha_t = margin_factor[pos_inds]
    pos_weight = pos_alpha_t * pos_alpha_s
    weight = torch.ones_like(loss)
    weight[pos_inds,pos_target] = pos_weight

    # filter out low quality samples
    # noisy_thr = 0.1
    # pos_pred = pred_sigmoid[pos_inds, pos_target]
    # num_pos = pos_loss.size(0)
    # num_noisy = int(num_pos * noisy_thr)
    # pos_pred_sorted, pos_pred_inds_sorted = torch.sort(pos_pred)
    # noisy_pred_inds = pos_pred_inds_sorted[:num_noisy] # last 10%
    # weight[pos_inds[noisy_pred_inds],pos_target[noisy_pred_inds]] = pos_pred_sorted[:num_noisy]
    
    ##########

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def rehardfocal_loss(pred,
                   target,
                   hard_target,
                   thr,
                   only_am,
                   only_as,
                   reverse_focal_rate,
                   margin_factor,
                   weight=None,
                   gamma=2.0,
                   alpha=0.25,
                   reduction='mean',
                   avg_factor=None):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    target_p = target #** beta  the weight of pos sample
    alpha_n = alpha #thr[2]**beta * alpha # the weight of neg sample
    pos_flag = target > thr #(target > thr[0]) & (pred_sigmoid > thr[1])
    #print(torch.sum(pos_flag))
    neg_flag = ~pos_flag
    # import pdb
    # print(target)
    # print(target.shape)
    # print(hard_target)
    # print(hard_target.shape)
    # print(pos_flag)
    # print(pos_flag.shape)
    # print(target*(pos_flag))
    # print((target*(pos_flag)).shape)
    # pdb.set_trace()
    #target 形状为[53372, 18]
    #hard_target 形状为[53372]
    #将hard_target转换为one-hot编码
    
    hard_target = F.one_hot(hard_target, num_classes=pred.shape[1]+1)
    
    # print(hard_target)
    # print(hard_target.shape)
    #将hard_target转换为float类型
    hard_target = hard_target.type_as(pred)
    #取hard_target的前pred.shape[1]列
    hard_target = hard_target[:, :pred.shape[1]]

    focal_weight = target_p * (pos_flag).float() + \
                    alpha_n * (pred_sigmoid).abs().pow(gamma) * \
                    (neg_flag).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, hard_target, reduction='none') * focal_weight
    
    ########## relative focal loss
    # alpha_t * alpha_s 
    pos_inds = pos_flag #((target>=0) & (target<num_classes)).nonzero().squeeze(-1)
    import pdb
    
    # print('pos_inds', pos_inds, pos_inds.shape)#[[False, False, False,  ..., False, False, False],
    #     [False, F
         #[53372, 8]
    
    
    
    pos_inds_1d, _ = torch.max(pos_inds, dim=1) 
    # print('pos_inds_1d', pos_inds_1d, pos_inds_1d.shape) #[False, False, False,  ..., False, False, False]
    # #[53372])
    # pdb.set_trace()

    #pos_target = target[pos_inds]
    pos_loss = loss[pos_inds]
    pos_reverse_focal_rate = reverse_focal_rate[pos_inds_1d]
    pos_alpha_s = pos_reverse_focal_rate * torch.sum(pos_loss)/torch.sum(pos_reverse_focal_rate * pos_loss)
    pos_alpha_t = margin_factor[pos_inds_1d]
    # print('pos_alpha_s', pos_alpha_s, pos_alpha_s.shape)#[125]
    # print('pos_alpha_t', pos_alpha_t, pos_alpha_t.shape)##[125]
    # pdb.set_trace()
    if only_am:
        pos_alpha_s = torch.ones_like(pos_alpha_s)
    if only_as:
        pos_alpha_t = torch.ones_like(pos_alpha_t)
    pos_weight = pos_alpha_t * pos_alpha_s
    weight = torch.ones_like(loss)
    weight[pos_inds] = pos_weight
    ##########

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



#@mmcv.jit(derivate=True, coderize=True)
def refocal_loss(pred,
                   target,
                   thr,
                   only_am,
                   only_as,
                   reverse_focal_rate,
                   margin_factor,
                   weight=None,
                   gamma=2.0,
                   alpha=0.25,
                   reduction='mean',
                   avg_factor=None,
                   time_consider =False,
                   epoch = 0 ):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    target_p = target #** beta  the weight of pos sample
    alpha_n = alpha #thr[2]**beta * alpha # the weight of neg sample
    pos_flag = target > thr #(target > thr[0]) & (pred_sigmoid > thr[1])
    #print(torch.sum(pos_flag))
    neg_flag = ~pos_flag

    focal_weight = target_p * (pos_flag).float() + \
                    alpha_n * (pred_sigmoid).abs().pow(gamma) * \
                    (neg_flag).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target*(pos_flag), reduction='none') * focal_weight
    
    ########## relative focal loss
    # alpha_t * alpha_s 
    pos_inds = pos_flag #((target>=0) & (target<num_classes)).nonzero().squeeze(-1)
    import pdb
    
    # print('pos_inds', pos_inds, pos_inds.shape)#[[False, False, False,  ..., False, False, False],
    #     [False, F
         #[53372, 8]
    
    
    
    pos_inds_1d, _ = torch.max(pos_inds, dim=1) 
    # print('pos_inds_1d', pos_inds_1d, pos_inds_1d.shape) #[False, False, False,  ..., False, False, False]
    # #[53372])
    # pdb.set_trace()

    #pos_target = target[pos_inds]
    pos_loss = loss[pos_inds]
    pos_reverse_focal_rate = reverse_focal_rate[pos_inds_1d]
    pos_alpha_s = pos_reverse_focal_rate * torch.sum(pos_loss)/torch.sum(pos_reverse_focal_rate * pos_loss)
    pos_alpha_t = margin_factor[pos_inds_1d]
    # print('pos_alpha_s', pos_alpha_s, pos_alpha_s.shape)#[125]
    # print('pos_alpha_t', pos_alpha_t, pos_alpha_t.shape)##[125]
    # pdb.set_trace()
    if time_consider:
        #分别计算pos_alpha_s和pos_alpha_t与1的差值
        pos_alpha_s_diff = pos_alpha_s - 1
        pos_alpha_t_diff = pos_alpha_t - 1
        #对差值用epoch的函数进行加权
        epoch_aware = (1- epoch/11)**2
        pos_alpha_s_diff = pos_alpha_s_diff * epoch_aware
        pos_alpha_t_diff = pos_alpha_t_diff * epoch_aware
        #将加权后的差值加回到1上
        pos_alpha_s = pos_alpha_s_diff + 1
        pos_alpha_t = pos_alpha_t_diff + 1
    
    
    if only_am:
        pos_alpha_s = torch.ones_like(pos_alpha_s)
    if only_as:
        pos_alpha_t = torch.ones_like(pos_alpha_t)
        
        
        
        
    pos_weight = pos_alpha_t * pos_alpha_s
    weight = torch.ones_like(loss)
    weight[pos_inds] = pos_weight
    ##########

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def refocal_deleteq_loss(pred,
                   target,
                   reverse_focal_rate,
                   margin_factor,
                   weight=None,
                   gamma=2.0,
                   alpha=0.25,
                   reduction='mean',
                   avg_factor=None ):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    target_p = target #** beta  the weight of pos sample
    alpha_n = alpha #thr[2]**beta * alpha # the weight of neg sample
    pos_flag = target > 0 #(target > thr[0]) & (pred_sigmoid > thr[1])
    #print(torch.sum(pos_flag))
    neg_flag = ~pos_flag

    focal_weight =  (pos_flag).float() + \
                    alpha_n * (pred_sigmoid).abs().pow(gamma) * \
                    (neg_flag).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target*(pos_flag), reduction='none') * focal_weight
    
    ########## relative focal loss
    # alpha_t * alpha_s 
    pos_inds = pos_flag #((target>=0) & (target<num_classes)).nonzero().squeeze(-1)
    import pdb
    
    # print('pos_inds', pos_inds, pos_inds.shape)#[[False, False, False,  ..., False, False, False],
    #     [False, F
         #[53372, 8]
    
    
    
    pos_inds_1d, _ = torch.max(pos_inds, dim=1) 
    # print('pos_inds_1d', pos_inds_1d, pos_inds_1d.shape) #[False, False, False,  ..., False, False, False]
    # #[53372])
    # pdb.set_trace()

    #pos_target = target[pos_inds]
    pos_loss = loss[pos_inds]
    pos_reverse_focal_rate = reverse_focal_rate[pos_inds_1d]
    pos_alpha_s = pos_reverse_focal_rate * torch.sum(pos_loss)/torch.sum(pos_reverse_focal_rate * pos_loss)
    pos_alpha_t = margin_factor[pos_inds_1d]
    # print('pos_alpha_s', pos_alpha_s, pos_alpha_s.shape)#[125]
    # print('pos_alpha_t', pos_alpha_t, pos_alpha_t.shape)##[125]
    # pdb.set_trace()
   
        
        
        
    pos_weight = pos_alpha_t * pos_alpha_s
    weight = torch.ones_like(loss)
    weight[pos_inds] = pos_weight
    ##########

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class RFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(RFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                reverse_focal_rate,
                margin_factor,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_relative_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_relative_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                reverse_focal_rate,
                margin_factor,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls


@LOSSES.register_module()
class ReFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(ReFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                thr,
                only_am,
                only_as,
                reverse_focal_rate,
                margin_factor,
                weight=None,
                avg_factor=None,
                time_consider = False,
                epoch = 0,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = refocal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                thr,
                only_am,
                only_as,
                reverse_focal_rate,
                margin_factor,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
                time_consider = time_consider,
                epoch = epoch)

        else:
            raise NotImplementedError
        return loss_cls



@LOSSES.register_module()
class ReDeqFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(ReDeqFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                reverse_focal_rate,
                margin_factor,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = refocal_deleteq_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                reverse_focal_rate,
                margin_factor,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls








@LOSSES.register_module()
class RehardFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(RehardFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                soft_target,
                hard_target,
                thr,
                only_am,
                only_as,
                reverse_focal_rate,
                margin_factor,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = rehardfocal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                soft_target,
                hard_target,
                thr,
                only_am,
                only_as,
                reverse_focal_rate,
                margin_factor,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls