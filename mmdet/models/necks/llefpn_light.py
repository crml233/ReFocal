import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder  import NECKS
from mmcv.cnn import build_conv_layer, build_norm_layer
import torch
import torch.nn.parallel
import numpy as np

class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer




def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal'] #uniform用于sigmoid激活的
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
 
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="BN"): #out_chan=256
        super(FeatureSelectionModule, self).__init__()
        # self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.norm_cfg=dict(type=norm, requires_grad=True)
        self.conv_cfg = None
        self.conv_atten = build_conv_layer(
                self.conv_cfg,
                in_chan,
                in_chan,
                kernel_size=1,
                bias=False)
 
        self.norm1 = build_norm_layer(
                self.norm_cfg,
                in_chan)[1]    
        # self.norm1 = build_norm_layer(
        #         self.norm_cfg,
        #         out_chan)[1]    
        
        
        self.sigmoid = nn.Sigmoid()

        self.conv = build_conv_layer(
                self.conv_cfg,
                in_chan,
                out_chan,
                kernel_size=1,
                bias=False)
        xavier_init(self.conv_atten,distribution='uniform')
        xavier_init(self.conv,distribution='uniform')

    def forward(self, x):
        atten = self.sigmoid(self.norm1(self.conv_atten(nn.AvgPool2d((x.shape[2], x.shape[3]))(x)))) #fm 就是激活加卷积，我觉得这平均池化用的贼巧妙
        feat = torch.mul(x, atten) #相乘，得到重要特征
        x = x + feat #再加上，加上更好
        feat = self.conv(x)
        return feat

@NECKS.register_module()
class Li_LLEFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 
                 num_outs,
                 blurpool = False,
                 if_maxpool = False,
                 start_level=1,
                 end_level=-1,
                 add_extra_convs='on_output',
                 add_lle_convs='on_output',
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(Li_LLEFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.blurpool = blurpool
        self.if_maxpool = if_maxpool

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        
        assert isinstance(add_extra_convs, (str, bool))
        assert isinstance(add_lle_convs, (str, bool))
        
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'
            
        if isinstance(add_lle_convs, str):
            # low_level_enchanced_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_lle_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_lle_convs:  # True
            self.add_lle_convs = 'on_input'


        self.add_lle_convs = add_lle_convs
        self.lateral_convs = nn.ModuleList()
        self.extract_mi_convs = nn.ModuleList()
        self.channel_attens = nn.ModuleList()

        for i in range(self.start_level-1, self.backbone_end_level): 
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv) #得到M_i
            
        for i in range(self.start_level-1, self.backbone_end_level):    #start_level =1,  backbone_end_level=4
            #for m0,m1,m2,m3
            
            # extract_mi_conv = ConvModule(
            #     out_channels,
            #     out_channels,
            #     3,
            #     padding=1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg,
            #     act_cfg=act_cfg,
            #     inplace=False)
            if self.blurpool:
                pool = BlurPool(out_channels,filt_size=2)
            
            else:
                if self.if_maxpool:
                    pool = nn.MaxPool2d(2, stride=2)
                else:
                    pool = nn.AvgPool2d(2, stride=2)
            
            # extract_and_pool = nn.Sequential(extract_mi_conv,pool)
            
            extract_and_pool = nn.Sequential(pool)
            
            self.extract_mi_convs.append(extract_and_pool)
       

        for i in range(self.start_level, self.backbone_end_level):    #start_level =1,
            #for m1,m2,m3
                        
            channel_atten = FeatureSelectionModule(in_chan=2*out_channels,out_chan=out_channels)

            self.channel_attens.append(channel_atten)

        # add extra conv layers (e.g., RetinaNet)
        self.fpn_convs = nn.ModuleList()
        
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i]) #现在每层都要经过一个1x1的卷积层，而不是i+start_level
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1): #
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # # build outputs
        # # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]
        
        ###################################################################
        # build low-level enhanced feature maps
        #从m0开始通过一个3x3的卷积层和blurpool
        if self.add_lle_convs == 'on_output':
            ME_0 = self.extract_mi_convs[0](laterals[0]) #现在大小和M_1一样
            #将ME_0和M_1拼接，然后通过一个通道注意力模块
            M1_aftercat = torch.cat((ME_0,laterals[1]),1)
            M1_afteratten = self.channel_attens[0](M1_aftercat)
            #M1_afteratten和M1相加得到P1
            P1 = M1_afteratten + laterals[1]
            
            #对P1进行3x3的卷积和blurpool
            ME_1 = self.extract_mi_convs[1](P1)
            #将ME_1和M2拼接，然后通过一个通道注意力模块
            M2_aftercat = torch.cat((ME_1,laterals[2]),1)
            M2_afteratten = self.channel_attens[1](M2_aftercat)
            #M2_afteratten和M2相加得到P2
            P2 = M2_afteratten + laterals[2]
            
            #对P2进行3x3的卷积和blurpool
            ME_2 = self.extract_mi_convs[2](P2)
            #将ME_2和M3拼接，然后通过一个通道注意力模块
            M3_aftercat = torch.cat((ME_2,laterals[3]),1)
            M3_afteratten = self.channel_attens[2](M3_aftercat)
            #M3_afteratten和M3相加得到P3
            P3 = M3_afteratten + laterals[3]
            
            outs = [P1, P2, P3]
           
            
    
            
            
        
        
        
        
        ###################################################################
        
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[0](extra_source))
                for i in range(1,self.num_outs-used_backbone_levels+1):#因为laterral比fpn多一层
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)




# if __name__ == '__main__' :
    
#     # >>> import torch
#     #     >>> in_channels = [2, 3, 5, 7]
#     #     >>> scales = [340, 170, 84, 43]
#     #     >>> inputs = [torch.rand(1, c, s, s)
#     #     ...           for c, s in zip(in_channels, scales)]
#     #     >>> self = FPN(in_channels, 11, len(in_channels)).eval()
#     #     >>> outputs = self.forward(inputs)
#     #     >>> for i in range(len(outputs)):
#     #     ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
#     #     outputs[0].shape = torch.Size([1, 11, 340, 340])
#     #     outputs[1].shape = torch.Size([1, 11, 170, 170])
#     #     outputs[2].shape = torch.Size([1, 11, 84, 84])
#     #     outputs[3].shape = torch.Size([1, 11, 43, 43])
#     in_channels = [2, 4, 8, 16]
#     scales = [200, 100, 50, 25]
#     inputs = [torch.rand(1, c, s, s)
#               for c, s in zip(in_channels, scales)]
    
#     llefpn = LLEFPN(in_channels=in_channels, out_channels=2, num_outs=5,blurpool=False).eval()
    
#     outputs = llefpn.forward(inputs)
#     for i in range(len(outputs)):
#         print(f'outputs[{i}].shape = {outputs[i].shape}')