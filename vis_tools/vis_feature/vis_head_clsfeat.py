import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import detection
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import pdb


from mmdet.apis import init_detector, inference_detector
# from mmdet.utils import register_all_modules
#     # register all modules in mmrotate into the registries
# register_all_modules()

# 根据配置文件和checkpoint文件构建模型
config_file='/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_r50_1x/aitodv2_fcos_r50_1x.py'
checkpoint_file = '/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_r50_1x/epoch_12.pth'

# config_file= '/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_revarifocal_1x/aitodv2_fcos_revarifocal_1x.py'
# checkpoint_file = '/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_revarifocal_1x/epoch_12.pth'

# config_file = '/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_revfl_qkld_0.2_lillefpn_1x/aitodv2_fcos_revfl_qkld_0.2_lillefpn_1x.py'
# checkpoint_file = '/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_revfl_qkld_0.2_lillefpn_1x/epoch_12.pth'

# config_file = '/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_p2_1x/aitodv2_fcos_p2_1x.py'
# checkpoint_file = '/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_p2_1x/epoch_12.pth'


# config_file = '/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_r50_lightllefpn_1x./aitodv2_fcos_r50_lightllefpn_1x..py'
# checkpoint_file = '/home/czj/mmdet-rfla/work_dirs/aitodv2_fcos_r50_lightllefpn_1x./epoch_12.pth'

device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# print(model)

# model = model.eval()




# 测试单张图片并生成推理结果
imgpath_list = [ '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/38__2400_600.png',
    # '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/P2689__1.0__1200___600.png' ,
    #        '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/P2346__1.0__849___0.png',
    #        '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/P2427__1.0__1200___1800.png',
    #         '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/9999994_00000_d_0000052__600_0.png',
    #         '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/P2427__1.0__1200___0.png'
            ]
            
# imgpath = '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/P2346__1.0__849___0.png'
# imgpath = '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/P2427__1.0__1200___1800.png'
# imgpath = '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/9999994_00000_d_0000052__600_0.png'
# imgpath = '/home/czj/mmdet-rfla/vis_tools/vis_feature/img_demo/P2427__1.0__1200___0.png'

dpath = '/home/czj/mmdet-rfla/vis_tools/vis_feature/head_show/fcos'

# transformss = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Resize((1024, 1024)),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# # #注意如果有中文路径需要先解码，最好不要用中文
for imgpath in imgpath_list:
    ori_img = cv2.imread(imgpath)



    result = inference_detector(model,imgpath)[0]

    # print(result)
    # pdb.set_trace()
    lenth = len(result)
    print(lenth)
    for i in range(len(result)):
        print(result[i].shape) #torch.Size([256, 100, 100])


    for i in range(lenth):
    
        conv_features = result[i].cpu()
        heat = conv_features.squeeze(0)#降维操作,尺寸变为(C,H,W)
        heatmap = torch.mean(heat,dim=0)#对各卷积层(C)求平均值,尺寸变为(H,W)
        # heatmap = torch.max(heat,dim=1).values.squeeze()
        print(heatmap.shape)
        wide = heatmap.shape[1]
        # pdb.set_trace()

        heatmap = heatmap.detach().numpy()#转换为numpy数组
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)#minmax归一化处理
        heatmap = cv2.resize(heatmap,(ori_img.shape[1],ori_img.shape[0]))#变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
        heatmap = np.uint8(255*heatmap)#像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
        heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)#颜色变换

        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        imgname = os.path.basename(imgpath)
        name =dpath + '/' + imgname.split('.p')[0]
        cv2.imwrite(name+'{}.jpg'.format(wide), heatmap)


