import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import detection
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
import pdb 

# # #手动导入预训练模型
# # model = detection.FCOS()#pretrained=False)
# checkpoint = torch.load('/home/czj/mmrotate-dev-1.x/work_dirs/fcos_fspn/epoch_12.pth')
# print(len(checkpoint))
# # for k in checkpoint.keys():
#     # print(k)
# # print(checkpoint['meta'])
# model.load_state_dict(checkpoint)
# print(model)

from mmdet.apis import init_detector, inference_detector
from mmrotate.utils import register_all_modules

from pycocotools.coco import COCO
from mmrotate.structures.bbox import RotatedBoxes
import aitool



    # register all modules in mmrotate into the registries


'''
/home/czj/data/ShipWhu/val/images/925_476_0.png #复杂背景
/home/czj/data/ShipWhu/val/images/935_0_476.png  #dense small
/home/czj/data/ShipWhu/val/images/100001518_0_0.png #multi-scale



[<DetDataSample(

    META INFORMATION
    scale_factor: (1.0, 1.0)
    pad_shape: (1024, 1024)
    ori_shape: (1024, 1024)
    batch_input_shape: (1024, 1024)
    img_path: '/home/czj/data/ShipWhu/val/images/100001449.png'
    instances: [{'ignore_flag': 0, 'bbox': [412.4350891113281, 782.30224609375, 230.5320281982422, 24.34803009033203, 1.2531609535217285], 'bbox_label': 12}, {'ignore_flag': 0, 'bbox': [488.0, 716.5, 229.1201171875, 27.513277053833008, 1.2324727773666382], 'bbox_label': 12}, {'ignore_flag': 0, 'bbox': [764.9500732421875, 694.8500366210938, 231.16253662109375, 22.135942459106445, 1.249045968055725], 'bbox_label': 12}]
    img_id: 1
    img_shape: (1024, 1024)

    DATA FIELDS
    ignored_instances: <InstanceData(
        
            META INFORMATION
        
            DATA FIELDS
            bboxes: RotatedBoxes(
                tensor([], device='cuda:0', size=(0, 5)))
            labels: tensor([], device='cuda:0', dtype=torch.int64)
        ) at 0x7fee29a31550>
    gt_instances: <InstanceData(
        
            META INFORMATION
        
            DATA FIELDS
            bboxes: RotatedBoxes(
                tensor([[412.4351, 782.3022, 230.5320,  24.3480,   1.2532],
                        [488.0000, 716.5000, 229.1201,  27.5133,   1.2325],
                        [764.9501, 694.8500, 231.1625,  22.1359,   1.2490]], device='cuda:0'))
            labels: tensor([12, 12, 12], device='cuda:0')
        ) at 0x7fee29a31520>
) at 0x7fee29a31610>]


return:
[<DetDataSample(

    META INFORMATION
    scale_factor: (1.0, 1.0)
    pad_shape: (1024, 1024)
    ori_shape: (1024, 1024)
    batch_input_shape: (1024, 1024)
    img_path: '/home/czj/data/ShipWhu/val/images/100001449.png'
    instances: [{'ignore_flag': 0, 'bbox': [412.4350891113281, 782.30224609375, 230.5320281982422, 24.34803009033203, 1.2531609535217285], 'bbox_label': 12}, {'ignore_flag': 0, 'bbox': [488.0, 716.5, 229.1201171875, 27.513277053833008, 1.2324727773666382], 'bbox_label': 12}, {'ignore_flag': 0, 'bbox': [764.9500732421875, 694.8500366210938, 231.16253662109375, 22.135942459106445, 1.249045968055725], 'bbox_label': 12}]
    img_id: 1
    img_shape: (1024, 1024)

    DATA FIELDS
    gt_instances: <InstanceData(
        
            META INFORMATION
        
            DATA FIELDS
            bboxes: tensor([[412.4351, 782.3022, 230.5320,  24.3480,   1.2532],
                        [488.0000, 716.5000, 229.1201,  27.5133,   1.2325],
                        [764.9501, 694.8500, 231.1625,  22.1359,   1.2490]], device='cuda:0')
            labels: tensor([12, 12, 12], device='cuda:0')
        ) at 0x7fee29a31520>
    ignored_instances: <InstanceData(
        
            META INFORMATION
        
            DATA FIELDS
            bboxes: tensor([], device='cuda:0', size=(0, 5))
            labels: tensor([], device='cuda:0', dtype=torch.int64)
        ) at 0x7fee29a31550>
    pred_instances: <InstanceData(
        
            META INFORMATION
        
            DATA FIELDS
            bboxes: tensor([[490.7380, 720.0721, 228.5283,  26.1714,   1.2451],
                        [765.1797, 695.0673, 226.9419,  26.2400,   1.2653],
                        [412.4075, 777.8934, 232.1275,  26.5019,   1.2533]], device='cuda:0')
            scores: tensor([0.9997, 0.9993, 0.9980], device='cuda:0')
            labels: tensor([12, 12, 12], device='cuda:0')
        ) at 0x7fee29b1a070>
) at 0x7fee29a31610>]


'''

def imgpath2annotation(jsonpath, imgpath): 
    
    data_sample = DetDataSample()
    
    data_sample.set_metainfo(dict(
        scale_factor = (1.0, 1.0),
        pad_shape = (1024, 1024),
        ori_shape = (1024, 1024),
        batch_input_shape = (1024, 1024),
        img_path = imgpath,
        img_id = 1,
        img_shape = (1024, 1024)
    ))
    
    #读取json文件
    coco=COCO(jsonpath)
    
    #将路径imgpath分割成path前缀和文件名imgname
    path, imgname = os.path.split(imgpath)
    print(imgname)
    annIds = []
       
    #获取所有图片的id
    img_ids = coco.getImgIds()
    for img_id in img_ids:
    # 加载图片的信息
        
        img_info = coco.loadImgs(img_id)[0] #返回的是一个列表，里面是字典(一个字典)
        # 检查图片的文件名是否匹配
        if img_info['file_name'] == imgname:
            annIds = coco.getAnnIds(imgIds=img_id)
            break
    
    anns = coco.loadAnns(annIds)

    # 从anns中获取pointobb和category_id
    bboexes = []
    labels = []
    for ann in anns:
        # print(ann)
        # pdb.set_trace()
        pointobb = ann['pointobb']
        #pointobb to thetaobb
        pointobb = np.intp(np.array(pointobb)) #pointobb (list): [x1, y1, x2, y2, x3, y3, x4, y4]
        pointobb.resize(4, 2)
        rect = cv2.minAreaRect(pointobb)
        x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
        theta = theta / 180.0 * np.pi
        thetaobb = [x, y, w, h, theta]
        #将thetobb转为tensor
        thetaobb = torch.tensor(thetaobb)
        category_id = ann['category_id']
        bboexes.append(thetaobb)
        labels.append(category_id-1)
    
    #将列表里的每个tensor拼接成二维tensor
    # bboexes = torch.stack(bboexes)
    # print(bboexes)
    
    gt_instances = InstanceData()
    gt_instances.bboxes = RotatedBoxes(torch.stack(bboexes))
    gt_instances.labels = torch.tensor(labels)
    
    data_sample.gt_instances = gt_instances
    
    
    
    
    return data_sample
    
def obb2poly_le135(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                        N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()

def show_gt_pred(img, pred_list, dpath ,img_name):

    gt_instance = pred_list[0].gt_instances
    pred_instance = pred_list[0].pred_instances
    gt = gt_instance.bboxes #返回的最终结果就是tensor，不是rotatedbbox
    # print(gt)
    # pdb.set_trace()
    pred = pred_instance.bboxes
    gt_bboxes = []
    pred_bboxes = []
    for i in range(gt.shape[0]):
        # print(gt[i,:])
        gt_poly = obb2poly_le135(gt[i,:].unsqueeze(dim=0)).squeeze().cpu().detach().numpy() 
        gt_bboxes.append(gt_poly)
        
    for i in range(pred.shape[0]):
        pred_poly = obb2poly_le135(pred[i,:].unsqueeze(dim=0)).squeeze().cpu().detach().numpy() 
        pred_bboxes.append(pred_poly)
   
    img = aitool.draw_confusion_matrix_rotate(img, gt_bboxes, pred_bboxes, with_gt_TP=False, line_width=2)

    output_file = os.path.join(dpath, img_name+'.png')
    cv2.imwrite(output_file,img)
    
 
  
def show_rpn(img, data_sample, rpn_list, dpath , img_name):
    gts = data_sample.gt_instances.bboxes
    rpns = rpn_list[0].bboxes.tensor
    rpn = []
    gt = []
    for i in range(rpns.shape[0]):
        rpn_poly = obb2poly_le135(rpns[i,:].unsqueeze(dim=0)).squeeze().cpu().detach().numpy() 
        rpn.append(rpn_poly)
        
    for i in range(gts.shape[0]):
        gt_poly = obb2poly_le135(gts[i,:].unsqueeze(dim=0)).squeeze().cpu().detach().numpy() 
        gt.append(gt_poly)
    img = aitool.draw_confusion_matrix_rotate(img, gt, rpn, with_gt_TP=False, line_width=2)
    output_file = os.path.join(dpath, img_name+'.png')
    cv2.imwrite(output_file,img)
    

def just_show_gt(img, data_sample, dpath , img_name):
    gts = data_sample.gt_instances.bboxes

    for i in range(gts.shape[0]):
        gt = gts[i,:].cpu().detach().numpy().tolist()
        aitool.show_thetaobb(img, gt)
    output_file = os.path.join(dpath, img_name+'.png')
    cv2.imwrite(output_file,img)

if __name__ == '__main__':
    register_all_modules()

    # 根据配置文件和checkpoint文件构建模型
    # config_file = '/home/czj/mmrotate-dev-1.x/myself/refine/fcos_fspn.py'
    config_file='/home/czj/mmrotate/cfg_ship/MCSD/orientedrcnn.py'
    # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
    # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint_file = '/home/czj/mmrotate/work_dirs/orientedrcnn/epoch_40.pth'
    device = 'cuda:0'
    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)
    # print(model)

    model = model.eval()

    # 测试单张图片并生成推理结果
    valjsonpath = '/home/czj/data/ShipWhu/val/ship-whu-val.json'
    # imgpath = '/home/czj/data/ShipWhu/val/images/100001518_0_0.png' 
    
    imgpath ='/home/czj/data/ShipWhu/val/images/1439_476_476.png'
    
    anno_data_sample = imgpath2annotation(valjsonpath, imgpath)

    dpath = '/home/czj/mmrotate/cfg_ship/MCSD/show_twostage_anck/'

    transformss = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((1024, 1024)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # #注意如果有中文路径需要先解码，最好不要用中文
    img = cv2.imread(imgpath)
    ori_img = img.copy() #bgr
    # print(ori_img.shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    #转换维度   
    img = transformss(img).unsqueeze(0)

    rpn_list ,  pred_list= model(img.cuda(), [anno_data_sample],mode='predict')#,mode="predict"

    # print(rpn_list) #instancedata: scores; labels; bboxes

    # pdb.set_trace()
    img_name = os.path.split(imgpath)[1]
    # show_gt_pred(ori_img, pred_list , dpath ,img_name=img_name+'gt_pred')
    show_rpn(ori_img,anno_data_sample, rpn_list, dpath, img_name=img_name+'rpn_tp')
    # num_pred_TP: 159   iouthr =0.5
    # num_pred_FP: 1841
    
    #小目标
    #num_pred_TP: 30
# num_pred_FP: 2
# num_pred_TP: 323
# num_pred_FP: 1677
    
    
    
    # just_show_gt(ori_img,anno_data_sample, dpath, img_name=img_name+'gt')
    










# print(_[0].size())
# for i in range(len):
#     # conv_features.append(_[i].cpu())
    

#     conv_features = _[i].cpu()
#     heat = conv_features.squeeze(0)#降维操作,尺寸变为(C,H,W)
#     heatmap = torch.mean(heat,dim=0)#对各卷积层(C)求平均值,尺寸变为(H,W)
#     # heatmap = torch.max(heat,dim=1).values.squeeze()

#     heatmap = heatmap.detach().numpy()#转换为numpy数组
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)#minmax归一化处理
#     heatmap = cv2.resize(heatmap,(ori_img.shape[1],ori_img.shape[0]))#变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
#     heatmap = np.uint8(255*heatmap)#像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
#     heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)#颜色变换
   
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     # cv2.imwrite(os.path.join(dpath,'feat1.jpg'), img_)
#     cv2.imwrite(os.path.join(dpath,'ori729feat{}.jpg'.format(i)), heatmap)








