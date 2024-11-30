import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mmdet.core.bbox.iou_calculators.metric_calculator import bbox_overlaps


if __name__ == '__main__':
    #总画布大小为800*800
    imgw = 400
    imgh = 400
    img = np.zeros((imgw, imgh, 3), dtype=np.uint8)
    #将画布变白
    img.fill(255)
    #生成一个位于中心的矩形框
    cx = imgw // 2
    cy = imgh // 2
    #中心框的绝对大小
    is_small =True
  
   
    if is_small:
        small = 'small'
        abs_size = 8
    else:
        small = 'big'
        abs_size = 32
    x1 = cx - abs_size // 2
    y1 = cy - abs_size // 2
    x2 = cx + abs_size // 2
    y2 = cy + abs_size // 2
    bbox_center = [x1, y1, x2, y2]
    #在画布上绘制出矩形框
    cv2.rectangle(img, (bbox_center[0], bbox_center[1]), (bbox_center[2], bbox_center[3]), (255, 0, 0), 2)
    cv2.imwrite('/home/czj/mmdet-rfla/mmdet/core/bbox/iou_calculators/vis/centerbbox'+small+'.jpg', img)
    #生成与中心框位置从左向右平移的矩形框，每一步平移的距离为1，从中心框的最左端平移到中心框的最右端
    shift_bboxs = []
    # shift为正时：向右平移，shift为负时：向左平移
    shift_limit = abs_size*2
    x = np.arange(-shift_limit, shift_limit + 1, 0.01)
    
    for i in x:
        shift_bboxs.append([x1 + i, y1, x2 + i, y2])
    
    # #画出所有平移后的矩形框
    # for bbox in shift_bboxs:
    #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # cv2.imwrite('/home/czj/mmdet-rfla/mmdet/core/bbox/iou_calculators/vis/shiftbbox'+small+'.jpg', img)
    
    #计算中心框与所有平移后的矩形框的IoU, giou, kld,exp_kl
    bboxes1 = torch.tensor(bbox_center).view(1, 4)
    bboxes2 = torch.tensor(shift_bboxs).view(len(shift_bboxs), 4)
    ious = bbox_overlaps(bboxes1, bboxes2, mode='iou')
    
    # ious_01 = ious**0.1
    ious_08 = ious**0.8
    # gious = bbox_overlaps(bboxes1, bboxes2, mode='giou')
    klds = bbox_overlaps(bboxes1, bboxes2, mode='kl')
    siwd = bbox_overlaps(bboxes1, bboxes2, mode='siwd')
    klds_09 = klds**0.9#过于平缓了
    klds_08 = klds**0.8
    # klds_05 = klds**0.5
    siwd_08 = siwd**0.8
    # exp_kls = bbox_overlaps(bboxes1, bboxes2, mode='exp_kl')
    #创建画布
    plt.figure(figsize=(10, 5))
    #画出IoU曲线
  
        
    # x = range(-shift_limit, shift_limit + 1)
    plt.plot(x, ious.squeeze() , label='IoU')
    plt.plot(x, klds.squeeze() , label='KLD')
    plt.plot(x, siwd.squeeze() , label='SIWD')
    # plt.plot(x, ious_01.squeeze() , label='IoU_0.1')
    # plt.plot(x, klds_09.squeeze() , label='KLD_0.9') #0.8还是0.9表达的是变化程度
    # plt.plot(x, ious_08.squeeze() , label='IoU_0.8')
    # plt.plot(x, klds_08.squeeze() , label='KLD_0.8')
    # # plt.plot(x, klds_05.squeeze() , label='KLD_0.5')
    # plt.plot(x, siwd_08.squeeze() , label='SIWD_0.8')
    # plt.plot(x, exp_kls.squeeze() , label='exp_KL')
    
    plt.xlabel('Shift', fontsize=20)
    plt.ylabel('Localization score', fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('/home/czj/mmdet-rfla/mmdet/core/bbox/iou_calculators/vis/iou_giou_kld_step001'+small+'.jpg')
    
    
    
    