import json
import numpy
import os
import aitool
import cv2
import torch


if __name__ == '__main__':

    res_file = '/home/czj/mmdet-rfla/vis_tools/vis_feature/det_result_json/aitodv2_qkldllefpn_det.json'  #检测结果 
    ann_file = '/home/czj/data/AI-TODv2/annotations/small_test_v2_2.0.json'
    img_dir = '/home/czj/data/AI-TODv2/test'
    # sample_basenames = aitool.get_basename_list(
    #     '/home/czj/mmrotate/data/RSDD-SAR/testimg')
    # samples = ['75_1_3.jpg']
    save_dir = '/home/czj/mmdet-rfla/vis_tools/vis_feature/det_result_json/aitodv2_qkld_llefpnthr0.2_det'   #检测结果可视化

    score = 0.2#可视化的confidence阈值

    final = dict()
    import pdb

    # read the predicted results
    file = open(res_file, 'r')
    contents = file.readlines()
    all_predictions = []
    for content in contents:
        item = json.loads(content)
        all_predictions.append(item)
        #print(item['image_id'])

    coco_parser = aitool.COCOParser(ann_file)
    
    objects = coco_parser.objects
    img_name_with_id = coco_parser.img_name_with_id
    #coco_result_parser = aitool.COCOJsonResultParser(res_file)
    
    for ori_img_name in list(objects.keys())[::-1]:
        #img_name = prediction['image_id']
        #print(img_name)
        img_name = ori_img_name + '.png'
        # if img_name in samples:
        #     continue
        # print('img_name:',img_name) #img_name: 1233__1800_0.png
        # print('ori_img_name:',ori_img_name) #ori_img_name: 1233__1800_0
        ground_truth = coco_parser(ori_img_name)

        img = cv2.imread(os.path.join(img_dir, img_name))

        gt_bboxes, pred_bboxes = [], []
        for _ in ground_truth:
            # print(_) 没问题
            gt_bboxes.append(_['bbox'])  #_[]
            
      

        for prediction in all_predictions:
            # print('prediction:',prediction)
            # prediction: {'image_id': 'P1871__1.0__6000___600.png', 'bbox': [581.1824340820312, 540.42724609375, 592.9332275390625, 558.970703125], 'score': 0.02832854352891445, 'category_id': 0}
            
            if prediction['image_id'] != img_name:
                # print('image_id:',prediction['image_id'])
                continue
            if prediction['score'] < score:
                # print('score:',prediction['score'])
      
                continue
           

            box = prediction['bbox']
            # box_tensor = torch.from_numpy(numpy.array(box))
            pred_bboxes.append(box)  

        # pdb.set_trace()
        # gt_bboxes = aitool.drop_invalid_bboxes([_ for _ in gt_bboxes])
        pred_bboxes = aitool.drop_invalid_bboxes([_ for _ in pred_bboxes])
        gt_bboxes = aitool.drop_invalid_bboxes([aitool.xywh2xyxy(_) for _ in gt_bboxes])
        # pred_bboxes = aitool.drop_invalid_bboxes([aitool.xywh2xyxy(_) for _ in pred_bboxes])
        # print('gt_bboxes:',gt_bboxes)
        # print('pred_bboxes:',pred_bboxes)
        # pdb.set_trace()

        if len(gt_bboxes) < 1:
            print('no gt bbox')
            continue

        img = aitool.draw_confusion_matrix(img, gt_bboxes, pred_bboxes, with_gt_TP=False, line_width=2)
        print('done',ori_img_name)

        if isinstance(img, list):
            continue

        output_file = os.path.join(save_dir, img_name)
        cv2.imwrite(output_file,img)




    






# def obb2poly_le135(rboxes):
#     """Convert oriented bounding boxes to polygons.

#     Args:
#         obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

#     Returns:
#         polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
#     """
#     N = rboxes.shape[0]
#     if N == 0:
#         return rboxes.new_zeros((rboxes.size(0), 8))
#     x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
#         1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
#     tl_x, tl_y, br_x, br_y = \
#         -width * 0.5, -height * 0.5, \
#         width * 0.5, height * 0.5
#     rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
#                         dim=0).reshape(2, 4, N).permute(2, 0, 1)
#     sin, cos = torch.sin(angle), torch.cos(angle)
#     M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
#                                                         N).permute(2, 0, 1)
#     polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
#     polys[:, ::2] += x_ctr.unsqueeze(1)
#     polys[:, 1::2] += y_ctr.unsqueeze(1)
#     return polys.contiguous()
        

# if __name__ == '__main__':
#rotated
#     res_file = '/home/xuchang/mmrotate/configs/dotav2_vis/v001.01.00.json'
#     ann_file = '/data/xuchang/DOTA-V2.0/vis_obb/data_500/ann_coco/dota_vis500_v2_best_keypoint.json'
#     img_dir = '/data/xuchang/DOTA-V2.0/vis_obb/data_500/img'
#     sample_basenames = aitool.get_basename_list(
#         '/data/xuchang/DOTA-V2.0/vis_obb/data_500/img')
#     samples = ['P0003__1024__0___0.png']
#     save_dir = '/home/xuchang/cctools/visualization/samples/vis_dota/vis_retina_base'

#     score = 0.3
#     final = dict()

#     # read the predicted results
#     file = open(res_file, 'r')
#     contents = file.readlines()
#     all_predictions = []
#     for content in contents:
#         item = json.loads(content)
#         all_predictions.append(item)
#         #print(item['image_id'])

#     coco_parser = aitool.COCOParser(ann_file)
#     objects = coco_parser.objects
#     img_name_with_id = coco_parser.img_name_with_id
#     #coco_result_parser = aitool.COCOJsonResultParser(res_file)
    
#     for ori_img_name in list(objects.keys())[::-1]:
#         #img_name = prediction['image_id']
#         #print(img_name)
#         img_name = ori_img_name + '.png'
#         if img_name in samples:
#             continue
#         ground_truth = coco_parser(ori_img_name)

#         img = cv2.imread(os.path.join(img_dir, img_name))

#         gt_bboxes, pred_bboxes = [], []
#         for _ in ground_truth:
#             gt_bboxes.append(_['pointobb'])  

#         for prediction in all_predictions:
#             if prediction['image_id'] != img_name:
#                 continue
#             if prediction['score'] < score:
#                 continue
#             box_le135 = prediction['bbox']
#             box_tensor = torch.from_numpy(numpy.array(box_le135))
#             box_poly = obb2poly_le135(box_tensor.unsqueeze(dim=0)).squeeze().numpy()
    
#             pred_bboxes.append(box_poly)   
#             #print(box_poly) 


#         #gt_bboxes = aitool.drop_invalid_bboxes([aitool.xywh2xyxy(_) for _ in gt_bboxes])
#         #pred_bboxes = aitool.drop_invalid_bboxes([aitool.xywh2xyxy(_) for _ in pred_bboxes])

#         if len(gt_bboxes) < 1:
#             continue

#         img = aitool.draw_confusion_matrix_rotate(img, gt_bboxes, pred_bboxes, with_gt_TP=False, line_width=2)

#         if isinstance(img, list):
#             continue

#         output_file = os.path.join(save_dir, img_name)
#         cv2.imwrite(output_file,img)




    
