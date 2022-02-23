import numpy as np
import torch
from functools import partial
import cv2
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):
    xA = torch.max(boxA[:,0], boxB[0])
    yA = torch.max(boxA[:,1], boxB[1])
    xB = torch.min(boxA[:,2], boxB[2])
    yB = torch.min(boxA[:,3], boxB[3])

    interArea = torch.max(torch.tensor(0), xB - xA) * torch.max(torch.tensor(0), yB - yA)
    
    boxAArea = (boxA[:,2] - boxA[:,0]) * (boxA[:,3] - boxA[:,1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / (boxAArea + boxBArea - interArea)
    # return the intersection over union value
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes
    ##################################
    return iou


    
def convert_xyxy(bboxes):
    bboxes_xy = torch.zeros_like(bboxes)
    bboxes_xy[:,0] = torch.max(bboxes[:,0] - bboxes[:,2] / 2, torch.tensor(0))
    bboxes_xy[:,2] = bboxes[:,0] + bboxes[:,2] / 2
    bboxes_xy[:,1] = torch.max(bboxes[:,1] - bboxes[:,3] / 2, torch.tensor(0))
    bboxes_xy[:,3] = bboxes[:,1] + bboxes[:,3] / 2

    return bboxes_xy

def convert_xywh(bboxes):
    bboxes_xywh = torch.zeros_like(bboxes)
    bboxes_xywh[:,0] = bboxes[:,0]+(bboxes[:,2]-bboxes[:,0])/2
    bboxes_xywh[:,1] = bboxes[:,1]+(bboxes[:,3]-bboxes[:,1])/2
    bboxes_xywh[:,2] = bboxes[:,2]-bboxes[:,0]
    bboxes_xywh[:,3] = bboxes[:,3]-bboxes[:,1]

    return bboxes_xywh

# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x,y,w,h] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decoding(flatten_out,flatten_anchors, device='cpu'):
    box = torch.zeros_like(flatten_out, device=device)
    box[:, 0] = flatten_out[:, 0] * flatten_anchors[:, 2] + flatten_anchors[:, 0]
    box[:, 1] = flatten_out[:, 1] * flatten_anchors[:, 3] + flatten_anchors[:, 1]
    box[:, 2] = torch.exp(flatten_out[:, 2]) * flatten_anchors[:, 2]
    box[:, 3] = torch.exp(flatten_out[:, 3]) * flatten_anchors[:, 3]
    
    #######################################
    # TODO decode the output
    #######################################
    return convert_xyxy(box)



def matrix_IOU(boxA, boxB, device='cpu'):
  inter_x1 = torch.max(boxA[..., 0], boxB[..., 0])
  inter_y1 = torch.max(boxA[..., 1], boxB[..., 1])
  inter_x2 = torch.min(boxA[..., 2], boxB[..., 2])
  inter_y2 = torch.min(boxA[..., 3], boxB[..., 3])
  inter = torch.max((inter_x2 - inter_x1), torch.zeros(inter_x2.shape).to(device)) * \
      torch.max((inter_y2 - inter_y1), torch.zeros(inter_x2.shape).to(device))
  iou = inter / ((boxA[..., 2] - boxA[..., 0]) * (boxA[..., 3] - boxA[..., 1]) +
                 (boxB[..., 2] - boxB[..., 0]) *
                 (boxB[..., 3] - boxB[..., 1]) - inter + 1)
  return iou

def visual_bbox_mask(image, bboxes,labels,scores = None,gt_bbox = None):
  out_img = ((np.copy(image.numpy().transpose(1, 2, 0)) * np.array([0.229, 0.224, 0.225]) +np.array([0.485, 0.456, 0.406])) * 255).astype(np.uint8)
  out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

  for i in range(bboxes.shape[0]):
    x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
    color = [0, 0, 0]
    color[labels[i].to(torch.int32)] = 255
    color = tuple(color)
    out_img = cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

  if gt_bbox is not None:
    for i in range(gt_bbox.shape[0]):
      x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox[i][0], gt_bbox[i][1], gt_bbox[i][2], gt_bbox[i][3]
      out_img = cv2.rectangle(out_img, (int(x1_gt), int(y1_gt)),(int(x2_gt), int(y2_gt)), (255, 255, 255), 3)

  return out_img


def IOU2(boxA, boxB):
  inter_x1 = max(boxA[0], boxB[0])
  inter_y1 = max(boxA[1], boxB[1])
  inter_x2 = min(boxA[2], boxB[2])
  inter_y2 = min(boxA[3], boxB[3])
  area_boxA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  area_boxB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  inter = max((inter_x2 - inter_x1), 0) * max((inter_y2 - inter_y1), 0)
  iou = inter / (area_boxA + area_boxB - inter + 1)
  return iou
