import numpy as np
import torch
from functools import partial
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rpn import *
import matplotlib.patches as patches

import torchvision

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function computes the IOU between two set of boxes
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

def IOU_edge_point(boxA, boxB):
    x_1a, y_1a, x_1b, y_1b = boxA[:, 0], boxA[:, 1], boxA[:, 2], boxA[:, 3]
    x_2a, y_2a, x_2b, y_2b = boxB[:, 0], boxB[:, 1], boxB[:, 2], boxB[:, 3]

    x_a = torch.max(x_1a, x_2a)
    y_a = torch.max(y_1a, y_2a)

    x_b = torch.min(x_1b, x_2b)
    y_b = torch.min(y_1b, y_2b)

    inter_area = (x_b - x_a).clamp(min=0) * (y_b - y_a).clamp(min=0)

    area_boxa = (x_1b - x_1a).clamp(min=0) * (y_1b - y_1a).clamp(min=0)
    area_boxb = (x_2b - x_2a).clamp(min=0) * (y_2b - y_2a).clamp(min=0)
    union_area = area_boxa + area_boxb - inter_area

    return (inter_area + 1e-3) / (union_area + 0.0000001)

# This function flattens the output of the network and the corresponding anchors 
# in the sense that it concatenates  the outputs and the anchors from all the grid cells
# from all the images into 2D matrices
# Each row of the 2D matrices corresponds to a specific anchor/grid cell
# Input:
#       out_r: (bz,4,grid_size[0],grid_size[1])
#       out_c: (bz,1,grid_size[0],grid_size[1])
#       anchors: (grid_size[0],grid_size[1],4)
# Output:
#       flatten_regr: (bz*grid_size[0]*grid_size[1],4)
#       flatten_clas: (bz*grid_size[0]*grid_size[1])
#       flatten_anchors: (bz*grid_size[0]*grid_size[1],4)
def output_flattening(out_r,out_c,anchors):
    #######################################
    # TODO flatten the output tensors and anchors
    #######################################
    flatten_regr = out_r.permute((0,2,3,1)).flatten(end_dim=-2)
    flatten_clas = out_c.permute((0,2,3,1)).flatten()
    flatten_anchors = torch.stack([anchors]*len(out_r), dim=0).flatten(end_dim=-2)
    return flatten_regr, flatten_clas, flatten_anchors




# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
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

def keep_top_K_batch(clas, K):
    out_res = torch.zeros_like(clas)
    for i in range(clas.shape[0]):
        tmp = torch.flatten(clas[i])
        top_values, _ = torch.topk(tmp, K)
        last_value = top_values[-1]
        mask = clas[i] >= last_value
        out_res[i, mask] = clas[i, mask]
    return out_res

def plot_mask_batch(rpn_net, clas_out_raw, reg_out, images, boxes, indice, result_dir, top_K):

    if top_K is None:
        clas_out = clas_out_raw
    else:
        clas_out = keep_top_K_batch(clas_out_raw, top_K)

    flatten_coord, flatten_clas, flatten_anchors = output_flattening(reg_out, clas_out, rpn_net.get_anchors())
    decoded_coord = output_decoding(flatten_coord, flatten_anchors)

    find_cor_bz = flatten_clas.nonzero().squeeze(dim=1)

    batch_size = len(boxes)
    total_number_of_anchors = clas_out.shape[2] * clas_out.shape[3]
    for i in range(batch_size):
        image = torchvision.transforms.functional.normalize(images[i].cpu().detach(),
                                                [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        fig, ax = plt.subplots(1, 1)
        image_vis = image.permute(1, 2, 0).cpu().detach().numpy()
        ax.imshow(image_vis)

        mask = ((find_cor_bz > i * total_number_of_anchors).flatten() * 1.0 + (find_cor_bz < (i + 1) * total_number_of_anchors).flatten() * 1.0) == 2.0
        find_cor = find_cor_bz[mask]

        for elem in find_cor:
            coord = decoded_coord[elem, :].view(-1)
            anchor = flatten_anchors[elem, :].view(-1)
            coord = coord.cpu().detach().numpy()
            anchor = anchor.cpu().detach().numpy()

            rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,color='b')
            ax.add_patch(rect)

        plt.savefig("{}/{}.png".format(result_dir, indice[i]))
        plt.show()
        plt.close('all')


def plot_NMS(flatten_box, image, visual_dir, index, mode, top_num):
        image = transforms.functional.normalize(image.cpu().detach(),
                                                [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        num_box=flatten_box.shape[0]
        fig, ax = plt.subplots(1, 1)
        image_vis = image.permute(1, 2, 0).cpu().detach().numpy()
        ax.imshow(image_vis)

        for elem in range(num_box):
            coord = flatten_box[elem, :].view(-1)
            coord = coord.cpu().detach().numpy()
            # plot bbox
            rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                     color='b')
            ax.add_patch(rect)

        plt.title("{} NMS Top {}".format(mode, top_num))
        plt.savefig("{}/{}.png".format(visual_dir, index))
        plt.show()
        plt.close('all')





