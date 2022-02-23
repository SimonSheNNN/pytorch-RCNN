import numpy as np
import torch
from functools import partial
import torchvision


def compute_overlaps_masks(masks1, masks2):
    #input is np
    masks1=np.transpose(masks1,(1,2,0))
    masks2=np.transpose(masks2,(1,2,0))
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

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


# This function flattens the output of the network and the corresponding anchors
# in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
# the FPN levels from all the images into 2D matrices
# Each row correspond of the 2D matrices corresponds to a specific grid cell
# Input:
#       out_r: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}
#       out_c: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
#       anchors: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
# Output:
#       flatten_regr: (total_number_of_anchors*bz,4)
#       flatten_clas: (total_number_of_anchors*bz)
#       flatten_anchors: (total_number_of_anchors*bz,4)
def output_flattening(out_r, out_c, anchors):

    return flatten_regr, flatten_clas, flatten_anchors

def output_flattening(out_r,out_c,anchors):
    #######################################
    # TODO flatten the output tensors and anchors
    #######################################
    flatten_regr = out_r.permute((0,2,3,1)).flatten(end_dim=-2)
    flatten_clas = out_c.permute((0,2,3,1)).flatten()
    flatten_anchors = torch.stack([anchors]*len(out_r), dim=0).flatten(end_dim=-2)
    return flatten_regr, flatten_clas, flatten_anchors
    
# This function decodes the output that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it returns the upper left and lower right corner of the bbox
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
# def output_decoding(flatten_out, flatten_anchors, device='cpu'):

#     return box


# This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
# a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
# Input:
#      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
#      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
#      P: scalar
# Output:
#      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
def MultiScaleRoiAlign(fpn_feat_list,proposals,P=14):
    fpn_boxes = [[],[],[],[]]           #For each level of feature pyramid
    orig_height= 800 
    orig_width = 1088
    feature_vectors=[]
    for i in range(len(proposals)):
        for j in range(proposals[i].shape[0]):
            x1,y1,x2,y2 = proposals[i][j]
            w = x2-x1
            h = y2-y1
            k = torch.clip(torch.floor(4+torch.log2(torch.sqrt(w*h)/224)),2,5).int()
        
            stride_x = orig_width/fpn_feat_list[k-2].shape[3]
            stride_y = orig_height/fpn_feat_list[k-2].shape[2]

            box = proposals[i][j].reshape(1,-1).clone()
            box[:,0] = box[:,0] / stride_x 
            box[:,2] = box[:,2] / stride_x
            box[:,1] = box[:,1] / stride_y
            box[:,3] = box[:,3] / stride_y
            # import pdb; pdb.set_trace()   
            inp = fpn_feat_list[k-2][i].unsqueeze(0)  # dim: (1,256,H_feat,W_feat)
            op  = torchvision.ops.roi_align(inp, [box], output_size=P, 
                                  spatial_scale=1,
                                  sampling_ratio=-1)  # dim : (1,256,P,P)

            feature_vectors.append(op[0]) #.append(op.view(-1))
            #fpn_boxes[k-2].append(element)
    
    feature_vectors = torch.stack(feature_vectors) #dim = (total_proposals, 256*P*P)
    
    '''
    output = [] #Dim len(fpn_list), element: tensor: (K,C,P,P)
    for i, level in enumerate(fpn_boxes):
        level_proposals = torch.stack(level,dim=1)[0][:,1:6]   # dim: (K,5) 
        output.append(torchvision.ops.roi_align(fpn_feat_list[i], level_proposals, output_size=P, 
                                  spatial_scale=orig_height/fpn_feat_list[i].shape[2],
                                  sampling_ratio=4))
        
    '''  
        
    return feature_vectors


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


def prepareimg(nms_labels,mask_output,nms_boxes,nms_scores,display=True):
  cls = nms_labels[0]
  mask_one_cls = mask_output[torch.LongTensor(range(mask_output.shape[0])), cls] # [100, 28, 28] get corresponding label mask
  mask_one_cls = mask_one_cls.unsqueeze(1)
  mask_one_cls

  keep=nms_boxes[0]
  x1, y1, x2, y2 = keep[:,0], keep[:,1], keep[:,2], keep[:,3]
  h = y2 - y1
  w = x2 - x1
  
  mask_total = torch.tensor([]).to(device)
  for idx in range(mask_output.shape[0]):

      if int(h[idx]) < 2 or int(w[idx]) < 2:
          continue
      mask_one = mask_one_cls[idx].unsqueeze(0)               # [1, 1, 28, 28]
      # print(mask_one.shape)
      mask_rescaled = F.interpolate(mask_one, size = (int(h[idx]), int(w[idx])), mode='bilinear', align_corners=True).squeeze()   # [h, w]

      # Padding
      p2d = (int(x1[idx]), 1088-int(x1[idx])-int(w[idx]), int(y1[idx]), 800-int(y1[idx])-int(h[idx]))
      mask_padded = F.pad(mask_rescaled, p2d).unsqueeze(0)    # [800, 1088]
      mask_total = torch.cat((mask_total,mask_padded), dim=0) # [100, 800, 1088]

  # Convert to binary form
  positive_idx = torch.where(mask_total >= 0.4)
  negative_idx = torch.where(mask_total < 0.4)
  mask_total[positive_idx] = 1
  mask_total[negative_idx] = 0
  if display==False:
    return display_instance(nms_scores[0], mask_total, nms_labels[0], images[0], keep[:,:4],5,display=display)
  else:
    display_instance(nms_scores[0], mask_total, nms_labels[0], images[0], keep[:,:4],5,display=display)



def display_instance(scores, masks, labels, image, boxes,topK,display=True):
    image = ((np.copy(image.cpu().numpy().transpose(1, 2, 0)) * np.array([0.229, 0.224, 0.225]) +np.array([0.485, 0.456, 0.406])) * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print(scores.shape,labels.shape,boxes.shape,masks.shape)
    # return
    
    if masks.shape[0]!=scores.shape[0]:
      scores=scores[:masks.shape[0]]
      labels=labels[:masks.shape[0]]
      boxes=boxes[:masks.shape[0]]

    # labels_idx= labels != 0
    # scores=scores[labels_idx]
    # labels=labels[labels_idx]
    # boxes=boxes[labels_idx]
    # masks=masks[labels_idx]

    _, idx_sorted = torch.sort(scores, descending=True)
    scores = scores[idx_sorted]
    scores = scores[:topK]
    labels = labels[idx_sorted]
    labels = labels[:topK]
    boxes = boxes[idx_sorted]
    boxes = boxes[:topK]
    masks = masks[idx_sorted]
    masks = masks[:topK]

    idx_filter=scores>0.9
    scores = scores[idx_filter]
    masks = masks[idx_filter]
    boxes = boxes[idx_filter]
    labels = labels[idx_filter]

    # print(labels)
    # return

    if display==False:
      return scores, masks, labels, boxes
    scores, masks, labels, boxes=scores.detach().cpu().numpy(),masks.detach().cpu().numpy(), labels.detach().cpu().numpy(),  boxes.detach().cpu().numpy()
    
    class_names = {3: "vehicle", 1: 'people', 2: 'animal'}

    # Number of instances
    N = boxes.shape[0]

    # Generate random colors
    def random_colors(N, bright=True):
      brightness = 1.0 if bright else 0.7
      hsv = [(i / N, 1, brightness) for i in range(N)]
      colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
      random.shuffle(colors)
      return colors
    colors = random_colors(N)

    # Plot bounding boxes
    for i in range(N):
        color = colors[i]
        # Bounding box
        x1, y1, x2, y2 = boxes[i,:]
        w = x2 - x1
        h = y2 - y1

        if labels[i] == 1:    
            # continue
            color_box = 'g'
        elif labels[i] == 2:
            color_box = 'b'  
        elif labels[i] == 3:  
            color_box = 'r'

        ax = plt.gca()
        rect = patches.Rectangle((x1,y1),w,h,linewidth=2,edgecolor=color_box,facecolor='none')
        ax.add_patch(rect)

        # Label
        label = labels[i]
        score = scores[i] if scores is not None else None
        label_name = class_names[int(label)]
        caption = "{} {:.3f}".format(label_name, score) if score else label_name
        ax.text(x1, y1, caption, color='w', size=12, backgroundcolor="black")
        # ax.text(x1, y1, caption,
        #         color='w', size=12, backgroundcolor="b")


        # Mask
        mask = masks[i].astype(np.uint32)
        def apply_mask(image, mask, color, alpha=0.5):

          for c in range(3):
              image[:, :, c] = np.where(mask == 1,
                                        image[:, :, c] *
                                        (1 - alpha) + alpha * color[c] * 255,
                                        image[:, :, c])
          return image
        image = apply_mask(image, mask, color)
        ax.imshow(image.astype(np.uint8))

    plt.show()