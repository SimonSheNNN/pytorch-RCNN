import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *

import torchvision


class RPNHead(torch.nn.Module):

    def __init__(self,  device='cuda', anchors_param=dict(ratio=0.8,scale= 256, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()

        self.device=device
        # TODO Define Backbone
        channels = [16, 32, 64, 128, 256]
        inchannel = 3
        self.backbone = nn.ModuleList()
        for i, chn in enumerate(channels):
            self.backbone.append(nn.Conv2d(inchannel, chn, 5, padding="same"))
            self.backbone.append(nn.BatchNorm2d(chn))
            self.backbone.append(nn.ReLU())
            if i < len(channels) - 1:
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            inchannel = chn


        # TODO  Define Intermediate Layer
        self.intermediate = nn.ModuleList()
        self.intermediate.append(nn.Conv2d(channels[-1], 256, 3, padding="same"))
        self.intermediate.append(nn.BatchNorm2d(256))
        self.intermediate.append(nn.ReLU())

        # TODO  Define Proposal Classifier Head
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Conv2d(256, 1, 1, padding="same"))
        self.classifier.append(nn.Sigmoid())

        # TODO Define Proposal Regressor Head
        self.regressor = nn.ModuleList()
        self.regressor.append(nn.Conv2d(256, 4, 1, padding="same"))

        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}



    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):

        #TODO forward through the Backbone
        X = self.forward_backbone(X)

        #TODO forward through the Intermediate layer
        for i, layer in enumerate(self.intermediate):
            X = layer(X)

        #TODO forward through the Classifier Head
        logits = X
        for i, layer in enumerate(self.classifier):
            logits = layer(logits)

        #TODO forward through the Regressor Head
        bbox_regs = X
        for i, layer in enumerate(self.regressor):
            bbox_regs = layer(bbox_regs)
        
        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs




    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    def forward_backbone(self,X):
        #####################################
        # TODO forward through the backbone
        #####################################
        for i, layer in enumerate(self.backbone):
            X = layer(X)

        assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return X



    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        anchors = torch.zeros((grid_sizes[0] , grid_sizes[1],4), device=self.device)
        img_h, img_w = grid_sizes[0] * stride, grid_sizes[1] * stride
        h = scale / aspect_ratio ** 0.5
        w = scale * aspect_ratio ** 0.5

        for i in range(grid_sizes[0]):
          for j in range(grid_sizes[1]):
            # x1 = max(stride*j+stride/2 - w/2, 0)
            # x2 = min(stride*j+stride/2 + w/2, img_w)
            # y1 = max(stride*i+stride/2 - h/2, 0)
            # y2 = min(stride*i+stride/2 + h/2, img_h)
            # anchors[i][j] = torch.Tensor([x1,y1,x2,y2])
            x = stride*j+stride/2
            y = stride*i+stride/2
            anchors[i][j] = torch.Tensor([x,y,w,h])

        ######################################
        # TODO create anchors
        ######################################
        assert anchors.shape == (grid_sizes[0] , grid_sizes[1],4)

        return anchors



    def get_anchors(self):
      
        return self.anchors



    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        bz = len(bboxes_list)
        ground_clas, ground_coord = MultiApply(self.create_ground_truth, bboxes_list, indexes, [self.anchors_param['grid_size']]*bz, [self.anchors]*bz, [image_shape]*bz)
        ground_clas = torch.stack(ground_clas, dim=0)
        ground_coord = torch.stack(ground_coord, dim=0)
        #####################################
        # TODO create ground truth for a batch of images
        #####################################
        assert ground_clas.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_clas, ground_coord


    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0],grid_size[1],4)
    # Output:
    #       ground_clas:  (1,grid_size[0],grid_size[1])
    #       ground_coord: (4,grid_size[0],grid_size[1])
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        new_bboxes = convert_xywh(bboxes)
        ious = []

        flatten_anchors = torch.flatten(anchors, end_dim=-2)
        flatten_anchors_xyxy = convert_xyxy(flatten_anchors)
        for bbox in bboxes:
            iou = IOU(flatten_anchors_xyxy, bbox)
            ious.append(iou)
        ious = torch.stack(ious, 1)
        ground_clas = torch.zeros(grid_size[0] * grid_size[1], device=self.device)
        ground_clas[(ious < 0.3).all(dim=1)] = -1
        ground_clas[ious.max(dim=0).indices] = 1
        ground_clas[(ious > 0.7).any(dim=1)] = 1
        ground_coord = torch.zeros((grid_size[0] * grid_size[1],4), device=self.device)
        assigned_boxes_indices = ious.max(dim=1).indices[ground_clas==1]
        # ground_coord[ground_clas==1] = new_bboxes[assigned_boxes_indices]
        assigned_anchors = flatten_anchors[ground_clas==1]
        assigned_boxes = new_bboxes[assigned_boxes_indices]
        assigned_boxes_relative = torch.zeros_like(assigned_boxes, device=self.device)
        assigned_boxes_relative[:, 0] = (assigned_boxes[:, 0] - assigned_anchors[:, 0]) / assigned_anchors[:, 2]
        assigned_boxes_relative[:, 1] = (assigned_boxes[:, 1] - assigned_anchors[:, 1]) / assigned_anchors[:, 3]
        assigned_boxes_relative[:, 2] = torch.log(assigned_boxes[:, 2]/assigned_anchors[:, 2])
        assigned_boxes_relative[:, 3] = torch.log(assigned_boxes[:, 3]/assigned_anchors[:, 3])

        ground_coord[ground_clas==1] = assigned_boxes_relative
        
        ground_clas = ground_clas.reshape((grid_size[0],grid_size[1],1)).permute((2,0,1))
        ground_coord = ground_coord.reshape((grid_size[0],grid_size[1], 4)).permute((2,0,1))
        #####################################################
        # TODO create ground truth for a single image
        #####################################################

        self.ground_dict[key] = (ground_clas, ground_coord)

        assert ground_clas.shape==(1,grid_size[0],grid_size[1])
        assert ground_coord.shape==(4,grid_size[0],grid_size[1])

        return ground_clas, ground_coord


    



    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self,p_out,n_out):
        criterion = torch.nn.BCELoss(reduction="sum")
        loss = criterion(p_out, torch.ones(len(p_out), device=self.device))
        loss += criterion(n_out, torch.zeros(len(n_out), device=self.device))
        sum_count = len(p_out) + len(n_out)
        #torch.nn.BCELoss()
        # TODO compute classifier's loss

        return loss,sum_count



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r):
        criterion =  torch.nn.SmoothL1Loss(reduction="sum")
        loss = criterion(pos_out_r, pos_target_coord)
        sum_count = len(pos_target_coord) * 4
        return loss, sum_count 
        
            #torch.nn.SmoothL1Loss()
            # TODO compute regressor's loss

            # return loss, sum_count



    # Compute the total loss
    # Input:xs
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    def compute_loss(self,clas_out,regr_out,targ_clas,targ_regr, l=1, effective_batch=50):
        flatten_clas_out = torch.flatten(clas_out.permute((0,2,3,1)))
        flatten_regr_out = torch.flatten(regr_out.permute((0,2,3,1)), end_dim=-2)
        flatten_targ_clas = torch.flatten(targ_clas.permute((0,2,3,1)))
        flatten_targ_regr = torch.flatten(targ_regr.permute((0,2,3,1)), end_dim=-2)
        find_cor=(flatten_targ_clas==1).nonzero().view(-1)
        find_neg=(flatten_targ_clas==-1).nonzero().view(-1)
        n_pos = len(find_cor)
        n_neg = len(find_neg)
        find_cor = find_cor[torch.randperm(n_pos)][:min(effective_batch//2, n_pos)]
        find_neg = find_neg[torch.randperm(n_neg)][:max(effective_batch//2, effective_batch-n_pos)]
        p_out = flatten_clas_out[find_cor]
        n_out = flatten_clas_out[find_neg]
        pos_target_coord = flatten_targ_regr[find_cor]
        pos_out_r = flatten_regr_out[find_cor]
        loss_c, c_count = self.loss_class(p_out,n_out)
        loss_c = loss_c / c_count
        loss_r, r_count = self.loss_reg(pos_target_coord,pos_out_r)
        loss_r = loss_r / r_count

        return loss_c + loss_r, loss_c, loss_r
        
            #############################
            # TODO compute the total loss
            #############################
            # return loss, loss_c, loss_r



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
    def postprocess(self,out_c,out_r, batch_image, indexes, IOU_thresh, keep_num_preNMS, keep_num_postNMS ):
      #  print(IOU_thresh)
       batch_size = out_c.shape[0]
       nms_clas=[]
       nms_prebox=[]
       for i in range(batch_size):
           one_clas, one_prebox = self.postprocessImg(out_c[i], out_r[i], batch_image[i], indexes[i], IOU_thresh, keep_num_preNMS, keep_num_postNMS)
           nms_clas.append(nms_clas)
           nms_prebox.append(nms_prebox)

       return nms_clas, nms_prebox



    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)


    def postprocessImg(self,mat_clas,mat_coord, image, index, IOU_thresh,keep_num_preNMS, keep_num_postNMS):
            reg_out = torch.unsqueeze(mat_coord, 0)
            cls_out = torch.unsqueeze(mat_clas, 0)
            flatten_coord, flatten_clas, flatten_anchors = output_flattening(reg_out, cls_out, self.anchors)
            preNMS= output_decoding(flatten_coord, flatten_anchors)
            a = [torch.rand(flatten_coord.shape[0]) > 0.5 for i in range(4)]
            x_low_outbound = (preNMS[:, 0] < 0)
            y_low_outbound = (preNMS[:, 1] < 0)
            x_high_outbound = (preNMS[:, 2] > 1088)
            y_high_outbound = (preNMS[:, 3] > 800)
            a[0] = x_low_outbound
            a[1] = y_low_outbound
            a[2] = x_high_outbound
            a[3] = y_high_outbound
            out_mask = (torch.sum(torch.stack(a), dim=0) > 0)
            flatten_clas[out_mask] = 0

            top_values, top_indices = torch.topk(flatten_clas, keep_num_preNMS)
            last_value = top_values[-1]
            topk_mask = flatten_clas >= last_value
            topk_clas = flatten_clas[topk_mask]
            topk_box = preNMS[topk_mask]
            plot_NMS(topk_box, image, "PreNMS",index, "Pre", keep_num_preNMS)

            nms_clas, nms_prebox = self.NMS(topk_clas,topk_box, IOU_thresh)

            num = min(nms_prebox.shape[0],keep_num_postNMS)
            top_values, top_indices = torch.topk(nms_clas, num)
            last_value = top_values[-1]
            topk_mask = nms_clas >= last_value
            clas = nms_clas[topk_mask]
            box = nms_prebox[topk_mask]
            plot_NMS(box, image, "PostNMS",index, "Post", keep_num_postNMS)
            return clas, box


    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)

    def NMS(self,clas,prebox, thresh):
        ##################################
        # TODO perform NSM
        ##################################

        iou = torch.zeros((prebox.shape[0],  prebox.shape[0]),device=self.device)
        for x in range( prebox.shape[0]):
            for y in range( prebox.shape[0]):
                iou[x, y] = IOU_edge_point(torch.unsqueeze(prebox[x, :], 0), torch.unsqueeze(prebox[y, :], 0))
        max_index = set()

        for idx in range(len(iou)):   
            above = True
            below = []                      
            for prev in max_index:         
                if iou[idx, prev] > thresh:
                    if clas[idx] > clas[prev]:            
                        below.append(prev)
                    else:
                        above = False                             

            max_index.difference_update(below)
            if above:                                              
                max_index.add(idx)

        nms_clas = clas[list(max_index)]
        nms_prebox = prebox[list(max_index), :]
        return nms_clas, nms_prebox