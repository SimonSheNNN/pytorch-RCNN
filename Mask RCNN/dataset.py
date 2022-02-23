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


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # TODO Initialize  Dataset
        #############################################
        self.img    = h5py.File(path[0],'r')['data']
        self.mask   = h5py.File(path[1],'r')['data']
        self.labels = np.load(path[2], allow_pickle=True)
        self.bounding_box   = np.load(path[3], allow_pickle=True)
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        self.new_masks = [] 
        i = 0 
        mask_shape = self.mask[0].shape
        for l in range(self.labels.shape[0]):
            temp = []
            for idx in range(self.labels[l].size):
                temp.append(self.mask[i,:,:])
                i+=1
            self.new_masks.append(temp)

    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index
    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################
        img   = self.img[index,:,:,:]
        label = self.labels[index]
        mask  = self.new_masks[index]
        bounding_box  = self.bounding_box[index]
        
        label = torch.tensor(label, dtype=torch.float)
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bounding_box)


        assert transed_img.shape == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        
        return transed_img, label, transed_mask, transed_bbox, index



    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox):
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes
        ######################################
        scale_factor_x = 800/300
        scale_factor_y = 1088/400
        
        processed_img   = torch.tensor(img.astype(np.float), dtype=torch.float)       
        processed_img =  torch.nn.functional.pad(self.normalize(torch.nn.functional.interpolate(torch.unsqueeze(processed_img/255, 0), size=(800, 1066), mode='bilinear') [0]), pad=(11,11), mode='constant')

        processed_bounding_box  = torch.tensor(bbox, dtype=torch.float)

        processed_masks = torch.zeros((len(mask),800,1088))
        for i, m in enumerate(mask):
            processed_mask = torch.unsqueeze(torch.unsqueeze(torch.tensor(m.astype(np.uint8), dtype=torch.float), 0), 0)
            processed_mask = torch.nn.functional.pad(torch.nn.functional.interpolate(processed_mask,size=(800, 1066), mode='bilinear'), pad=(11,11), mode='constant',value=0)
            processed_mask[processed_mask > 0.5] = 1
            processed_mask[processed_mask < 0.5] = 0
            processed_masks[i] = processed_mask[0,0,:,:]

        processed_bounding_box[:,0] = processed_bounding_box[:,0] * scale_factor_x +11
        processed_bounding_box[:,2] = processed_bounding_box[:,2] * scale_factor_x +11
        processed_bounding_box[:,1] = processed_bounding_box[:,1] * scale_factor_y
        processed_bounding_box[:,3] = processed_bounding_box[:,3] * scale_factor_y
        return processed_img, processed_masks, processed_bounding_box
        
        # assert img.squeeze(0).shape == (3, 800, 1088)
        # assert bbox.shape[0] == mask.squeeze(0).shape[0]

        # return img.squeeze(0), mask.squeeze(0), bbox
    

    
    def __len__(self):
        return len(self.img)




class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        images, labels, masks, bounding_boxes, indices = list(zip(*batch))
        return torch.stack(images), labels, masks, bounding_boxes, indices

        # return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)