#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:26:34 2019

@author: cordolo
"""
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

class BaseDataset(Dataset):

    def __init__(self, data_path, indices, patch_size_padded, channels):
        'Initialization'
        self.data_path = data_path
        self.indices= indices
        self.patch_size_padded = patch_size_padded
        self.channels = channels
        
        self.num_sample = len(self.indices)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
           
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)
    
    def data_augmentation(self, x, y):
        if bool(random.getrandbits(1)):
            x = np.fliplr(x)
            y = np.fliplr(y)
        if bool(random.getrandbits(1)):
            x = np.flipud(x)
            y = np.flipud(y)
        rot90 =random.randint(0,3)
        x = np.rot90(x, k=rot90)
        y = np.rot90(y, k=rot90)
    
        return(x,y)

    def img_transform(self, x):
        x = x[:,:, self.channels]
        x = torch.from_numpy(x).permute(2,0,1)
        return x

    def segm_transform(self, y):
        y = np.array(y, dtype=np.uint8)      
        y = torch.from_numpy(y).long()
        return y

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p
    
    # crop patch
    def crop(self, x, y):
        start = int((x.shape[0]-self.patch_size_padded)/2)
        x = x[start:self.patch_size_padded+start,start:self.patch_size_padded+start,:]
        y = y[start:self.patch_size_padded+start,start:self.patch_size_padded+start,:]
        return(x,y)
    
class TrainDataset(BaseDataset):
    
    def __init__(self, data_path, indices, patch_size_padded=96, channels=[0,1,2], n_classes = 5, y_downsampling_rate=4, augment=False):
        super(TrainDataset, self).__init__(data_path, indices, patch_size_padded, channels)
    
        self.n_channels = len(channels)
        self.n_classes = n_classes
        self.y_downsampling_rate = y_downsampling_rate
        self.augment = augment

    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        idx = self.indices[index]
    
        # Load data and get label
        patch = np.load(self.data_path + 'images/' + str(idx) + '.npy').astype(np.float32)
        y = np.load(self.data_path + 'labels/' + str(idx) + '.npy')
        assert(patch.shape[0] == y.shape[0])
        assert(patch.shape[1] == y.shape[1])
        
        # crop
        if self.patch_size_padded < patch.shape[0]:
            patch, y = self.crop(patch,y)
        
        # data augmentation
        if self.augment:
            patch, y = self.data_augmentation(patch, y)
            
        # image transform, to torch float tensor 3xHxW
        patch = self.img_transform(patch)
        patch = torch.unsqueeze(patch, 0) 
        
        # segmentation mask instead of one-hot-labels
        y = np.argmax(y, axis=2)
        
        # for binary problem (grassland vs. no grassland)
        if self.n_classes == 2:
            y_full = y
            y = np.zeros((self.patch_size_padded, self.patch_size_padded), dtype=np.uint8)
            y[(y_full == 1) | (y_full == 2)] = 0
            y[(y_full == 3) | (y_full == 4)] = 1
        
        # downsample label mask
        if self.y_downsampling_rate != 0:
            y = Image.fromarray(y.astype(np.uint8))
            y = imresize(
                y,
                (y.size[0] // self.y_downsampling_rate, \
                y.size[1] // self.y_downsampling_rate), \
                interp='nearest')
            
        # segm transform, to torch long tensor HxW
        y = self.segm_transform(y)
        y = torch.unsqueeze(y, 0) 
        
        output = dict()
        output['img_data'] = patch
        output['seg_label'] = y
        return output
    
class TestDataset(BaseDataset):
    
    def __init__(self, data_path, indices, patch_size_padded=96, channels=[0,1,2], n_classes = 5, augment=False):
        super(TestDataset, self).__init__(data_path, indices, patch_size_padded, channels)

        self.n_classes = n_classes      
        self.augment = augment
    
    def __getitem__(self, index):
        idx = self.indices[index]
        
        # load image and label
        patch = np.load(self.data_path + 'images/' + str(idx) + '.npy').astype(np.float32)
        y = np.load(self.data_path + 'labels/' + str(idx) + '.npy')
        assert(patch.shape[0] == y.shape[0])
        assert(patch.shape[1] == y.shape[1]) 
        
        # crop
        if self.patch_size_padded < patch.shape[0]:
            patch, y = self.crop(patch,y)
       
        if self.augment:
            patch, y = self.data_augmentation(patch, y)
        
        patch_tensor = self.img_transform(patch)
        patch_tensor = torch.unsqueeze(patch_tensor, 0)

        patch_resized_list = []
        patch_resized_list.append(patch_tensor)
     
        # segmentation mask instead of one-hot-labels        
        y = np.argmax(y, axis=2)
        
        # for binary problem (grassland vs. no grassland)
        if self.n_classes == 2:
            y_full = y
            y = np.zeros((self.patch_size_padded, self.patch_size_padded), dtype=np.uint8)
            y[(y_full == 1) | (y_full == 2)] = 0
            y[(y_full == 3) | (y_full == 4)] = 1
        
        # segm transform, to torch long tensor HxW
        y = self.segm_transform(y)
        batch_segms = torch.unsqueeze(y, 0)

        output = dict()
        output['img_ori'] = patch[:,:,self.channels].astype(np.float64)
        output['img_data'] = [x.contiguous() for x in patch_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = str(idx)+'.npy'
        return output



    
    

    
    
    