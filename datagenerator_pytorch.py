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

    def __init__(self, data_path, indices, y_downsampling_rate=4, channels=[0,1,2], patch_size_padded=96, augment=False, shuffle=False):
        'Initialization'
        self.data_path = data_path
        self.indices= indices
        self.channels = channels
        self.n_channels = len(channels)
        self.patch_size_padded = patch_size_padded
        self.y_downsampling_rate = y_downsampling_rate
        self.augment = augment
        self.shuffle = shuffle
        
        self.num_sample = len(self.indices)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
           
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        idx = self.indices[index]
        
        item = dict()
        #item['fpath_img'] = self.data_path + 'images/' + str(idx) + '.npy'
        #item['fpath_segm'] = self.data_path + 'labels/' + str(idx) +'.npy'
        item['width'] = self.patch_size_padded
        item['height'] = self.patch_size_padded
    
        # Load data and get label
        patch = np.load(self.data_path + 'images/' + str(idx) + '.npy').astype(np.float32)
        y = torch.from_numpy(np.load(self.data_path + 'labels/' + str(idx) + '.npy'))
        assert(patch.shape[0] == y.shape[0])
        assert(patch.shape[1] == y.shape[1])
        
        # data augmentation
        if self.augment:
            patch, y = self.data_augmentation(patch, y)
            
        # image transform, to torch float tensor 3xHxW
        patch = self.img_transform(patch)
            
        # downsample label mask    
        y = np.argmax(y, axis=2)
        y = Image.fromarray(y.astype(np.uint8))
        y = imresize(
            y,
            (y.size[0] // self.y_downsampling_rate, \
            y.size[1] // self.y_downsampling_rate), \
            interp='nearest')
            
        # segm transform, to torch long tensor HxW
        y = self.segm_transform(y)
        
        return patch, y


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
        y = np.array(y)      
        y = torch.from_numpy(y).long()
        return y

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p