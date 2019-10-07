#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:26:34 2019

@author: cordolo
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class GrasslandDataset(Dataset):
    
    def __init__(self, data_path, indices, channels):
        'Initialization'
        self.data_path = data_path
        self.indices= indices
        self.channels = channels
        self.n_channels = len(channels)
        
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.indices[index]
    
        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        patch = np.load(self.data_path + 'images/' + str(ID) + '.npy')
        
        if self.n_channels < 5:
            patch = patch[:,:,self.channels]
                
        x = torch.from_numpy(patch.permute(2,0,1))
        y = torch.from_numpy(np.load(self.data_path + 'labels/' + str(ID) + '.npy'))
    
        return x, y