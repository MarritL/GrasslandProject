#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:26:34 2019

@author: cordolo
"""

import torch
from torch.utils.data import Dataset


class GrasslandDataset(Dataset):
    
    def __init__(self, data_path, indices):
        'Initialization'
        self.data_path = data_path
        self.indices= indices
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #ID = self.indices[index]
    
        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        x = torch.from_numpy(self.data_path + 'images/' + str(index) + '.npy')
        y = torch.from_numpy(self.data_path + 'labels/' + str(index) + '.npy')
    
        return x, y