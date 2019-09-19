#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:18:33 2019

@author: cordolo
"""
import random
import numpy as np
import tensorflow.keras as keras

class DataGen(keras.utils.Sequence):
    'Generates data for Keras'   
    
    def __init__(self, data_path, n_patches, shuffle, augment, indices, batch_size=128, patch_size=480, n_classes=5, channels=[0,1,2,3,4], max_size=480):
        'Initialization'
        self.data_path = data_path
        self.batch_size = batch_size        
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.channels = channels
        self.n_channels = len(channels)
        self.n_patches = n_patches
        self.shuffle = shuffle
        self.augment = augment
        self.indices = indices
        self.max_size = max_size
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_patches / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data' 
                
        # Generate indexes of the batch
        indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_idx_temp = [k for k in indexes]
        
        # print indexes to check
        for k in indexes:
            print(k)

        # Generate data
        X, y = self.__data_generation(list_idx_temp)
        
        return X,y
    
    def __data_generation(self, list_idx_temp):
        'Generates data containing batch_size samples'
        
        # test with smaller dataset
        #list_idx_temp = np.random.choice(self.indices, size= self.batch_size,replace=False)
        
        # Initialization
        X = np.zeros((self.batch_size, self.max_size, self.max_size, 5), dtype=np.float16) 
        y = np.zeros((self.batch_size, self.max_size, self.max_size, 5), dtype=np.int8)
        
        for i, idx in enumerate(list_idx_temp):
            
            #load patch
            X[i,] = np.load(self.data_path + 'images/' + str(idx) + '.npy')
            y[i,] = np.load(self.data_path + 'labels/' + str(idx) + '.npy')
            
            if self.augment:
                X[i,],y[i,] = self.data_augmentation(X[i,],y[i,])
        
        if self.patch_size < self.max_size:
            X = X[:,0:self.patch_size, 0:self.patch_size,]
            y = y[:,0:self.patch_size, 0:self.patch_size,]
        
        if self.n_channels < 5:
            X = X[:,:,:,self.channels]
            
        return X, y
    
    
    def on_epoch_end(self):
        'Shuffle indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            print("shuffle")
    
    def data_augmentation(self, X, y):
        'Data augmentation based on flips and rotations'
        
        fliplr = bool(random.getrandbits(1))
        flipud = bool(random.getrandbits(1))
        rot90 =random.randint(0,3)
                
        if fliplr:
            X = np.fliplr(X)
            y = np.fliplr(y)            
        if flipud:
            X = np.flipud(X)
            y = np.flipud(y)
        X = np.rot90(X,k=rot90)
        y = np.rot90(y,k=rot90)  
        
        return X,y