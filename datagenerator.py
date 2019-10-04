#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:18:33 2019

@author: cordolo
"""
import random
import numpy as np
import tensorflow.keras as keras
from dataset import to_categorical_classes
from scipy import ndimage
from skimage.segmentation import find_boundaries

class DataGen(keras.utils.Sequence):
    'Generates data for Keras'   
    
    def __init__(self, data_path, n_patches, shuffle, augment, indices, batch_size=128, patch_size=480, n_classes=5, channels=[0,1,2,3,4], max_size=480, pretrained_resnet50=False):
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
        self.pretrained_resnet50 = pretrained_resnet50
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

        # Generate data
        X, y = self.__data_generation(list_idx_temp)
        
# =============================================================================
#         # different preprocessing
#         if self.pretrained_resnet50:
#             X = self.__preprocess_imagenet(X)
# =============================================================================
            
        # group tara classes in one single class
        if self.n_classes == 3:
            y_full = y
            y = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.int8)
            for i in range(self.batch_size):
                gt_classes = np.argmax(y_full[i,], axis=2)
                gt_classes[gt_classes == 1] = 0
                gt_classes[gt_classes == 2] = 0
                y[i,] = to_categorical_classes(gt_classes, [0,3,4])
        
        # edge detection instead of classes
        if self.n_classes == 2:
            y_full = y
            y = np.zeros((self.batch_size, self.patch_size, self.patch_size, 2), dtype=np.int8)
            for i in range(self.batch_size):
                gt_classes = np.argmax(y_full[i,], axis=2)
                y[i,] = find_boundaries(gt_classes, mode='inner')

# =============================================================================
#                 edges = find_boundaries(gt_classes, mode='inner')
#                 y[i,] = to_categorical_classes(edges, [0,1])
# =============================================================================
        
        ######### TEST ############
        #y = y.reshape(self.patch_size*self.patch_size,self.n_classes)
        
# =============================================================================
#         sample_weights = np.zeros((self.patch_size*self.patch_size,self.n_classes))
#         sample_weights[:, 0] += 1
#         sample_weights[:, 1] += 1000
# =============================================================================
        
        return X,y #,sample_weights
    
    def __data_generation(self, list_idx_temp):
        'Generates data containing batch_size samples'
        
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
    
    def __preprocess_imagenet(self, X):
        'Preprocesses in same way as imagenet on pretrained ResNet50'
        
        # imagenet means and sds, for fourth and fifth channel, take avg of first 3
        means = [0.485, 0.456, 0.406, 0.449, 0.499]
        stds = [0.229, 0.224, 0.225, 0.226, 0.226]
        
        # normalize
        X *= 2  # because images are already normalized to values between 0-1
        X -= 1
        
        # Zero-center by mean pixel
        for i in range(self.n_channels):
            X[...,i] -= means[i]
            X[...,i] /= stds[i]
            
        return X              
  