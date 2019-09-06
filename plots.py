#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:05:50 2019

@author: cordolo
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import list_files
from matplotlib.colors import ListedColormap
import earthpy.plot as ep

# colormap
colors = ['linen', 'lightgreen', 'green', 'darkgreen', 'yellow']
cmap = ListedColormap(colors)

def plot_random_patches(patches_path, n_patches):
    """ plot random patches with ground truth
    
    arguments
    ---------
        patches_path: string
        
        n_patches: int
            number of random patches to plot
            
    output
    ------
        figure with n_patches plotted in first row and ground truth in second row.
    """
    
    patchespath = '/media/cordolo/FREECOM HDD/GrasslandProject/Patches/'
    images_path = patchespath + 'images/'
    labels_path = patchespath + 'labels/'
    
    rows = 2
    cols = n_patches    
    
    
    # get path images list
    im_list = list_files(images_path, '.npy')
    gt_list = list_files(labels_path, '.npy')
    
    # prepare
    index = np.array([0,1,2])
    fig, ax = plt.subplots(rows,cols)
    
    for i in range(n_patches):
        idx=np.random.randint(len(im_list))
        im = np.load(images_path + im_list[idx])
        gt = np.load(labels_path + gt_list[idx])
        
        # prepare RGB plot
        plt_im = im[:, :, index].astype(np.float64)
        
        # prepare gt plot
        plt_gt  = np.zeros_like(gt, dtype=np.uint8)
        plt_gt = np.argmax(gt, axis=2)
    
        # plot training image
        image = ax[0,i].imshow(plt_im)
        
        # plot gt 
        grtr = ax[1,i].imshow(plt_gt, cmap=cmap, vmin=0, vmax=4) #colors not right
    
    ep.draw_legend(grtr,titles=["tara0", "tara20", "tara50", "woods","no coltivable"],classes=[0, 1, 2, 3,4])

def plot_predicted_patches(predictions, groundtruth):
    """ plot predicted patches with ground truth
    
    arguments
    ---------
        predictions: numpy nd.array
            probability maps of classes of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
        groundtruth: numpy nd.array
            one-hot lables of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
    
    output
    ------
        figure with n predictions plotted in first row and ground truth in second row.
    """

    n_patches = len(predictions)
    rows = 2
    
    # prepare
    fig, ax = plt.subplots(rows,n_patches)
    
    for i in range(n_patches):
    
        im = predictions[i]
        gt = groundtruth[i]
        
        # prepare prediction plot
        plt_im  = np.zeros_like(im, dtype=np.uint8)
        plt_im = np.argmax(im, axis=2)
        
        # prepare gt plot
        plt_gt  = np.zeros_like(gt, dtype=np.uint8)
        plt_gt = np.argmax(gt, axis=2)
    
        # plot training image
        ax[0,i].imshow(plt_im, cmap=cmap, vmin=0, vmax=4)
        
        # plot gt 
        grtr = ax[1,i].imshow(plt_gt, cmap=cmap, vmin=0, vmax=4) 
    
    ep.draw_legend(grtr,titles=["tara0", "tara20", "tara50", "woods","no coltivable"],classes=[0, 1, 2, 3,4])
     
def plot_patches(patch, gt, n_patches):
    """ plot random patches with ground truth
    
    arguments
    ---------
        patches_path: string
        
        n_patches: int
            number of random patches to plot
            
    output
    ------
        figure with n_patches plotted in first row and ground truth in second row.
    """
    
    rows = 2
    cols = n_patches    
    
    # prepare
    index = np.array([0,1,2])
    fig, ax = plt.subplots(rows,cols)
    
    for i in range(n_patches):
        
        # prepare RGB plot
        plt_im = patch[:, :, index].astype(np.float64)
        
        # prepare gt plot
        plt_gt  = np.zeros_like(gt, dtype=np.uint8)
        plt_gt = np.argmax(gt, axis=2)
    
        # plot training image
        image = ax[0].imshow(plt_im)
        
        # plot gt 
        grtr = ax[1].imshow(plt_gt, cmap=cmap, vmin=0, vmax=4) #colors not right
    
    ep.draw_legend(grtr,titles=["tara0", "tara20", "tara50", "woods","no coltivable"],classes=[0, 1, 2, 3,4])

def plot_predicted_probabilities(predictions, groundtruth, n_classes):
    """ plot predicted patches with ground truth
    
    arguments
    ---------
        predictions: numpy nd.array
            probability maps of classes of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
        groundtruth: numpy nd.array
            one-hot lables of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
        nclasses: int
            number of prediction classes
    
    output
    ------
        figure with n predictions plotted in first row and ground truth in second row.
    """

    colors_extra = ['linen', 'lightgreen', 'green', 'darkgreen', 'yellow', 'black']
    cmap_extra = ListedColormap(colors_extra)
    
    n_patches = len(predictions)
    rows = 7
    
    # prepare
    fig, ax = plt.subplots(rows,n_patches)
    
    for i in range(n_patches):
    
        im = predictions[i]
        gt = groundtruth[i]
        
        # prepare prediction plot
        plt_im  = np.zeros_like(im, dtype=np.uint8)
        plt_im = np.argmax(im, axis=2)
        plt_im[np.max(im, axis=2)<0.5] = 5

        
        # plot probability maps
        for j in range(n_classes):
            ax[j,i].imshow(im[:,:,j], vmin=0, vmax=1)
        
        # prepare gt plot
        plt_gt  = np.zeros_like(gt, dtype=np.uint8)
        plt_gt = np.argmax(gt, axis=2)
        
        # plot prediction 
        im = ax[5,i].imshow(plt_im, cmap=cmap_extra, vmin=0, vmax=5) 
        
        # plot gt 
        grtr = ax[6,i].imshow(plt_gt, cmap=cmap, vmin=0, vmax=4) 
    
    ep.draw_legend(im,titles=["tara0", "tara20", "tara50", "woods","no coltivable","not sure"],classes=[0, 1, 2, 3,4,5])