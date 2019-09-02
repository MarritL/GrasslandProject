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
%matplotlib gt

save_path = '/media/cordolo/FREECOM HDD/GrasslandProject/Patches/'
images_path = save_path + 'images/'
labels_path = save_path + 'labels/'


#### plot    
num = 6 # number of plots
rows = num/6
cols = 6    
class_bins = [0,1,2,3,4]
colors = ['linen', 'lightgreen', 'green', 'darkgreen', 'yellow']
cmap = ListedColormap(colors)
#norm = BoundaryNorm(class_bins, len(colors))

# get path images list
im_list = list_files(images_path, '.npy')
gt_list = list_files(labels_path, '.npy')
#shuffle(trn_im_list)

# prepare
index = np.array([0,1,2])
fig, ax = plt.subplots(int(rows*2),cols)

for i in range(num):
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
    
    # plot training image
    grtr = ax[1,i].imshow(plt_gt, cmap=cmap, vmin=0, vmax=4) #colors not right
ep.draw_legend(grtr,titles=["tara0", "tara20", "tara50", "woods","no coltivable"],classes=[0, 1, 2, 3,4])

 
     
 
