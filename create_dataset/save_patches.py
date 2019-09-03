#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:00:36 2019

@author: MLeenstra
"""
import numpy as np
from osgeo import gdal
from utils import read_patch, to_categorical_classes,list_dir
import pandas as pd
import os

data_path = '/media/cordolo/elements/Data/'
dtm_path = '/media/cordolo/FREECOM HDD/GrasslandProject/DTM/'
coords_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/patches.csv'
save_path = '/media/cordolo/FREECOM HDD/GrasslandProject/Patches/'
images_path = save_path + 'images/'
labels_path = save_path + 'labels/'

if not os.path.isdir(images_path):
    os.makedirs(images_path)    
if not os.path.isdir(labels_path):
    os.makedirs(labels_path)
    
dirs = list_dir(data_path)
coords = pd.read_csv(coords_file, sep=',',header=None)
final_patch_size = 160
patch_size = int(final_patch_size * 3)
classes = [638,659,654,650,770]

# resample dtm to 20cmx20xm
for d in dirs:    
    if not os.path.isdir(dtm_path + d + '/'):
        os.makedirs(dtm_path + d + '/')   
    input_file = data_path + d + '/dtm135.tif'
    shadow_file = data_path + d + '/' + d + '_NIR.tif'
    dtm_file = dtm_path + d + '/dtm135_20cm.tif'
    
    ds = gdal.Open(shadow_file)
    width = ds.RasterXSize
    height = ds.RasterYSize 
    ds = gdal.Warp(dtm_file, input_file, format='GTiff', width=width, height=height, resampleAlg=1)
    ds = None

# extract patches
for idx in range(len(coords)):
    im, gt = read_patch(data_path, dtm_path, coords, patch_size, idx, classes)
    np.save(images_path + str(idx)+'.npy', im)
    np.save(labels_path+str(idx) + '.npy', gt)
    if idx % 500 == 0: 
        print('\r {}/{}'.format(idx, len(coords)),end='')

