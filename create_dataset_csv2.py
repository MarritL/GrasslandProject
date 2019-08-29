# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:06:03 2019

@author: lbergamasco

Create the dataset
"""

import numpy as np
import gc,os
from osgeo import gdal
from scripts.utils import list_dir, sample_patches_of_class

data_path = '/media/cordolo/FREECOM HDD/GrasslandProject/Data/'
output_path = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/'
output_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/patches.csv'

dirs = list_dir(data_path)
classes = [638,659,654,650,770]
dataset_size = 100*len(dirs) #Luca: 1000000
sample_division = len(dirs)
i_ds = 0
final_patch_size = 160 // 5 # 160 at 20cmx20xm
patch_size = final_patch_size*3
training_set=[]
ds_per_image = dataset_size//sample_division

div_classes = True

if not os.path.isdir(output_path):
    os.makedirs(output_path)

elapsed_time = 0

for i_lap, d in enumerate(dirs):
    #start = time.time()
    
    # ground truth
    path_SHP = data_path + d + '/tare.tif'
    gt = gdal.Open(path_SHP,gdal.GA_ReadOnly)
    gt = gdal.Warp('', [gt], format='MEM', width=gt.RasterXSize//5, height=gt.RasterYSize//5, resampleAlg=0)
    band = gt.GetRasterBand(1)
    gt = np.int16(band.ReadAsArray())
    del band
    
    #end = time.time()
    #elapsed_time = (elapsed_time + (end - start)) 
    #print('\ntime elapsed = ' + str(end - start))
    #print('avg time = ' + str(elapsed_time/ (i_lap+1)))
        
    
    tara0_mask = gt==638
    tara20_mask = gt==659
    tara50_mask = gt==654
    woods_mask = np.logical_or(gt==650,gt==656)
    no_coltivable_mask = np.logical_or(gt==770,gt==780)
    gt[woods_mask]=650
    gt[no_coltivable_mask]=770
    none_mask = gt==0
    classes_mask = np.logical_or(tara50_mask,np.logical_or(tara0_mask,tara20_mask))
    classes_mask = np.logical_or(no_coltivable_mask,np.logical_or(classes_mask,woods_mask))
    gt[np.logical_not(classes_mask)]=0
    rc_tara0 = np.argwhere(tara0_mask[0:-patch_size, 0:-patch_size])
    rc_tara20 = np.argwhere(tara20_mask[0:-patch_size, 0:-patch_size])
    rc_tara50 = np.argwhere(tara50_mask[0:-patch_size, 0:-patch_size])
    rc_woods = np.argwhere(woods_mask[0:-patch_size, 0:-patch_size])
    rc_no_coltivable = np.argwhere(no_coltivable_mask[0:-patch_size, 0:-patch_size])
    rc_UPAS = np.argwhere(gt[0:-patch_size, 0:-patch_size]!=0)
    
    if np.sum(tara0_mask)==0 and np.sum(tara20_mask)==0 and np.sum(tara50_mask)==0 and np.sum(woods_mask)==0 and np.sum(no_coltivable_mask)==0 :
        #print('continue to next folder')
        continue
    
    # percentage
    p_tara0 = len(rc_tara0)/len(rc_UPAS)
    p_tara20 = len(rc_tara20)/len(rc_UPAS)
    p_tara50 = len(rc_tara50)/len(rc_UPAS)
    p_woods = len(rc_woods)/len(rc_UPAS)
    p_no_coltivable = len(rc_no_coltivable)/len(rc_UPAS)
    
    # number of samples
    n_samples_tara0 = np.uint8(ds_per_image*p_tara0)
    n_samples_tara20 = np.uint8(ds_per_image*p_tara20)
    n_samples_tara50 = np.uint8(ds_per_image*p_tara50)
    n_samples_woods = np.uint8(ds_per_image*p_woods)
    n_samples_no_coltivable = np.uint8(ds_per_image*p_no_coltivable)

    # sample patches and write coordinate of origin to output csv-file
    sample_patches_of_class(rc_tara0, n_samples_tara0, classes[0], gt, patch_size, output_file,d)
    sample_patches_of_class(rc_tara20, n_samples_tara20, classes[1], gt, patch_size, output_file,d)
    sample_patches_of_class(rc_tara50, n_samples_tara50, classes[2], gt, patch_size, output_file,d)
    sample_patches_of_class(rc_woods, n_samples_woods, classes[3], gt, patch_size, output_file,d)
    sample_patches_of_class(rc_no_coltivable, n_samples_no_coltivable, classes[4], gt, patch_size, output_file,d)

    del gt
    gc.collect()