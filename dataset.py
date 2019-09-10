#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:29:39 2019

@author: MLeenstra
"""

import os
import random
import csv
import gc
import numpy as np
import pandas as pd
from osgeo import gdal
from utils import list_dir

def tile_to_patch(inputpath, coordspath, coordsfilename, dtmpath, patchespath,
                  patch_size, patches_per_tile, classes, resolution):
    """ extract patches from tile.
    
    parameters
    ----------
        inputpath: string
            path to folders with tiles. Each tile should be in separate folder  
        coordspath: string
            path to outputfolder to save file with coordinates
        coordsfilename: string
            output filename, extention should be '.csv'
        dtmpath: string 
            path to folder containing a dtm (will be resampled to same resolution)
        patchespath: string 
            path to outputfolder to save the patches
        patch_size: int
            size of patch to extract. Final extracted patches will include padding 
            to be able to predict full image.
        patches_per_tile: int
            number of patches to extract per tile. Final number can be lower if the 
            classes cover very few pixels
        classes: list
            list with classes to be predicted
        resolution: int
            either 1 for 1m or 20 for 20cm
    
    
    Calls
    -----
        tile_to_csv()
        csv_to_patch()
        
    Output
    ------
        patches saved at patchespath in two folders: 
            images and labels.
    
    """
    # extract coordinates for patches and save in csv    
    tile_to_csv(inputpath=inputpath, coordspath=coordspath,
                    coordsfilename=coordsfilename, patch_size=patch_size, patches_per_tile = patches_per_tile, classes = classes)
    
    print("\n")
    
    # extract patches based on coordinates and save
    csv_to_patch(inputpath = inputpath, dtmpath = dtmpath,
                    coordsfile = coordspath + coordsfilename, 
                    patchespath = patchespath,
                    patch_size=patch_size, classes = classes, resolution = resolution)
    
        
def tile_to_csv(inputpath, coordspath, coordsfilename, patches_per_tile, patch_size, classes):
    """ save location of patches in csv-file.
    
    Locations of top-left corner-pixel of patches are saved. These pixels
    are chosen at random, however the percentual class division is respected.
    
    parameters
    ----------
        inputpath: string
            path to folders with tiles. Each tile should be in separate folder  
        coordspath: string
            path to outputfolder to save file with coordinates
        coordsfilename: string
            output filename, extention should be '.csv'
        patches_per_tile: int
            number of patches to extract per tile. Final number can be lower if the 
            classes cover very few pixels
        patch_size: int
            size of patch to extract. Final extracted patches will include padding 
            to be able to predict full image.
        classes: list
            list with classes to be predicted
    
    calls
    -----
        sample_patches_of_class()
    
    output
    -------
        outputfile: csv
            each row contains tile-name and row + column of top-left pixel of patch: 
            tile,row,column
            saved at outputpath.
    """
    
    # init
    dirs = list_dir(inputpath)   
    patch_size = patch_size // 5 # because downsample from 20cmX20cm to 1mx1m 
    patch_size_padded = int(patch_size * 3) 
    
    if not os.path.isdir(coordspath):
        os.makedirs(coordspath)  
    
    for i_lap, d in enumerate(dirs):
    
        # ground truth
        path_SHP = inputpath + d + '/tare.tif'
        gt = gdal.Open(path_SHP,gdal.GA_ReadOnly)
        # resample to 1m resolution
        gt = gdal.Warp('', [gt], format='MEM', width=gt.RasterXSize//5, height=gt.RasterYSize//5, resampleAlg=0) 
        band = gt.GetRasterBand(1)
        gt = np.int16(band.ReadAsArray())
        del band        
        
        # take care of classes
        tara0_mask = gt==classes[0]
        tara20_mask = gt==classes[1]
        tara50_mask = gt==classes[2]
        woods_mask = np.logical_or(gt==classes[3],gt==656)
        no_coltivable_mask = np.logical_or(gt==classes[4],gt==780)
        gt[woods_mask]=classes[3]
        gt[no_coltivable_mask]=classes[4]
        classes_mask = np.logical_or(tara50_mask,np.logical_or(tara0_mask,tara20_mask))
        classes_mask = np.logical_or(no_coltivable_mask,np.logical_or(classes_mask,woods_mask))
        gt[np.logical_not(classes_mask)]=0
        rc_tara0 = np.argwhere(tara0_mask[0:-patch_size_padded, 0:-patch_size_padded])
        rc_tara20 = np.argwhere(tara20_mask[0:-patch_size_padded, 0:-patch_size_padded])
        rc_tara50 = np.argwhere(tara50_mask[0:-patch_size_padded, 0:-patch_size_padded])
        rc_woods = np.argwhere(woods_mask[0:-patch_size_padded, 0:-patch_size_padded])
        rc_no_coltivable = np.argwhere(no_coltivable_mask[0:-patch_size_padded, 0:-patch_size_padded])
        rc_UPAS = np.argwhere(gt[0:-patch_size_padded, 0:-patch_size_padded]!=0)
        
        if np.sum(tara0_mask)==0 and np.sum(tara20_mask)==0 and np.sum(tara50_mask)==0 and np.sum(woods_mask)==0 and np.sum(no_coltivable_mask)==0 :
            continue
    
        # sample patches and write coordinate of origin to output csv-file
        sample_patches_of_class(rc_tara0, rc_UPAS, patches_per_tile, classes[0], gt, patch_size_padded, coordspath+coordsfilename,d)
        sample_patches_of_class(rc_tara20, rc_UPAS, patches_per_tile, classes[1], gt, patch_size_padded, coordspath+coordsfilename,d)
        sample_patches_of_class(rc_tara50, rc_UPAS, patches_per_tile, classes[2], gt, patch_size_padded, coordspath+coordsfilename,d)
        sample_patches_of_class(rc_woods, rc_UPAS, patches_per_tile, classes[3], gt, patch_size_padded, coordspath+coordsfilename,d)
        sample_patches_of_class(rc_no_coltivable, rc_UPAS, patches_per_tile, classes[4], gt, patch_size_padded, coordspath+coordsfilename,d)
    
        del gt
        gc.collect()
        if i_lap+1 % 10 == 0: 
            print('\r {}/{}'.format(i_lap+1, len(dirs)),end='')

# =============================================================================
# def csv_to_patch(inputpath, dtmpath, patchespath, coordsfile, patch_size, classes):
#     """ extract the patches from the original images, normalize and save.
#     
#     Ground truth is converted to one-hot labels. 
#     
#     Parameters
#     ----------
#         inputpath: string
#             path to folders with tiles. Each tile should be in separate folder
#         dtmpath: string 
#             path to folder containing a dtm (will be resampled to same resolution)
#         patchespath: string 
#             path to outputfolder to save the patches
#         coordsfile: csv-file
#             path to file where the coordinates are saved
#         patch_size: int
#             size of patch to extract. Final extracted patches will include padding 
#             to be able to predict full image.
#         classes: list
#             list with classes to be predicted
#     
#     Calls
#     -----
#         read_patch() 
#         to_categorical_classes()
#     
#     Output
#     ------
#         patches saved at patchespath in two folders: 
#             images and labels.
#     """
#     imagespath = patchespath + 'images/'
#     labelspath = patchespath + 'labels/'
#     
#     if not os.path.isdir(imagespath):
#         os.makedirs(imagespath)    
#     if not os.path.isdir(labelspath):
#         os.makedirs(labelspath)
#         
#     dirs = list_dir(inputpath)
#     coords = pd.read_csv(coordsfile, sep=',',header=None)
#     patch_size_padded = int(patch_size * 3)
#     
#     # resample dtm to 20cmx20xm
#     for d in dirs:    
#         if not os.path.isdir(dtmpath + d + '/'):
#             os.makedirs(dtmpath + d + '/')   
#         input_file = inputpath + d + '/dtm135.tif'
#         shadow_file = inputpath + d + '/' + d + '_NIR.tif'
#         dtm_file = dtmpath + d + '/dtm135_20cm.tif'
#         
#         ds = gdal.Open(shadow_file)
#         width = ds.RasterXSize
#         height = ds.RasterYSize 
#         ds = gdal.Warp(dtm_file, input_file, format='GTiff', width=width, height=height, resampleAlg=1)
#         ds = None
#     
#     # extract patches
#     for idx in range(len(coords)):
#         im, gt = read_patch(inputpath, dtmpath, coords, patch_size_padded, idx, classes)
#         np.save(imagespath + str(idx)+'.npy', im)
#         np.save(labelspath+str(idx) + '.npy', gt)
#         if idx % 500 == 0: 
#             print('\r {}/{}'.format(idx, len(coords)),end='')
# =============================================================================
            
def csv_to_patch(inputpath, dtmpath, patchespath, coordsfile, patch_size, classes, resolution):
    """ extract the patches from the original images, normalize and save.
    
    Ground truth is converted to one-hot labels. 
    
    Parameters
    ----------
        inputpath: string
            path to folders with tiles. Each tile should be in separate folder
        dtmpath: string 
            path to folder containing a dtm (will be resampled to same resolution)
        patchespath: string 
            path to outputfolder to save the patches
        coordsfile: csv-file
            path to file where the coordinates are saved
        patch_size: int
            size of patch to extract. Final extracted patches will include padding 
            to be able to predict full image.
        classes: list
            list with classes to be predicted
        resolution: int
            either 20 for 20cm or 1 for 1m
            
    
    Calls
    -----
        read_patch() 
        to_categorical_classes()
    
    Output
    ------
        patches saved at patchespath in two folders: 
            images and labels.
    """
    imagespath = patchespath + 'images/'
    labelspath = patchespath + 'labels/'
    
    if not os.path.isdir(imagespath):
        os.makedirs(imagespath)    
    if not os.path.isdir(labelspath):
        os.makedirs(labelspath)
        
    dirs = list_dir(inputpath)
    coords = pd.read_csv(coordsfile, sep=',',header=None)
    patch_size_padded = int(patch_size * 3)
    
    if resolution == 20:
        # resample dtm to 20cmx20xm
        for d in dirs:    
            if not os.path.isdir(dtmpath + d + '/'):
                os.makedirs(dtmpath + d + '/')   
            input_file = inputpath + d + '/dtm135.tif'
            shadow_file = inputpath + d + '/' + d + '_NIR.tif'
            dtm_file = dtmpath + d + '/dtm135_20cm.tif'
            
            ds = gdal.Open(shadow_file)
            width = ds.RasterXSize
            height = ds.RasterYSize 
            ds = gdal.Warp(dtm_file, input_file, format='GTiff', width=width, height=height, resampleAlg=1)
            ds = None
        
        # extract patches
        for idx in range(len(coords)):
            im, gt = read_patch(inputpath, dtmpath, coords, patch_size_padded, idx, classes)
            np.save(imagespath + str(idx)+'.npy', im)
            np.save(labelspath+str(idx) + '.npy', gt)
            if idx % 500 == 0: 
                print('\r {}/{}'.format(idx, len(coords)),end='')
    
    else:
        warpedtile = None
        #for idx, tile in enumerate(coords[0]): 
        for idx, d in enumerate(coords[0][78:85]):    
            # resample rgb + nir to 1m (keep in memory (no space on disk))
            if not d == warpedtile:
                print(d)
                shadow_file = inputpath + d + '/dtm135.tif'
                nir_file = inputpath + d + '/' + d + '_NIR.tif'
                rgb_file = inputpath + d + '/' + d + '_RGB.tif'
                gt_file = inputpath + d + '/tare.tif'
                
                ds = gdal.Open(shadow_file)
                width = ds.RasterXSize
                height = ds.RasterYSize 
                nir_1m = gdal.Warp("", nir_file, format='mem', width=width, height=height, resampleAlg=1)
                rgb_1m = gdal.Warp("", rgb_file, format='mem', width=width, height=height, resampleAlg=1)
                gt_1m = gdal.Warp("", gt_file, format='mem', width=width, height=height, resampleAlg=0)
                warpedtile = d
                ds = None       
            
            # extract patches
            im, gt = read_patch_1m(rgb_1m, nir_1m, dtmpath, gt_1m, coords, patch_size_padded, idx, classes)
            np.save(imagespath + str(idx)+'.npy', im)
            np.save(labelspath + str(idx) + '.npy', gt)
            if idx % 500 == 0: 
                print('\r {}/{}'.format(idx, len(coords)),end='')            
        

    

def sample_patches_of_class(population, population_all, patches_per_tile, cl, gt, patch_size_padded, outputfile, d):
    """ sample origin pixels (row+col) of patches and save in csv file. 
    
    Parameters
    ----------
        population: numpy.ndarray 
            pixels containing class cl
        population_all: numpy.ndarray
            pixels containing all classes (except null-class)
        patches_per_tile: int
            number of samples to extract for whole tile
        cl: int or string
            ground truth class (e.g. classes[4])
        gt: numpy.ndarray
            ground truth image 
        patch_size_padded: int
            size of patch (including padding) to be extracted in number of pixels 
            (patch will be squared)
        outputfile: string
            path + filename + extention to save the coordiantes
        
    output
    -------
        outputfile: csv
            each row contains tile-name and row + column of top-left pixel of patch: 
            tile,row,column
            saved at outputpath.
    """
    # percentage
    p_class = len(population)/len(population_all)
        
    # number of samples
    n_samples = np.uint8(patches_per_tile*p_class)

    n_loops = 0
        
    with open(outputfile, 'a') as file:
        writer = csv.writer(file, delimiter =',')     
        while n_samples > 0 and len(population) > n_samples:
            idx = random.sample(range(len(population)),n_samples)
            gt_idx = population[idx]
            population = np.delete(population,idx,axis=0)
            for r,c in gt_idx:
                patch = gt[r:(r+patch_size_padded),c:(c+patch_size_padded)]
                mask_past = np.uint16(patch==cl)
                mask_el = np.uint16(patch==0) 
                if (np.sum(mask_past)>=1000 and np.sum(mask_el)==0 and patch.shape[0]==patch_size_padded and patch.shape[1]==patch_size_padded) or n_loops>=1000:
                    writer.writerow([d, r, c])
                    n_samples -= 1
                    n_loops =0
                    continue
                n_loops += 1
                if n_loops==100:
                    n_samples==0
                    break

def read_patch_1m(inputRGB, inputNIR, dtmpath, inputGT, coords_df, patch_size_padded, idx, classes):
    """ load patch based on left-top coordinate
    
    Parameters
    ----------
        inputpath: string
            path to the folder containing dataset
        dtmpath: string
            path to folder containing the resampled dtm 
        coords_df: pandas.dataframe
            dataframe with coordinates in the format [tile, r (=left), c (=top)]
        patch_size_padded: int
            size of patch (including padding) to be extracted in number of pixels 
            (patch will be squared)
        idx: int
            index of row in coords file to be read
        classes: list
            list with classes to be predicted
          
    Calls
    -----
        to_categorical_classes

    Returns
    ------
        patch: numpy.ndarray 
            patch of size (patch_size, patch_size, 5) with normalized RGB, NIR, DTM 
        gt: numpy.ndarray 
            patch of size (patch_size, patch_size, n_classes) with one-hot encoded ground truth
    """

    n_features = 5 #R,G,B, NIR, DTM
    
    folder, r, c = coords_df.iloc[idx]
    r = int(r)*5
    c = int(c)*5

    patch = np.zeros([patch_size_padded,patch_size_padded,n_features],dtype=np.float16)
    
    # RGB
    ds = inputRGB  
    # needed for resampling of dtm 
    for x in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(x)
        patch[:,:,x-1] = band.ReadAsArray(c,r,patch_size_padded, patch_size_padded)

    # NIR
    ds = inputNIR
    band = ds.GetRasterBand(1)
    patch[:,:,3] = band.ReadAsArray(c, r, patch_size_padded, patch_size_padded)

    # DTM
    ds = gdal.Open(dtmpath + folder + '/dtm135.tif' ,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    patch[:,:,4] = band.ReadAsArray(c, r, patch_size_padded, patch_size_padded)

    #normalization
    patch = np.divide(patch,255)
    
    # load ground truth
    ds = inputGT
    band = ds.GetRasterBand(1)
    gt = band.ReadAsArray(c, r, patch_size_padded, patch_size_padded)
    
    # take care of classes
    gt[np.where(gt == 656)] = 650
    gt[np.where(gt == 780)] = 770
    gt[np.isin(gt, classes)==False] = 0
    
    gt = to_categorical_classes(gt, classes)
    
    return((patch,gt))
            
def read_patch(inputpath, dtmpath, coords_df, patch_size_padded, idx, classes):
    """ load patch based on left-top coordinate
    
    Parameters
    ----------
        inputpath: string
            path to the folder containing dataset
        dtmpath: string
            path to folder containing the resampled dtm 
        coords_df: pandas.dataframe
            dataframe with coordinates in the format [tile, r (=left), c (=top)]
        patch_size_padded: int
            size of patch (including padding) to be extracted in number of pixels 
            (patch will be squared)
        idx: int
            index of row in coords file to be read
        classes: list
            list with classes to be predicted
          
    Calls
    -----
        to_categorical_classes

    Returns
    ------
        patch: numpy.ndarray 
            patch of size (patch_size, patch_size, 5) with normalized RGB, NIR, DTM 
        gt: numpy.ndarray 
            patch of size (patch_size, patch_size, n_classes) with one-hot encoded ground truth
    """

    n_features = 5 #R,G,B, NIR, DTM
    
    folder, r, c = coords_df.iloc[idx]
    r = int(r)*5
    c = int(c)*5

    patch = np.zeros([patch_size_padded,patch_size_padded,n_features],dtype=np.float16)
    
    # RGB
    ds = gdal.Open(inputpath + folder + '/' + folder + '_RGB.tif',gdal.GA_ReadOnly)    
    # needed for resampling of dtm 
    for x in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(x)
        patch[:,:,x-1] = band.ReadAsArray(c,r,patch_size_padded, patch_size_padded)

    # NIR
    ds = gdal.Open(inputpath + folder + '/' + folder + '_NIR.tif' ,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    patch[:,:,3] = band.ReadAsArray(c, r, patch_size_padded, patch_size_padded)

    # DTM
    ds = gdal.Open(dtmpath + folder + '/dtm135_20cm.tif' ,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    patch[:,:,4] = band.ReadAsArray(c, r, patch_size_padded, patch_size_padded)

    #normalization
    patch = np.divide(patch,255)
    
    # load ground truth
    ds = gdal.Open(inputpath + folder + '/tare.tif' ,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    gt = band.ReadAsArray(c, r, patch_size_padded, patch_size_padded)
    
    # take care of classes
    gt[np.where(gt == 656)] = 650
    gt[np.where(gt == 780)] = 770
    gt[np.isin(gt, classes)==False] = 0
    
    gt = to_categorical_classes(gt, classes)
    
    return((patch,gt))

def to_categorical_classes(y, classes, dtype=np.int8):
  """Converts a class vector to binary class matrix.

  Parameters
  ---------
      y: numpy.ndarray
          class array to be converted into one-hot-encoding. 
          (if classes are specified by integers from 0 to num_classes use 
          keras.utils.to_categorical instead).
      dtype: np.dtype 
          The data type expected by the input. Default: `'int8'`.
      classes: list
          list with classes

  Returns
  -------
      A binary matrix representation of the input. The classes axis is placed
      last.
  """

  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  n_classes = len(classes)
  n = y.shape[0]
  categorical = np.zeros((n, n_classes), dtype=dtype)
  for i, cl in enumerate(classes):
    y[np.where(y == cl)]=i
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (n_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical

def train_test_split(coordsfile, tiles_cv_file, tiles_test_file, n_test_tiles):
    """
    
    arguments
    ---------
        coordsfile: string
            path to file where the coordinates are saved
        tiles_cv_file: string
            path to file where the tilenames for cross-validation are saved
        tiles_test_file: string
            path to file where the tilenames for testing are saved
    
    """
   
    coords_df = pd.read_csv(coordsfile, sep=',',header=None, names=['tiles', 'row', 'col'])
    coords_df = coords_df['tiles']
    
    # train / test split
    tiles = np.unique(coords_df)
    testtiles = np.random.choice(tiles, n_test_tiles, replace=False)
    np.save(tiles_test_file, testtiles)

    # remove testtiles
    tiles = tiles[np.isin(tiles, testtiles) == False]
    
    # shuffle and save train-part
    random.shuffle(tiles)
    tiles_shuffled = tiles
    np.save(tiles_cv_file, tiles_shuffled)

def train_val_split(tiles_cv_file, coordsfile, folds, k):    
    """ select indices of train and validation patches in coordfile
    
    arguments
    ---------
        tiles_cv_file: string
            path to file where the tilenames for cross-validation are saved
        coordsfile: string
            path to file where the coordinates are saved
        folds: int
            number of folds
        k: int
            fold to return, first fold is 0.
            
    return
    ------
        index_train: numpy ndarray
            indices of train patches in coordsfile
        index_val: numpy ndarray
            indices of validation patches in coordsfile
     
    """
    coords_df = pd.read_csv(coordsfile, sep=',',header=None, names=['tiles', 'row', 'col'])
    coords_df = coords_df['tiles']
    
    tiles = np.load(tiles_cv_file, allow_pickle=True)
    tiles_per_fold = int(len(tiles)/folds)
    
    # train / val split  
    valtiles = tiles[k*tiles_per_fold:(k+1)*tiles_per_fold]
    traintiles = tiles[np.isin(tiles, valtiles) == False]
    
    # find indices
    coords_val = coords_df[np.isin(coords_df, valtiles)]
    coords_train = coords_df[np.isin(coords_df, traintiles)]
    
    index_val = np.array(coords_val.index)
    index_train = np.array(coords_train.index)
    
    return(index_train, index_val)
    
def load_test_indices(tiles_test_file, coordsfile):
    """ select indices of train and validation patches in coordfile
    
    arguments
    ---------
        tiles_test_file: string
            path to file where the tilenames for testing are saved
        coordsfile: string
            path to file where the coordinates are saved
            
    return
    ------
        index_test: numpy ndarray
            indices of test patches in coordsfile     
    """
    coords_df = pd.read_csv(coordsfile, sep=',',header=None, names=['tiles', 'row', 'col'])
    coords_df = coords_df['tiles']
    
    testtiles = np.load(tiles_test_file, allow_pickle=True)
    
    # find indices
    coords_test = coords_df[np.isin(coords_df, testtiles)]
    
    index_test = np.array(coords_test.index)
    
    return(index_test)
    
def get_patches(patchespath, indices, patch_size_padded, channels):
    """ load patches
    
    arguments
    ---------
        patchespatch: string
            path to folder containing the patches
        indices: list
            indices of patches to load
        patch_size_patddes: ubt
            size of patch to extract, including padding.
        channels: list
            list with classes to be predicted
            
    return
    ------
        X: numpy.ndarray
            patches stored in array of size (n_patches, patch_size, patch_size, n_channels)
        y: numpy.ndarray
            ground truth stored in array of size (n_patches, patch_size, patch_size, n_classes)    
    """
    
    # Initialization
    n_channels = len(channels)
    n_patches = len(indices)
    X = np.zeros((n_patches, 480, 480, 5), dtype=np.float16) 
    y = np.zeros((n_patches, 480, 480, 5), dtype=np.int8)
    
    for i, idx in enumerate(indices):
        #load patch
        X[i,] = np.load(patchespath + 'images/' + str(idx) + '.npy')
        y[i,] = np.load(patchespath + 'labels/' + str(idx) + '.npy')
        
    # crop patch 
    if patch_size_padded < 480:
        X = X[:,0:patch_size_padded, 0:patch_size_padded,]
        y = y[:,0:patch_size_padded, 0:patch_size_padded,]
    
    # drop channels 
    if n_channels < 5:
        X = X[:,:,:,channels]
        
    return X, y
    