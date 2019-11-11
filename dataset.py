#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:29:39 2019

@author: MLeenstra
"""

import os
import random
import numpy as np
import pandas as pd
from osgeo import gdal
from utils import list_dir
from plots import plot_patch_options, barplot_classes

def tile_to_csv_grid(inputpath, coordspath, coordsfilename, patch_size_padded):
    """ save location of patches in csv-file.
    
    Locations of top-left corner-pixel of patches are saved in csv. These corner
    pixels are based on a grid overlaying the original tiles.
    
    parameters
    ----------
        inputpath: string
            path to folders with tiles. Each tile should be in separate folder  
        coordspath: string
            path to outputfolder to save file with coordinates
        coordsfilename: string
            output filename, extention should be '.csv'
        patch_size_padded: int
            size of patch to extract including padding to be able to predict 
            full image.
    
    calls
    -----
        find_patches_options()
    
    output
    -------
        outputfile: csv
            each row contains row + column of top-left pixel of patch and tilename: 
            row,column,tile
            saved at coordspath with name coordsfilename
    """
    
    # init
    dirs = list_dir(inputpath) 
    
    if not os.path.isdir(coordspath):
        os.makedirs(coordspath)  
    
    for i_lap, d in enumerate(dirs):
        
        # get tile
        path_shp = inputpath + d + '/tare.tif'
        ds = gdal.Open(path_shp,gdal.GA_ReadOnly)
        gt = ds.GetRasterBand(1).ReadAsArray()
        gt = np.uint16(gt)
        ds = None
        
        # get options
        if i_lap == 0:
            options = find_patches_options(gt, patch_size_padded, d)
        else:
            options = options.append(find_patches_options(gt, patch_size_padded, d))
            
        if i_lap % 50 == 0: 
                print('\r {}/{}'.format(i_lap, len(dirs)),end='')
    
    # save in csv 
    options['row'] = options['row'].multiply(patch_size_padded)
    options['col'] = options['col'].multiply(patch_size_padded)
    options.to_csv(coordspath+coordsfilename, index=False)

           
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
        read_patch_1m()
        to_categorical_classes()
    
    Output
    ------
        patches saved at patchespath in two folders: 
            images and labels.
    """
    imagespath = patchespath + 'images/'
    labelspath = patchespath + 'labels/'
    
    if not os.path.isdir(patchespath):
        os.makedirs(patchespath) 
    if not os.path.isdir(imagespath):
        os.makedirs(imagespath)    
    if not os.path.isdir(labelspath):
        os.makedirs(labelspath)
        
    dirs = list_dir(inputpath)
    coords = pd.read_csv(coordsfile, sep=',')
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
    
    elif resolution == 1:
        warpedtile = None
        #for idx, tile in enumerate(coords[0]): 
        for idx, d in enumerate(coords['tiles']):    
            # resample rgb + nir to 1m (keep in memory (no space on disk))
            if not d == warpedtile:
                dtm_file = inputpath + d + '/dtm135.tif'
                nir_file = inputpath + d + '/' + d + '_NIR.tif'
                rgb_file = inputpath + d + '/' + d + '_RGB.tif'
                gt_file = inputpath + d + '/tare.tif'
                
                ds = gdal.Open(dtm_file)
                width = ds.RasterXSize
                height = ds.RasterYSize 
                nir_1m = gdal.Warp("", nir_file, format='mem', width=width, height=height, resampleAlg=1)
                rgb_1m = gdal.Warp("", rgb_file, format='mem', width=width, height=height, resampleAlg=1)
                gt_1m = gdal.Warp("", gt_file, format='mem', width=width, height=height, resampleAlg=0)
                warpedtile = d
                ds = None       
            
            # extract patches
            im, gt = read_patch_1m(rgb_1m, nir_1m, dtm_file, gt_1m, coords, patch_size_padded, idx, classes)     
            np.save(imagespath + str(idx)+'.npy', im)
            np.save(labelspath + str(idx) + '.npy', gt)
            if idx % 500 == 0: 
                print('\r {}/{}'.format(idx, len(coords)),end='')            
    else:
        print("Only resoltions of 20cmx20xm (20) and 1mx1m (1) are supported.")

    

def read_patch_1m(inputRGB, inputNIR, inputDTMfile, inputGT, coords_df, patch_size_padded, idx, classes):
    """ load patch based on left-top coordinate for 1mx1m resolution 
    
    Parameters
    ----------
        inputRGB: gdal raster
            RGB tile in gdal format
        inputNIR: gdal raster
            NIR tile in gdal format
        inputDTMfile: string
            path to DTM tiff file
        inputGT: gdal raster
            Ground Truth tile in gdal format
        coords_df: pandas.dataframe
            dataframe with coordinates in the format [r (=left), c (=top), tile]
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
    
    r, c, folder = coords_df.iloc[idx]
    r = int(r//5)
    c = int(c//5)

    patch = np.zeros([patch_size_padded,patch_size_padded,n_features],dtype=np.float16)
    
    # RGB
    ds = inputRGB  
    # needed for resampling of dtm 
    for x in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(x)
        patch[:,:,x-1] = band.ReadAsArray(c,r,int(patch_size_padded), int(patch_size_padded))

    # NIR
    ds = inputNIR
    band = ds.GetRasterBand(1)
    patch[:,:,3] = band.ReadAsArray(c, r, patch_size_padded, patch_size_padded)

    # DTM
    ds = gdal.Open(inputDTMfile, gdal.GA_ReadOnly)
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
    """ load patch based on left-top coordinate for 20cmx20cm resolution 
    
    Parameters
    ----------
        inputpath: string
            path to the folder containing dataset
        dtmpath: string
            path to folder containing the resampled dtm 
        coords_df: pandas.dataframe
            dataframe with coordinates in the format [r (=left), c (=top), tile]
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
    
    r, c, folder = coords_df.iloc[idx]
    r = int(r)
    c = int(c)

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
    """ split dataset in a train and test part based on the original tiles
    
    arguments
    ---------
        coordsfile: string
            path to file where the coordinates are saved
        tiles_cv_file: string
            path to file where the tilenames for cross-validation are saved
        tiles_test_file: string
            path to file where the tilenames for testing are saved
        n_test_tiles: int
            number of test tiles to extract from the dataset
    
    saves
    -------
        2 csv files:
            1 for training and validation (tiles_cv_file)
            1 for testing (tiles_test_file)
    
    """
   
    coords_df = pd.read_csv(coordsfile, sep=',')
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
    """ select indices of train and validation patches in coordfile based on fold
    
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
    coords_df = pd.read_csv(coordsfile, sep=',')
    coords_df = coords_df[['tiles','Unnamed: 0']]
    
    tiles = np.load(tiles_cv_file, allow_pickle=True)
    
    tiles_per_fold = int(len(tiles)/folds)
    
    # train / val split  
    valtiles = tiles[k*tiles_per_fold:(k+1)*tiles_per_fold]
    traintiles = tiles[np.isin(tiles, valtiles) == False]
    
    # find indices
    coords_val = coords_df[np.isin(coords_df['tiles'], valtiles)]
    coords_train = coords_df[np.isin(coords_df['tiles'], traintiles)]
    
    index_val = np.array(coords_val['Unnamed: 0'])
    index_train = np.array(coords_train['Unnamed: 0'])
    
    random.shuffle(index_val)
    random.shuffle(index_train)
    
    return(index_train,index_val)

def train_val_split_random(tiles_cv_file, coordsfile, n_val_patches):    
    """ select indices of train and validation patches in coordsfile.
        Naive method, does not take tiles into account original tiles, no 
        splits for CV. However, test tiles are removed based on original tiles
    
    arguments
    ---------
        tiles_cv_file: string
            path to file where the tilenames for cross-validation are saved
        coordsfile: string
            path to file where the coordinates are saved
            
    return
    ------
        index_train: numpy ndarray
            indices of train patches in coordsfile
        index_val: numpy ndarray
            indices of validation patches in coordsfile
     
    """
    coords_df = pd.read_csv(coordsfile, sep=',')
    coords_df = coords_df[['tiles','Unnamed: 0']]
    
    tiles = np.load(tiles_cv_file, allow_pickle=True)
    
    # remove indices used for testing
    coords_trainval = coords_df[np.isin(coords_df['tiles'], tiles)]
    
    index_val = np.random.choice(coords_trainval['Unnamed: 0'], n_val_patches, replace=False)
    index_train = coords_trainval[np.isin(coords_trainval['Unnamed: 0'], index_val) == False]
    index_train = np.array(index_train['Unnamed: 0'])
    
    random.shuffle(index_train)
    random.shuffle(index_val)
    
    return(index_train, index_val)
    
def train_test_split_random(coordsfile):    
    """ select indices of train, validation and test patches in coordsfile.
        Naive method, does not take tiles into account original tiles, no 
        splits for CV.
    
    arguments
    ---------
        coordsfile: string
            path to file where the coordinates are saved
            
    return
    ------
        index_train: numpy ndarray
            indices of train patches in coordsfile
        index_val: numpy ndarray
            indices of validation patches in coordsfile
        index_test: numpy ndarray
            indices of test patches in coordsfile
     
    """    
    coords_df = pd.read_csv(coordsfile, sep=',')
    n_patches = len(coords_df)
    patches = np.arange(n_patches)  
    
    
    index_test = np.random.choice(patches, int(0.2*n_patches), replace=False)
    patches = patches[np.isin(patches, index_test) == False]
    
    index_val = np.random.choice(patches, int(0.2*n_patches), replace=False)
    index_train = patches[np.isin(patches, index_val) == False]
    
    return(index_train, index_val, index_test)
    
def train_val_split_subset(tiles_cv_file, coordsfile, folds, k, max_tiles):    
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
        max_tiles: int
            number of tiles to use for subset
            
    return
    ------
        index_train: numpy ndarray
            indices of train patches in coordsfile
        index_val: numpy ndarray
            indices of validation patches in coordsfile
     
    """
    coords_df = pd.read_csv(coordsfile, sep=',')
    coords_df = coords_df['tiles']
    
    tiles = np.load(tiles_cv_file, allow_pickle=True)
    tiles = tiles[:max_tiles]
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
    """ return indices of test patches
    
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
    coords_df = pd.read_csv(coordsfile, sep=',')
    coords_df = coords_df[['tiles','Unnamed: 0']]
    
    testtiles = np.load(tiles_test_file, allow_pickle=True)
    
    # find indices
    coords_test = coords_df[np.isin(coords_df['tiles'], testtiles)]
    
    index_test = np.array(coords_test['Unnamed: 0'])
    
    return(index_test)
    
def get_patches(patchespath, indices, patch_size_padded, channels, resolution):
    """ load patches
    
    arguments
    ---------
        patchespatch: string
            path to folder containing the patches
        indices: list
            indices of patches to load
        patch_size_padded: ubt
            size of patch to extract, including padding.
        channels: list
            list with channels in the image to use for prediction to be predicted
        resolution: int
            either 1 for 1m or 20 for 20cm
            
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
    if resolution == 20:
        X = np.zeros((n_patches, 480, 480, 5), dtype=np.float16) 
        y = np.zeros((n_patches, 480, 480, 5), dtype=np.int8)
    elif resolution == 1:
        X = np.zeros((n_patches, 96, 96, 5), dtype=np.float16) 
        y = np.zeros((n_patches, 96, 96, 5), dtype=np.int8)
    
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

def count_classes(patchespath, coordsfile, class_names, res, plot=False, 
                  savepath = '/data3/marrit/GrasslandProject/output/images',
                  filename = None):
    """ Count the number of pixels of each class and return percentage per class
    
    arguments
    ---------
        patchespatch: string
            path to folder containing the patches   
        coordsfile: string
            path to file where the coordinates are saved  
        class_names: list
            list with class names
        res: int 
            resolution of the patches to count classes
        plot: boolean
            if True plot distribution of classes. Default = False
        savepath: string
            location to save plot
        filename: string
            filename of plot
            
    returns
    -------
        percentage: pandas.Series
            pandas series with the percentage per class
    """
    # read coordsfile
    coords_df = pd.read_csv(coordsfile, sep=',')
    
    # add empty colums for class counts
    col_names = {i: class_names[i]+'_' + str(res) for i in range(len(class_names))}
    
    for i in range(len(class_names)):
        coords_df[col_names[i]] = np.nan
      
    # loop over patches
    for i in range(len(coords_df)):
            
        #load patch
        patch = np.load(patchespath + 'labels/' + str(i) + '.npy')
        
        # count classes
        for j in range(len(class_names)):
            coords_df.at[i,col_names[j]]= np.sum(patch[:,:,j])
        
        if i % 500 == 0: 
                print('\r {}/{}'.format(i, len(coords_df)),end='') 
            
    # save csv
    savefile = coordsfile.replace('patches.csv', 'patches_'+str(res)+'.csv')
    coords_df.to_csv(savefile)        

    # calculate percentages
    tot_per_class = coords_df[[v for v in col_names.values()]].sum(axis=0)
    tot = tot_per_class.sum()   
    percentage = tot_per_class / tot
    
    if plot:
        barplot_classes(tot_per_class, class_names, savepath, filename)

    return percentage    

def find_patches_options(gt, patch_size_padded, tile):
    """
    
    arguments
    ---------
        gt: numpy.ndarray
            2D numpy array with ground truth
        patch_size_padded: int
            patch size including padding for later predictions
    
    returns
    -------
        combinations_usable: pandas Dataframe
            Dataframe with optional starting points for patches
    """
     
    # find starting points on grid, should include gt
    gt_usable = np.argwhere(gt!=0)
    df_usable = pd.DataFrame(gt_usable, columns=['row', 'col'])
    df_usable = df_usable.divide(patch_size_padded).astype(int)

    pixels_per_patch = patch_size_padded*patch_size_padded
    
    combinations = df_usable.groupby(['row','col']).size().reset_index()
    combinations_usable = combinations[combinations[0] == pixels_per_patch].copy()
    combinations_usable['tiles'] = tile
    combinations_usable.drop(0, axis=1, inplace=True)

    #plot_patch_options(gt, combinations_usable, patch_size_padded)
    
    return combinations_usable
    

