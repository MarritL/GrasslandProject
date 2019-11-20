#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:05:50 2019

@author: cordolo
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import list_files
from matplotlib.colors import ListedColormap
import earthpy.plot as ep
import umap
from sklearn.decomposition import PCA
from osgeo import gdal
import matplotlib.patches as patches
import seaborn as sns
import os


# colormap
#colors = ['linen', 'lightgreen', 'limegreen', 'darkgreen', 'yellow']
colors = np.array([(194/256,230/256,153/256),(120/256,198/256,121/256),
    (49/256,163/256,84/256),(0/256,76/256,38/256),(229/256,224/256,204/256)])
cmap = ListedColormap(colors)

def plot_random_patches(patches_path, n_patches, classes, class_names):
    """ plot random patches with ground truth
    
    arguments
    ---------
        patches_path: string
            path to folder containing the patches 
        n_patches: int
            number of random patches to plot
        classes: list
            list of predicted classes
        class_names: list
            
    output
    ------
        figure with n_patches plotted in first row and ground truth in second row.
    """
    
    images_path = patches_path + 'images/'
    labels_path = patches_path + 'labels/'
    
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
    
    ep.draw_legend(grtr,titles=class_names,classes=classes)

def plot_predicted_patches(predictions, groundtruth, patch=None):
    """ plot predicted patches with ground truth
    
    arguments
    ---------
        predictions: numpy nd.array
            probability maps of classes of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
        groundtruth: numpy nd.array
            one-hot lables of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
        patch: numpy.ndarray
            image of patch
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
    
    output
    ------
        figure with n predictions plotted in first row and ground truth in second row.
        if patch is specified the third row contains RGB image
    """

    n_patches = len(predictions)
    cols = 2
    if np.any(patch != None):
        cols = 3
    
    
    # prepare
    fig, ax = plt.subplots(n_patches,cols, figsize=(15,15))
    
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
        ax[i,0].imshow(plt_im, cmap=cmap, vmin=0, vmax=4)
        
        # plot gt 
        grtr = ax[i,1].imshow(plt_gt, cmap=cmap, vmin=0, vmax=4) 
        
        # plot RGB
        if np.any(patch != None):
            plt_im = patch[i][:, :, [0,1,2]].astype(np.float64)
            ax[i,2].imshow(plt_im)
    
    ep.draw_legend(grtr,titles=["tara0", "tara20", "tara50", "woods","no coltivable"],classes=[0, 1, 2, 3,4])


def plot_prediction_patch(predictions, groundtruth, patch):
    """ plot predicted patches with ground truth
    
    arguments
    ---------
        predictions: numpy nd.array
            probability maps of classes of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
        groundtruth: numpy nd.array
            one-hot lables of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
        patch: numpy.ndarray
            image of patch
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
    
    output
    ------
        figure with n predictions plotted in first row and ground truth in second row.
        if patch is specified the third row contains RGB image
    """

    rows = 1
    cols = 4  
    

    
    pred = predictions[0]
    gt = groundtruth[0]
    
    # prepare prediction plot
    plt_pred  = np.zeros_like(pred, dtype=np.uint8)
    plt_pred = np.argmax(pred, axis=2)
    
    # prepare gt plot
    plt_gt  = np.zeros_like(gt, dtype=np.uint8)
    plt_gt = np.argmax(gt, axis=2)

    # plot training image
    plt_im = patch[0][:, :, [0,1,2]].astype(np.float64)
    
    # probability plot
    plt_prob  = np.zeros_like(pred, dtype=np.float32)
    plt_prob = 1-np.max(pred, axis=2)
    
    # prepare
    fig, ax = plt.subplots(rows,cols, figsize=(10,10))

    # add to plot
    ax[0].imshow(plt_im)
    ax[1].imshow(plt_gt, cmap=cmap, vmin=0, vmax=4) 
    ax[2].imshow(plt_pred, cmap=cmap, vmin=0, vmax=4)
    ax[3].imshow(plt_prob)
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])
    ax[3].set_yticklabels([])
    ax[3].set_xticklabels([])


    

        
    

     
def plot_patches(patch, gt, n_patches):
    """ plot patches with ground truth
    
    arguments
    ---------
        patch: numpy.ndarray
            image of patch
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
        gt: numpy.ndarray
            one-hot lables of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)
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
        plt_im = patch[i][:, :, index].astype(np.float64)
        
        # prepare gt plot
        plt_gt  = np.zeros_like(gt[i], dtype=np.uint8)
        plt_gt = np.argmax(gt[i], axis=2)
    
        # plot training image
        image = ax[0,i].imshow(plt_im)
        
        # plot gt 
        grtr = ax[1,i].imshow(plt_gt, cmap=cmap, vmin=0, vmax=4)
    
    ep.draw_legend(grtr,titles=["tara0", "tara20", "tara50", "forest","non-cultivable"],classes=[0, 1, 2, 3,4])

def plot_predicted_probabilities(predictions, groundtruth, n_classes, uncertainty):
    """ plot predictions with ground truth
    
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
        uncertainty: float between 0 and 1
            if uncertainty is higher than this number, class 'unsure' is assigned
            
    
    output
    ------
        figure with predictionsmaps plotted in first n rows and ground truth in last row. 
        The columns represent differnet patches.
    """

    #colors_extra = ['linen', 'lightgreen', 'limegreen', 'darkgreen', 'yellow', 'black']
    colors_extra = np.array([(194/256,230/256,153/256),(120/256,198/256,121/256),
    (49/256,163/256,84/256),(0/256,76/256,38/256),(229/256,224/256,204/256),(0,0,0)])
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
        plt_im[np.max(im, axis=2)<(1-uncertainty)] = 5

        
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
    
    ep.draw_legend(im,titles=["tara0", "tara20", "tara50", "forest","non-cultivable","not sure"],classes=[0, 1, 2, 3,4,5])
    
def plot_prediction_uncertainty(predictions, groundtruth, n_classes):
    """plot predictions uncertainties
    
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
        figure with prediction plotted in first row, the uncertainty map in the
        second row and the ground truth in last row. 
        The columns represent different patches
    """
    
    n_patches = len(predictions)
    rows = 3
    
    # prepre
    fig, ax = plt.subplots(rows,n_patches)
    
    for i in range(n_patches):
    
        prob = predictions[i]
        gt = groundtruth[i]
        
        # prepare prediction uncertainty plot
        plt_prob  = np.zeros_like(prob, dtype=np.float32)
        plt_prob = np.max(prob, axis=2)
        
        # prepare prediction plot
        plt_im  = np.zeros_like(prob, dtype=np.uint8)
        plt_im = np.argmax(prob, axis=2)
        
        # prepare gt plot
        plt_gt  = np.zeros_like(gt, dtype=np.uint8)
        plt_gt = np.argmax(gt, axis=2)
        
        # plot prediction 
        im = ax[0,i].imshow(plt_im, cmap=cmap, vmin=0, vmax=4) 
        
        # plot uncertainty
        prob = ax[1,i].imshow(plt_prob, vmin=0, vmax=1)
        
        # plot gt 
        grtr = ax[2,i].imshow(plt_gt, cmap=cmap, vmin=0, vmax=4) 
    
    plt.colorbar(prob, ax=ax[1,i])
    

def plot_confusion_matrix(cm, class_names, normalize = True, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix for images with one-hot encoded labels.
    Normalization can be applied by setting `normalize=True`.
    
    arguments
    ---------
        cm: numpy.ndarray
            confusion matrix
        class_names: list
            class labels for confusion matrix
        normalize: boolean
            default=True
        title: string
            default = None
        cmap: matplotlib color map
            default = plt.cm.Blues
    
    returns
    -------
        plot of confusion table
    """
    if not title:
        title = 'Confusion matrix'

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_confusion_matrix2(cm, class_names, normalize = True, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix for images with one-hot encoded labels.
    Normalization can be applied by setting `normalize=True`.
    
    arguments
    ---------
        cm: numpy.ndarray
            confusion matrix
        class_names: list
            class labels for confusion matrix
        normalize: boolean
            default=True
        title: string
            default = None
        cmap: matplotlib color map
            default = plt.cm.Blues
    
    returns
    -------
        plot of confusion table
    """
   
    if not title:
        title = ''
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        ax.figure.colorbar(im, ax=ax)
    else:    
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
   
    return ax

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['categorical_accuracy'])
    plt.plot(network_history.history['val_categorical_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
def umap_plot(pixel_values, labels, n_neighbors=15, min_dist=0.2, metric='euclidean', title=''):
    """ Create a UMAP reduced dimensionlity plot 
    
    eyword arguments:
    pixel_values -- np.array with shape (nrows, ncols) 
                    (e.g. create with im.reshape(im.shape[0]*im.shape[1], im.shape[2])
                    in case of multiple bands)  
    labels -- np.array with shape (nrows,)
              e.g. if shape is (x,1) squeeze will create the right shape.  
    
    To run e.g.
    umap_plot(umap_im, umap_gt)
    """    
    # UMAP 
    reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            min_dist = min_dist,
            metric = metric)
    embedding = reducer.fit_transform(pixel_values)
    
    # make sure the shape of the labels array is right
    #umap_gt_sq = np.squeeze(labels)
    umap_gt_sq = labels
    
    #plot
    colors = ['red','green','blue','purple','yellow','black']
    colors_map = umap_gt_sq[:,]
    #tare = [770,659,654,690,650]
    #for i, cl in enumerate(tare):
    for cl in range(6):
        indices = np.where(colors_map==cl)
        plt.scatter(embedding[indices,0], embedding[indices, 1], c=colors[cl], label=[cl])
    plt.legend()
    plt.title(title)
    plt.show() 

    
def pca_plot(pixel_values, labels):
    """ Create a PCA reduced dimensionlity plot 
    
    Keyword arguments:
    pixel_values -- np.array with shape (nrows, ncols) 
                    (e.g. create with im.reshape(im.shape[0]*im.shape[1], im.shape[2])
                    in case of multiple bands)  
    labels -- np.array with shape (nrows,)
              e.g. if shape is (x,1) squeeze will create the right shape.  
    
    To run e.g.
    pca_plot(umap_im, umap_gt)
    """
    # pca
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(pixel_values)
    
     # make sure the shape of the labels array is right
    umap_gt_sq = np.squeeze(labels)
    
    # plot
    colors = ['red','green','blue','purple','yellow']
    colors_map = umap_gt_sq[:,]
    tare = [770,659,654,690,650]
    #for i, cl in enumerate(tare):
    for cl in range(5):
        indices = np.where(colors_map==cl)
        plt.scatter(principalComponents[indices,0], principalComponents[indices, 1], c=colors[cl], label=[cl])
    plt.legend()
    plt.show()
    
def plot_patches_on_tile(coordsfile, tiles_path, tile, patch_size_padded):
    """ plot the patches on top of the orginal tile
    
    arguments
    ---------
        coordsfile: string
            path to file where the coordinates are saved
        tiles_path: string
            path to folder containing the tiles
        tile: string
            name of tile to plot
        patch_size_padded: int
            
    
    returns
    -------
        plot of tile with the patches outlined
    """
    
    #colors = ['black', 'linen', 'lightgreen', 'green', 'darkgreen', 'yellow']    
    colors = np.array([(0,0,0),(194/256,230/256,153/256),(120/256,198/256,121/256),
    (49/256,163/256,84/256),(0/256,76/256,38/256),(229/256,224/256,204/256)])
    
    cmap = ListedColormap(colors)
    coords_df = pd.read_csv(coordsfile, sep=',')
    is_tile = coords_df['tiles'] == tile
    patches_tile = coords_df[is_tile]
    
    # get tile
    path_shp = tiles_path + tile + '/tare.tif'
    ds = gdal.Open(path_shp,gdal.GA_ReadOnly)
    gt = ds.GetRasterBand(1).ReadAsArray()
    gt = np.uint16(gt)
    ds = None
       
    
    # set classes for plotting
    gt[gt==638] = 1
    gt[gt==659] = 2
    gt[gt==654] = 3
    gt[gt==650] = 4
    gt[gt==770] = 5
    
    # Create figure and axes
    fig,ax = plt.subplots(figsize=(10,10))
    
    # plot image
    im = ax.imshow(gt, cmap=cmap, vmin=0, vmax=5)
    
    # Create a square for patch
    for index, row in patches_tile.iterrows(): 
        r = int(row['row'])
        c = int(row['col'])
        patch = patches.Rectangle((c,r),patch_size_padded,patch_size_padded,linewidth=1,edgecolor='r',facecolor='none')
    
        # Add the patch to the Axes
        ax.add_patch(patch)   
    
    #ep.draw_legend(im,titles=["","tara0", "tara20", "tara50", "forest","non-cultivable"],classes=[0,1, 2, 3,4,5])
    plt.savefig('/data3/marrit/GrasslandProject/output/images/grid_patches_tile_061032w.png')
    plt.show()

def plot_patch_options(gt, starting_points, patch_size_padded):
    """ plot the options for patches on the tile
    
    arguments
    ---------
        gt: np.array
            2D numpy array with ground truth
        starting_points: pandas dataframe
            dataframe with the starting points in columns 'row' and 'col' 
            will by multiplied by patch_size_padded
        patch_size_padded: int
            patch size with padding for prediction
            
    returns
    -------
        plot of the ground truth tile with squares patch options.
    
    """
    colors = ['black', 'linen', 'lightgreen', 'green', 'darkgreen', 'yellow']
    cmap = ListedColormap(colors)
    
    gt[gt==638] = 1
    gt[gt==659] = 2
    gt[gt==654] = 3  
    gt[gt==650] = 4
    gt[gt==770] = 5
    
    # Create figure and axes
    fig,ax = plt.subplots(figsize=(10,10))
    
    # plot image
    ax.imshow(gt, cmap=cmap, vmin=0, vmax=5)
    
    # Create a square for patch
    for index, row in starting_points.iterrows(): 
        r = row['row']
        c = row['col']
        patch = patches.Rectangle((c,r),patch_size_padded,patch_size_padded,linewidth=1,edgecolor='r',facecolor=None)
    
        # Add the patch to the Axes
        ax.add_patch(patch)
    
    plt.show()   

def plot_tile(inputpath, tile):
    """ plot a complete tile
    
    arguments
    ---------
        inputpath: string
            path to folders with tiles. Each tile should be in separate folder 
        tile: string
            tileidentifier, should be the file where tile is saved
            
    return
    ------
        plot of specified tile
    """    
    
    colors = ['black', 'linen', 'lightgreen', 'green', 'darkgreen', 'yellow']
    cmap = ListedColormap(colors)

    # get tile
    path_shp = inputpath + tile + '/tare.tif'
    ds = gdal.Open(path_shp,gdal.GA_ReadOnly)
    gt = ds.GetRasterBand(1).ReadAsArray()
    gt = np.uint16(gt)
    ds = None
    
    gt[gt==638] = 1
    gt[gt==659] = 2
    gt[gt==654] = 3  
    gt[gt==650] = 4
    gt[gt==770] = 5
    
    # Create figure and axes
    fig,ax = plt.subplots(figsize=(5,5))
    
    # plot image
    ax.imshow(gt, cmap=cmap, vmin=0, vmax=5)
    
    plt.show()

def barplot_classes(tot_per_class, class_names, savepath, filename):
    
    df = pd.DataFrame(tot_per_class, columns = ['number of pixels'])
    df['class']  = class_names
    sns.set(rc={'figure.figsize':(10,8)})
    ax = sns.barplot(y="number of pixels", x="class", data=df, palette=colors)
    
    total = sum(df['number of pixels'])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + 0.12
        y = p.get_y() + p.get_height() + 0.1
        ax.annotate(percentage, (x, y))
    
    plt.savefig(os.path.join(savepath,filename))
    
    plt.show()
    
# =============================================================================
# plot_tile(inputpath, '025164w')
# 
# # find starting points on grid, should include gt
# def get_patches_for_full_images():    
#     colors = ['black', 'linen', 'lightgreen', 'green', 'darkgreen', 'yellow']
#     cmap = ListedColormap(colors)
#     
#     # load tile
#     path_shp = inputpath + tile + '/tare.tif'
#     ds = gdal.Open(path_shp,gdal.GA_ReadOnly)
#     gt = ds.GetRasterBand(1).ReadAsArray()
#     gt = np.uint16(gt)
#     ds = None
#     
#     # get classes right
#     gt[gt==638] = 1
#     gt[gt==659] = 2
#     gt[gt==654] = 3  
#     gt[gt==650] = 4
#     gt[gt==770] = 5
#     
#     # find starting point of patches
#     gt_usable = np.argwhere(gt!=0)
#     df_usable = pd.DataFrame(gt_usable, columns=['row', 'col'])
#     df_usable = df_usable.divide(patch_size).astype(int)
#     combinations = df_usable.groupby(['row','col']).size().reset_index()
#     combinations_usable = combinations
#     combinations_usable['tiles'] = tile
#     combinations_usable.drop(0, axis=1, inplace=True)
#     combinations_usable['row'] = combinations_usable['row'].multiply(patch_size)
#     combinations_usable['col'] = combinations_usable['col'].multiply(patch_size)
#     
#     # remove 'halve' patches on edge
#     combinations_usable = combinations_usable[combinations_usable['row'] < (gt.shape[0]-patch_size_padded)]
#     combinations_usable = combinations_usable[combinations_usable['col'] < (gt.shape[1]-patch_size_padded)]
#     
#     # save in csv-file
#     combinations_usable.to_csv(coordspath+coordsfilename, index=False)
# 
# 
#     fig,ax = plt.subplots(figsize=(10,10))
#     
#     # plot image
#     ax.imshow(gt, cmap=cmap, vmin=0, vmax=5)
#     
#     # Create a square for patch
#     for index, row in combinations_usable.iterrows(): 
#         r = row['row']
#         c = row['col']
#         patch = patches.Rectangle((c,r),patch_size,patch_size,linewidth=1,edgecolor='r', fill=False)
#     
#         # Add the patch to the Axes
#         ax.add_patch(patch)
#     
#     plt.show() 
# 
# =============================================================================
