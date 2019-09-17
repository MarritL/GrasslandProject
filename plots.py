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
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import umap
from sklearn.decomposition import PCA

# colormap
colors = ['linen', 'lightgreen', 'green', 'darkgreen', 'yellow']
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
    


def plot_confusion_matrix(gt, pred, classes, class_names, normalize=False, axis = 1, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix for images with one-hot encoded labels.
    Normalization can be applied by setting `normalize=True`.
    
    arguments
    ---------
        gt: np.ndarray
        pred: np.ndarray
        classes: list
        class_names: list
        normalize: boolean
            default=False 
        axis: int: 0 or 1
            default = 1 
        title: string
            default = None
        cmap: matplotlib color map
            default = plt.cm.Blues
    
    returns
    -------
        printed confusion table and plot
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    
    y_true = np.zeros(gt.shape[:3], dtype=np.uint8)
    y_pred = np.zeros(pred.shape[:3], dtype=np.uint8)
    for i in range(gt.shape[0]):
        y_true[i] = np.argmax(gt[i], axis=2)
        y_pred[i] = np.argmax(pred[i], axis=2)

    # Compute confusion matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=classes)
    
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true.flatten(), y_pred.flatten())]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
    colors = ['red','green','blue','purple','yellow']
    colors_map = umap_gt_sq[:,]
    #tare = [770,659,654,690,650]
    #for i, cl in enumerate(tare):
    for cl in range(5):
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
