#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:28:16 2019

@author: cordolo
"""

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from plots import plot_confusion_matrix
import numpy as np
 
def compute_confusion_matrix(gt, pred, classes, class_names, user_producer=True, normalize=False, axis = 1, plot=True, title=None):
    """Compute the confusion matrix 
    
    arguments
    ---------
        gt: numpy.ndarray
            one-hot lables of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)    
        pred: numpy.ndarray
            probability maps of classes of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)  
        user_producer: boolean
            if true: the user, producer, and total accuracy will be calculated
        normalize: boolean
            default=False
        axis: int
            one of 0 or 1. Default=1
            0 for division by column total and 1 for division by row total
            i.e. thus in the TP-cells if axis=0:User's acc, if axis=1:Producer's acc.
        plot: boolean
            if true: the confusion matrix will be plotted. default is True. 
        title: string
            title of the plot. default = None
        cmap: matplotlib color map
            cmap of the plot. default = plt.cm.Blues

    returns
    -------
        cm: numpy.ndarray
            confuion matrix of shape (n_classes, nclasses)
    if plot=True
        plot: fig
            plot of the confusion amtrix
    """  
    
    y_true = np.zeros(gt.shape[:3], dtype=np.uint8)
    y_pred = np.zeros(pred.shape[:3], dtype=np.uint8)
    for i in range(gt.shape[0]):
        y_true[i] = np.argmax(gt[i], axis=2)
        y_pred[i] = np.argmax(pred[i], axis=2)

    # Compute confusion matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis]
        
    if user_producer:
        if normalize:
            print("user and producer accuracy can only be calculated for un- \
                  normalized confusion matrices")
        else:
            cm = compute_user_producer_acc(cm)
            class_names.extend('accuracy')
    
    if plot:
        plot_confusion_matrix(cm, class_names, normalize, title)  
    
    return(cm)
    
def compute_matthews_corrcoef(gt, pred):
    """Compute the Matthews correlation coefficient (MCC)
    
    arguments
    ---------
        gt: numpy.ndarray
            one-hot lables of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)    
        pred: numpy.ndarray
            probability maps of classes of patches
            shape = (n_patches, patch_size_padded, patch_size_padded, n_classes)   

    returns
    -------
        mcc: float
            Matthews correlation coefficient (MCC).
    """
    
    y_true = np.zeros(gt.shape[:3], dtype=np.uint8)
    y_pred = np.zeros(pred.shape[:3], dtype=np.uint8)
    for i in range(gt.shape[0]):
        y_true[i] = np.argmax(gt[i], axis=2)
        y_pred[i] = np.argmax(pred[i], axis=2)

    # Compute matthews correlation coef
    mcc = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
    
    return(mcc)

def compute_user_producer_acc(cm):
    """ computes user and producer accuracy for existing confusion matrix
    
    arguments
    ---------
        cm: numpy.ndarray
            
    
    returns
    -------
        new_cm: numpy_ndarray
            confionsion matrix with an extra column and row for producer and 
            user accuracys. The last cell holds overall accuracy. 
    """
    
    cm = cm.astype('float')
    row_sum = np.sum(cm,axis=1)
    col_sum = np.sum(cm,axis=0)
    
    diag = np.eye(cm.shape[0])*cm
    
    user = diag[np.nonzero(diag)]/col_sum
    producer = diag[np.nonzero(diag)]/row_sum
    accuracy = np.sum(diag)/np.sum(cm)
    
    new_cm = np.zeros((cm.shape[0]+1, cm.shape[1]+1), dtype='float')
    new_cm[:cm.shape[0],:cm.shape[1]] = cm
    new_cm[cm.shape[0]+1,:-1] = user
    new_cm[:-1,cm.shape[0]+1] = producer
    new_cm[cm.shape[0]+1, cm.shape[1]+1] = accuracy
    
    return(new_cm)
    
    
    
    
    
    
    
    