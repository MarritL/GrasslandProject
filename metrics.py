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
import tensorflow as tf
from keras import backend as K
 
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
            
    calls:
    ------
        compute_user_producer_acc()
        plot_confusion_matrix()

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
    for i, r in enumerate(row_sum):
        if r == 0:
            row_sum[i]= 1e-8
    
    col_sum = np.sum(cm,axis=0)
    for i, c in enumerate(col_sum):
        if c == 0:
            col_sum[i]= 1e-8
    
    diag = np.eye(cm.shape[0])*cm
    
    user = diag[np.nonzero(diag)]/col_sum
    producer = diag[np.nonzero(diag)]/row_sum
    accuracy = np.sum(diag)/np.sum(cm)
    
    new_cm = np.zeros((cm.shape[0]+1, cm.shape[1]+1), dtype='float')
    new_cm[:cm.shape[0],:cm.shape[1]] = cm
    new_cm[cm.shape[0],:-1] = user
    new_cm[:-1,cm.shape[0]] = producer
    new_cm[cm.shape[0], cm.shape[1]] = accuracy
    
    return(new_cm)

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

# =============================================================================
# def compute_mcc_tf():    
#     lb = LabelEncoder()
#     lb.fit(np.hstack([y_true, y_pred]))
#     y_true = lb.transform(y_true)
#     y_pred = lb.transform(y_pred)
# 
#     C = confusion_matrix(y_true, y_pred)
#     t_sum = C.sum(axis=1, dtype=np.float64)
#     p_sum = C.sum(axis=0, dtype=np.float64)
#     n_correct = np.trace(C, dtype=np.float64)
#     n_samples = p_sum.sum()
#     cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
#     cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
#     cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
#     mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
# 
#     if np.isnan(mcc):
#         return 0.
#     else:
#         return mcc    
# =============================================================================

# =============================================================================
# def compute_mcc_tf(output, target, from_logits=False):
#     """Matthews correlation coefficient (MCC) between an output tensor and a target tensor.
#     # Arguments
#         output: A tensor resulting from a softmax
#             (unless `from_logits` is True, in which
#             case `output` is expected to be the logits).
#         target: A tensor of the same shape as `output`.
#         from_logits: Boolean, whether `output` is the
#             result of a softmax, or is a tensor of logits.
#     # Returns
#         Output tensor.
#     """
#     _EPSILON = 1e-7
#     
#     # Note: tf.nn.softmax_cross_entropy_with_logits
#     # expects logits, Keras expects probabilities.
# 
#     # scale preds so that the class probas of each sample sum to 1
#     output /= tf.reduce_sum(output,
#                             reduction_indices=len(output.get_shape()) - 1,
#                             keep_dims=True)
#     # manual computation of crossentropy
#     epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
#     output = tf.clip_by_value(output, epsilon, 1. - epsilon)
#     
#     C = tf.math.confusion_matrix(target, output)
#     t_sum = tf.reduced_sum(C, axis=1)
#     p_sum = tf.reduced_sum(C, axis=0)
#     n_correct = tf.trace(C)
#     n_samples = tf.reduced_sum(p_sum)
#     
# 
#     cov_ytyp = tf.multipy(n_correct, n_samples) - tf.tensordot(t_sum, p_sum, axes = 0)
#     cov_ypyp = tf.math.pow(n_samples,2) - tf.tensordot(p_sum, p_sum, axes=0)
#     cov_ytyt = tf.math.pow(n_samples,2) - tf.tensordot(t_sum, t_sum, axes=0)
#             
#     return tf.math.divide(cov_ytyp, tf.math.sqrt(tf.math.multiply(cov_ytyt, cov_ypyp)))
# 
# def _to_tensor(x, dtype):
#     """Convert the input `x` to a tensor of type `dtype`.
#     # Arguments
#         x: An object to be converted (numpy array, list, tensors).
#         dtype: The destination type.
#     # Returns
#         A tensor.
#     """
#     x = tf.convert_to_tensor(x)
#     if x.dtype != dtype:
#         x = tf.cast(x, dtype)
#     return x  
# =============================================================================
    
def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy
    source: https://forums.fast.ai/t/unbalanced-classes-in-image-segmentation/18289/2
    
        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    if isinstance(weights,list) or isinstance(np.ndarray):
        weights=K.variable(weights)

    def loss(target,output,from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1)
        else:
            raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')
    return loss    

def dice_loss(y_true, y_pred):
    """
    source: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
    
    """
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

    return 1 - (numerator + 1) / (denominator + 1)

def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
    
def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def updateConfusionMatrix(confMatrix, preds, label):
    
    n_classes= confMatrix.shape[0]

    for cl in range(n_classes):
        pix = np.argwhere(label == cl)
        hist, edges = np.histogram(preds[pix], bins=n_classes, range=(0,n_classes-1))
        for cl_pred in range(n_classes):
            confMatrix[cl,cl_pred] += hist[cl_pred]
    
    return confMatrix   

def compute_mcc(conf_matrix):
    """ compute multi-class mcc using equation from source:
        https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef
        
    arguments
    ---------
        conf_matrix: numpy.ndarray
            confusion matrix. true in rows, predicted in columns
            
    return
    ------
        mcc: float
            matthews correlation coefficient
    
    """
    correct = np.trace(conf_matrix)
    tot = np.sum(conf_matrix)
    cl_true = np.sum(conf_matrix,axis = 1)
    cl_pred = np.sum(conf_matrix, axis = 0)

    mcc = (correct*tot-np.sum(cl_true*cl_pred))/(np.sqrt((tot**2-np.sum(cl_pred**2))*(tot**2-np.sum(cl_true**2))))
    return mcc