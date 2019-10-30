#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:31:08 2019

@author: cordolo
"""
from metrics import AverageMeter, accuracy, intersectionAndUnion, updateConfusionMatrix, compute_mcc
import numpy as np
import pandas as pd
import time
from dataset import get_patches
from tensorflow.keras.models import load_model
from models.BilinearUpSampling import BilinearUpSampling2D
from tqdm import tqdm
from plots import plot_confusion_matrix2
import os


def test(classes, index_test, data_path, patch_size, patch_size_padded, max_size, channels, resolution, model_path, results_path, visualize=False, class_names):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    acc_meter_patch = AverageMeter()
    intersection_meter_patch = AverageMeter()
    union_meter_patch = AverageMeter()
    time_meter = AverageMeter()
    
    # initiate confusion matrix
    conf_matrix = np.zeros((len(classes), len(classes)))
    conf_matrix_patch = np.zeros((len(classes), len(classes)))
    
    # TODO initialise for umap
    #area_activations_mean = np.zeros((len(index_test),32*32))
    #area_activations_max = np.zeros((len(index_test),32*32))
    #area_cl = np.zeros((len(index_test),), dtype=np.int)
    #area_loc = np.zeros((len(index_test),3), dtype=np.int)
    #j = 0
    
    # initiate model
    model = load_model(model_path, custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D})
    #extra_output = [layer.output for layer in model.layers[-3::2]]
    #visualization_model = Model(i)
    
    pbar = tqdm(total=len(index_test))
    for i, idx in enumerate(index_test):

        # load data
        X, y = get_patches(data_path, [idx], patch_size_padded, channels, resolution=resolution)
        y = np.argmax(np.squeeze(y),axis = 2)
        tic = time.perf_counter()
        
        pred = np.argmax(np.squeeze(model.predict(X)),axis = 2)
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, y)
        acc_patch, pix_patch = accuracy(
                pred[patch_size:2*patch_size, 
                     patch_size:2*patch_size], 
                y[patch_size:2*patch_size, 
                     patch_size:2*patch_size])
        
        intersection, union = intersectionAndUnion(pred, y, len(classes))
        intersection_patch, union_patch = intersectionAndUnion(
                pred[patch_size:2*patch_size, 
                     patch_size:2*patch_size], 
                y[patch_size:2*patch_size, 
                     patch_size:2*patch_size],
                len(classes))
        
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_meter_patch.update(acc_patch, pix_patch)
        intersection_meter_patch.update(intersection_patch)
        union_meter_patch.update(union_patch)
        
        conf_matrix = updateConfusionMatrix(conf_matrix, pred, y)
        
        # update conf matrix patch
        conf_matrix_patch = updateConfusionMatrix(
                conf_matrix_patch, 
                pred[patch_size:2*patch_size, 
                     patch_size:2*patch_size], 
                y[patch_size:2*patch_size, 
                     patch_size:2*patch_size])

        # visualization
        if visualize:
            np.save(os.path.join(results_path, str(idx)+'.npy'))

        pbar.update(1)
 
# TODO find activations       
# =============================================================================
#         row, col, cl = find_constant_area(seg_label,32,cfg['DATASET']['patch_size_padded']) #TODO patch_size_padded must be patch_size if only inner patch is checked.
#         if not (row == 999999):
#             activ_mean = np.mean(as_numpy(activations.features.squeeze(0).cpu()),axis=0, keepdims=True)[:,row//4:row//4+8, col//4:col//4+8].reshape(1,8*8)
#             activ_max = np.max(as_numpy(activations.features.squeeze(0).cpu()),axis=0, keepdims=True)[:,row//4:row//4+8, col//4:col//4+8].reshape(1,8*8)
#     
#             area_activations_mean[j] = activ_mean
#             area_activations_max[j] = activ_max
#             area_cl[j] = cl
#             area_loc[j,0] = row
#             area_loc[j,1] = col
#             area_loc[j,2] = int(batch_data['info'].split('.')[0])
#             j+=1
#         else:
#             area_activations_mean[j] = np.full((1,64),np.nan,dtype=np.float32)
#             area_activations_max[j] = np.full((1,64),np.nan,dtype=np.float32)
#             area_cl[j] = 999999
#             area_loc[j,0] = row
#             area_loc[j,1] = col
#             area_loc[j,2] = int(batch_data['info'].split('.')[0])
#             j+=1
# =============================================================================
            
        #activ = np.mean(as_numpy(activations.features.squeeze(0).cpu()),axis=0)[row//4:row//4+8, col//4:col//4+8]
        #activ = as_numpy(activations.features.squeeze(0).cpu())

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))
    iou_patch = intersection_meter_patch.sum / (union_meter_patch.sum + 1e-10)
    for i, _iou_patch in enumerate(iou_patch):
        print('class [{}], patch IoU: {:.4f}'.format(i, _iou_patch))    

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))
    print('Patch: Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou_patch.mean(), acc_meter_patch.average()*100, time_meter.average()))
    
    print('Confusion matrix:')
    plot_confusion_matrix2(conf_matrix, class_names, 
                          normalize = True, title='confusion matrix patch+padding',
                          cmap=plt.cm.Blues)
    plot_confusion_matrix2(conf_matrix_patch, class_names, 
                          normalize = True, title='confusion matrix patch',
                          cmap=plt.cm.Blues)
    
    np.save(os.path.join(results_path,'confmatrix.npy'), conf_matrix)
    np.save(os.path.join(results_path,'confmatrix_patch.npy'), conf_matrix_patch)
    #np.save(os.path.join(results_path, 'activations_mean.npy'), area_activations_mean) # TODO
    #np.save(os.path.join(results_path, 'activations_max.npy'), area_activations_max)
    #np.save(os.path.join(results_path, 'activations_labels.npy'), area_cl)
    #np.save(os.path.join(results_path, 'activations_loc.npy'), area_loc)
    
    mcc = compute_mcc(conf_matrix)
    mcc_patch = compute_mcc(conf_matrix_patch)
    # save summary of results in csv
    summary = pd.DataFrame([[model_path.split('_')[0].split('/')[-1],resolution, 
                             patch_size, channels, acc_meter.average(),
                             acc_meter_patch.average(), iou.mean(),iou_patch.mean(), mcc, mcc_patch]], 
    columns=['model','resolution','patch_size','channels', 'test_accuracy','test_accuracy_patch', 'meanIoU', 'meanIoU_patch', 'mcc','mcc_patch'])
    summary.to_csv(os.path.join(results_path,'summary_results.csv'))

    print('Evaluation Done!')


