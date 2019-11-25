# System libs
import os
import time
from sklearn.metrics import matthews_corrcoef
# Numerical libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# Our libs
from datagenerator_pytorch import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion
from utils import updateConfusionMatrix, compute_mcc, find_constant_area
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from plots import plot_confusion_matrix
import matplotlib.pyplot as plt

colors = np.load('/data3/marrit/GrasslandProject/GrasslandProject_pytorch/data/colors5.npy')

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=False)
    def close(self):
        self.hook.remove()

def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    img = img*255

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img[:,:,[0,1,2]], seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.npy', '.png')))
  

def evaluate(segmentation_module, loader, cfg, gpu, activations, num_class, 
             patch_size, patch_size_padded, class_names, channels, index_test, 
             visualize, results_dir, arch_encoder):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    acc_meter_patch = AverageMeter()
    intersection_meter_patch = AverageMeter()
    union_meter_patch = AverageMeter()
    time_meter = AverageMeter()
    
    # initiate confusion matrix
    conf_matrix = np.zeros((num_class, num_class))
    conf_matrix_patch = np.zeros((num_class, num_class ))
    # turn on for initialise for umap
    area_activations_mean = np.zeros((len(index_test),32//4*32//4))
    area_activations_max = np.zeros((len(index_test),32//4*32//4))
    area_cl = np.zeros((len(index_test),), dtype=np.int)
    area_loc = np.zeros((len(index_test),3), dtype=np.int)
    j = 0

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:

        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        acc_patch, pix_patch = accuracy(
                pred[patch_size:2*patch_size, 
                     patch_size:2*patch_size], 
                seg_label[patch_size:2*patch_size, 
                     patch_size:2*patch_size])
        
        intersection, union = intersectionAndUnion(pred, seg_label,num_class)
        intersection_patch, union_patch = intersectionAndUnion(
                pred[patch_size:2*patch_size, 
                     patch_size:2*patch_size], 
                seg_label[patch_size:2*patch_size, 
                     patch_size:2*patch_size],
                num_class)
        
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_meter_patch.update(acc_patch, pix_patch)
        intersection_meter_patch.update(intersection_patch)
        union_meter_patch.update(union_patch)
        
        conf_matrix = updateConfusionMatrix(conf_matrix, pred, seg_label)
        
        # update conf matrix patch
        conf_matrix_patch = updateConfusionMatrix(
                conf_matrix_patch, 
                pred[patch_size:2*patch_size, 
                     patch_size:2*patch_size], 
                seg_label[patch_size:2*patch_size, 
                     patch_size:2*patch_size])

        # visualization
        if visualize:
            info = batch_data['info']
            img_name = info.split('/')[-1]
            #np.save(os.path.join(test_dir, 'result', img_name), pred)
            np.save(os.path.join(results_dir, img_name), pred)
            
# =============================================================================
#         if visualize:
#             visualize_result(
#                 (batch_data['img_ori'], seg_label, batch_data['info']),
#                 pred,
#                 os.path.join(test_dir, 'result')
#             )
# =============================================================================

        pbar.update(1)
  
# turn on for UMAP      
        row, col, cl = find_constant_area(seg_label,32,patch_size_padded) #TODO patch_size_padded must be patch_size if only inner patch is checked.
        if not (row == 999999):
            activ_mean = np.mean(as_numpy(activations.features.squeeze(0).cpu()),axis=0, keepdims=True)[:,row//4:row//4+8, col//4:col//4+8].reshape(1,8*8)
            activ_max = np.max(as_numpy(activations.features.squeeze(0).cpu()),axis=0, keepdims=True)[:,row//4:row//4+8, col//4:col//4+8].reshape(1,8*8)
    
            area_activations_mean[j] = activ_mean
            area_activations_max[j] = activ_max
            area_cl[j] = cl
            area_loc[j,0] = row
            area_loc[j,1] = col
            area_loc[j,2] = int(batch_data['info'].split('.')[0])
            j+=1
        else:
            area_activations_mean[j] = np.full((1,64),np.nan,dtype=np.float32)
            area_activations_max[j] = np.full((1,64),np.nan,dtype=np.float32)
            area_cl[j] = 999999
            area_loc[j,0] = row
            area_loc[j,1] = col
            area_loc[j,2] = int(batch_data['info'].split('.')[0])
            j+=1
            
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
    plot_confusion_matrix(conf_matrix, class_names, 
                          normalize = True, title='confusion matrix patch+padding',
                          cmap=plt.cm.Blues)
    plot_confusion_matrix(conf_matrix_patch, class_names, 
                          normalize = True, title='confusion matrix patch',
                          cmap=plt.cm.Blues)
    
    np.save(os.path.join(results_dir,'confmatrix.npy'), conf_matrix)
    np.save(os.path.join(results_dir,'confmatrix_patch.npy'), conf_matrix_patch)
    # turn on for UMAP
    np.save(os.path.join(results_dir, 'activations_mean.npy'), area_activations_mean)
    np.save(os.path.join(results_dir, 'activations_max.npy'), area_activations_max)
    np.save(os.path.join(results_dir, 'activations_labels.npy'), area_cl)
    np.save(os.path.join(results_dir, 'activations_loc.npy'), area_loc)
    
    mcc = compute_mcc(conf_matrix)
    mcc_patch = compute_mcc(conf_matrix_patch)
    # save summary of results in csv
    summary = pd.DataFrame([[arch_encoder,patch_size,
                             channels, acc_meter.average(),
                             acc_meter_patch.average(), iou.mean(),iou_patch.mean(), mcc, mcc_patch]], 
    columns=['model','patch_size','channels', 'test_accuracy','test_accuracy_patch', 'meanIoU', 'meanIoU_patch', 'mcc','mcc_patch'])
    summary.to_csv(os.path.join(results_dir,'summary_results.csv'))


def test(cfg, gpu, arch_encoder, arch_decoder, fc_dim, channels, weights_encoder, 
         weights_decoder, num_class, class_names, pretrained, patchespath, patch_size, 
         patch_size_padded, index_test, visualize, results_dir):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=arch_encoder.lower(),
        fc_dim=fc_dim,
        n_channels= len(channels),
        weights=weights_encoder,
        pretrained=pretrained)    
            
    net_decoder = ModelBuilder.build_decoder(
        arch=arch_decoder.lower(),
        fc_dim=fc_dim,
        num_class=num_class,
        weights=weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test= TestDataset(
        data_path=patchespath, 
        indices = index_test, 
        patch_size_padded=patch_size*3, 
        channels=channels, 
        n_classes = num_class, 
        augment=False)
            
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=False)
    
    activations = SaveFeatures(list(net_decoder.children())[0])

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_test, cfg, gpu, activations, num_class, 
             patch_size, patch_size_padded, class_names, channels, index_test, 
             visualize, results_dir, arch_encoder)

    print('Evaluation Done!')
    
    activations.hook.remove()

