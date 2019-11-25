# System libs
import numpy as np
import pandas as pd
import time
from tensorboardX import SummaryWriter
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from pytorch.datagenerator_pytorch import TrainDataset
from pytorch.models.models import ModelBuilder, SegmentationModule
from pytorch.utils import AverageMeter
from pytorch.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback 


# train one epoch
def train_epoch(segmentation_module, loader, optimizers, history, epoch, cfg, 
                writer, epoch_iters, channels, patch_size, disp_iter, lr_encoder, lr_decoder):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    
    iterator = iter(loader)

    segmentation_module.train(not cfg['TRAIN']['fix_bn']) #i.e. True

    # main loop
    tic = time.time()
    for i in range(epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

# =============================================================================
#         # adjust learning rate # TODO turn off if you want stable lr.
#         cur_iter = i + (epoch - 1) * cfg['TRAIN']['epoch_iters']
#         adjust_learning_rate(optimizers, cur_iter, cfg, lr_encoder, lr_decoder)
# =============================================================================

        
        # get the data in correct format
        batch_images = torch.zeros(
                len(batch_data), 
                len(channels), 
                patch_size*3, 
                patch_size*3)
       
        if cfg['DATASET']['segm_downsampling_rate'] == 0:
            batch_segms = torch.zeros(
                    len(batch_data), 
                    patch_size*3, 
                    patch_size*3).long()
        else:
            batch_segms = torch.zeros(
                    len(batch_data), 
                    patch_size*3//cfg['DATASET']['segm_downsampling_rate'], 
                    patch_size*3//cfg['DATASET']['segm_downsampling_rate']).long()
        
        for j, bd in enumerate(batch_data): 
            batch_images[j] = bd['img_data']
            batch_segms[j] = bd['seg_label']
            
        batch_data = {'img_data': batch_images.cuda(), 'seg_label':batch_segms.cuda()}

        # forward pass
        #for HRNET # TODO: first one for HR model with acc/loss only on inner patch, second for smallmodel (or model without downsampling)
        #loss, acc = segmentation_module(batch_data, patch_size = int(patch_size/4)) 
        #loss, acc = segmentation_module(batch_data, patch_size = int(patch_size)) 
        loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg['TRAIN']['running_lr_encoder'], cfg['TRAIN']['running_lr_decoder'],
                          ave_acc.average(), ave_total_loss.average()))

        fractional_epoch = epoch - 1 + 1. * i / epoch_iters
        history['train']['epoch'].append(fractional_epoch)
        history['train']['loss'].append(loss.data.item())
        history['train']['acc'].append(acc.data.item())
        
    
    writer.add_scalar('Train/Loss', ave_total_loss.average(), epoch)
    writer.add_scalar('Train/Acc', ave_acc.average(), epoch)


# validate one epoch
def validate(segmentation_module, loader, optimizers, history, epoch, cfg, writer,val_epoch_iters, channels, patch_size):
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()
    
    iterator = iter(loader)

    # main loop
    tic = time.time()
    for i in range(val_epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        
        # get the data in correct format
        batch_images = torch.zeros(
                len(batch_data), 
                len(channels), 
                patch_size*3, 
                patch_size*3)
        
        if cfg['DATASET']['segm_downsampling_rate'] == 0:
            batch_segms = torch.zeros(
                    len(batch_data), 
                    patch_size*3, 
                    patch_size*3).long()
        else:
            batch_segms = torch.zeros(
                len(batch_data), 
                patch_size*3//cfg['DATASET']['segm_downsampling_rate'], 
                patch_size*3//cfg['DATASET']['segm_downsampling_rate']).long()
        
        for j, bd in enumerate(batch_data): 
            batch_images[j] = bd['img_data']
            batch_segms[j] = bd['seg_label']
            
        batch_data = {'img_data': batch_images.cuda(), 'seg_label':batch_segms.cuda()}
      
        with torch.no_grad():
            # forward pass
            loss, acc = segmentation_module(batch_data, patch_size = int(patch_size/4))
        
        loss = loss.mean()
        acc = acc.mean()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)


        # measure elapsed time
        time_meter.update(time.time() - tic)
        tic = time.time()

        

        # calculate accuracy, and display
        fractional_epoch = epoch - 1 + 1. * i / val_epoch_iters
        history['val']['epoch'].append(fractional_epoch)
        history['val']['loss'].append(loss.data.item())
        history['val']['acc'].append(acc.data.item())
        
    print('Epoch: [{}], Time: {:.2f}, ' 
          'Val_Accuracy: {:4.2f}, Val_Loss: {:.6f}'
          .format(epoch, time_meter.average(),
                  ave_acc.average(), ave_total_loss.average()))
    writer.add_scalar('Val/Loss', ave_total_loss.average(), epoch)
    writer.add_scalar('Val/Acc', ave_acc.average(), epoch)

    return ave_total_loss.average()

def checkpoint(nets, history, cfg, epoch, DIR):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(DIR, epoch))



def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


# =============================================================================
# def create_optimizers(nets, cfg, lr_encoder, lr_decoder, weight_decay):
#     (net_encoder, net_decoder, crit) = nets
#     optimizer_encoder = torch.optim.SGD(
#         group_weight(net_encoder),
#         lr=lr_encoder,
#         momentum=cfg['TRAIN']['beta1'],
#         weight_decay=weight_decay)
#     optimizer_decoder = torch.optim.SGD(
#         group_weight(net_decoder),
#         lr=lr_decoder,
#         momentum=cfg['TRAIN']['beta1'],
#         weight_decay=weight_decay)
#     return (optimizer_encoder, optimizer_decoder)
# =============================================================================

# TODO: used in smallnet
def create_optimizers(nets, cfg, lr_encoder, lr_decoder, weight_decay):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.Adam(
        net_encoder.parameters(),
        lr=lr_encoder,
        weight_decay=weight_decay)
    optimizer_decoder = torch.optim.Adam(
        net_decoder.parameters(),
        lr=lr_decoder,
        weight_decay=weight_decay)
    return (optimizer_encoder, optimizer_decoder)


# TODO: use if you want to adjust lr every batch
# =============================================================================
# def adjust_learning_rate(optimizers, cur_iter, cfg, lr_encoder, lr_decoder):
#     scale_running_lr = ((1. - float(cur_iter) / cfg['TRAIN']['max_iters']) ** cfg['TRAIN']['lr_pow'])
#     cfg['TRAIN']['running_lr_encoder'] = lr_encoder * scale_running_lr
#     cfg['TRAIN']['running_lr_decoder'] = lr_decoder * scale_running_lr
# 
#     (optimizer_encoder, optimizer_decoder) = optimizers
#     for param_group in optimizer_encoder.param_groups:
#         param_group['lr'] = cfg['TRAIN']['running_lr_encoder']
#     for param_group in optimizer_decoder.param_groups:
#         param_group['lr'] = cfg['TRAIN']['running_lr_decoder']
# =============================================================================


# TODO: used to adjust learning rate manually
# =============================================================================
# def adjust_learning_rate(optimizers):
# 
#     (optimizer_encoder, optimizer_decoder) = optimizers
#     for param_group in optimizer_encoder.param_groups:
#         param_group['lr'] = param_group['lr']*0.5
#         print('updated lr to: {}',param_group['lr'])
#     for param_group in optimizer_decoder.param_groups:
#         param_group['lr'] = param_group['lr']*0.5
# =============================================================================
     

#%%
def train(cfg, gpus, patchespath, index_train, index_val, patch_size, tb_DIR, 
          arch_encoder, arch_decoder,fc_dim, channels, num_class, pretrained,
          batch_size_per_gpu, val_batch_size, workers, val_workers, start_epoch,
          num_epoch, lr_encoder, lr_decoder, weight_decay, DIR, val_epoch_iters,
          epoch_iters, disp_iter):
    
    # init
    loss = cfg['TRAIN']['loss']
    counter = cfg['TRAIN']['counter']
    early_stopping = 0
    
    writer = SummaryWriter(logdir=tb_DIR)
    print("tensorboard --logdir {}".format(tb_DIR))
    
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=arch_encoder.lower(),
        fc_dim=fc_dim,
        n_channels= len(channels),
        weights=cfg['MODEL']['weights_encoder'],
        pretrained=pretrained)
    net_decoder = ModelBuilder.build_decoder(
        arch=arch_decoder.lower(),
        fc_dim=fc_dim,
        num_class=num_class,
        weights=cfg['MODEL']['weights_decoder'])     
    
    crit = nn.NLLLoss(ignore_index=-1)

    if arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg['TRAIN']['deep_sup_scale'])
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # Datasets
    dataset_train = TrainDataset(
            data_path = patchespath, 
            indices = index_train, 
            y_downsampling_rate= cfg['DATASET']['segm_downsampling_rate'], 
            channels=channels, 
            n_classes = num_class, 
            patch_size_padded=patch_size*3, 
            augment=True) 
    
    dataset_val = TrainDataset(
            data_path = patchespath, 
            indices = index_val, 
            y_downsampling_rate= cfg['DATASET']['segm_downsampling_rate'], 
            channels=channels, 
            n_classes = num_class, 
            patch_size_padded=patch_size*3, 
            augment=False) 
    
    # data loaders
    loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size = batch_size_per_gpu,
            shuffle = True,
            collate_fn=user_scattered_collate,
            num_workers= workers,
            drop_last=True,
            pin_memory=True)

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size= val_batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers= val_workers,
        drop_last=True)
    
    # load nets into gpu
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg, lr_encoder, lr_decoder, weight_decay)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'val':{'epoch': [], 'loss': [], 'acc': []}}

    
    for epoch in range(start_epoch, num_epoch): 
        train_epoch(segmentation_module, loader_train, optimizers, history, epoch+1, cfg, 
                    writer,epoch_iters, channels, patch_size, disp_iter, lr_encoder, lr_decoder)
        loss_now = validate(segmentation_module, loader_val, optimizers, history, epoch+1, cfg, writer,val_epoch_iters,channels, patch_size) #TODO turn on for validation!
        checkpoint(nets, history, cfg, epoch+1, DIR) 
 
# TODO used for early stopping in CV       
# =============================================================================
#         # early stopping if no improvement in val loss for 15 epochs
#         if loss_now < loss:
#             loss = loss_now
#             counter = 0
#         else: 
#             counter += 1
#         if counter > 15:
#             break
# =============================================================================
        
       
# TODO used when adjusting lr manually        
# =============================================================================
#         # decrease lr if 5x no improvement in val loss, do this max 6x
#         if loss_now < loss:
#             loss = loss_now
#         else: 
#             counter += 1
#         if counter == 5:
#             early_stopping +=1
#             if early_stopping < 7:
#                 adjust_learning_rate(optimizers)
#                 counter = 0
#             else:
#                 break
# =============================================================================

    print('Training Done!')
    writer.close()



