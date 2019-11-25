#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:29:59 2019

@author: MLeenstra
"""

compute = "optimus"
patch_size=32
patch_size_padded = patch_size*3
classes = [638,659,654,650,770]
class_names = ["tara0", "tara20", "tara50", "forest","non-cultivable"]
channels = [0,1,2,3,4] # 0:R, 1:G, 2:B, 3:NIR, 4:DEM
n_channels = len(channels)
n_classes = 5
resolution = 1 #1 for 1m; 20 for 20cm
if resolution == 1:
    max_size = 96
elif resolution == 20:
    max_size = 480


# set paths according to computer
if compute == "optimus":
    inputpath='/marrit2/Data/' # path to tiles
    dtmpath = 'marrit1/GrasslandProject/DTM/' # path to dtm
    coordspath='/data3/marrit/GrasslandProject/input/files/' # path to coords-file
    coordsfilename = 'patches_grid_clean.csv'
    patchespath = '/marrit1/GrasslandProject/PatchesNew/' # path to patches
    tiles_cv_file = '/data3/marrit/GrasslandProject/input/files/folders_cv_grid.npy' # tiles used for cv
    tiles_test_file = '/data3/marrit/GrasslandProject/input/files/folders_test_grid.npy' # tiles used for test
    model_savepath = '/data3/marrit/GrasslandProject/output/models/' 
    log_dir = '/data3/marrit/GrasslandProject/output/logs/scalars/'
    results_dir = '/data3/marrit/GrasslandProject/output/results/' # path to folder to save results of validation/test
elif compute == "personal":
    inputpath='/media/cordolo/elements/Data/'
    dtmpath = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/DTM/'
    coordspath='/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/'
    coordsfilename= 'patches.csv'
    patchespath = '/media/cordolo/marrit/GrasslandProject/Patches/'
    tiles_cv_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/folders_cv.npy'
    tiles_test_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/folders_test.npy'
    model_savepath = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/Output/Models/'
    log_dir = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/Output/logs/scalars/'


patchespath = patchespath + 'res_' + str(resolution) + '/'

#%% Generate old dataset
"""
Generate the dataset by extracting patches from the tiles. These patches are
divided over a training and test set, based on the original tiles.
"""
from dataset_old import tile_to_patch
from dataset import train_test_split, csv_to_patch, count_classes

# init
patches_per_tile = 200
n_test_tiles = 97 # to use all folders when using 4-fold CV


# generate dataset (patches) at 20cm resolution
tile_to_patch(inputpath, coordspath, coordsfilename, dtmpath, patchespath, 
              patch_size, patches_per_tile, classes, 20)

# generate dataset (patches) at 1m resolution 
coordsfile = coordspath+coordsfilename
csv_to_patch(inputpath, dtmpath, patchespath, coordsfile, patch_size, classes, 1)

# split dataset in train + testset
train_test_split(coordspath + coordsfilename, tiles_cv_file, tiles_test_file, n_test_tiles)

# check if dataset is balanced
percentage = count_classes(patchespath, coordsfile, class_names, resolution)

#%% Generate new dataset
"""
Generate new dataset based on grid to avoid overlapping patches
"""
from dataset import tile_to_csv_grid, csv_to_patch, train_test_split

# find patches and save in csv
tile_to_csv_grid(inputpath, coordspath, coordsfilename, patch_size_padded)

# save patches on disk
coordsfile = coordspath+coordsfilename
tile_to_csv_grid(
        inputpath='/marrit2/Data/', 
        coordspath='/data3/marrit/GrasslandProject/input/files/',
        coordsfilename='patches_grid.csv', 
        patch_size_padded=480,
        classes=[638,659,654,650,770])
csv_to_patch(inputpath, dtmpath, patchespath='/marrit1/GrasslandProject/PatchesNew/res_20/', 
        coordsfile='/data3/marrit/GrasslandProject/input/files/patches_grid.csv', 
        patch_size=160, classes=[638,659,654,650,770], resolution=20)
csv_to_patch(inputpath, dtmpath, patchespath='/marrit1/GrasslandProject/PatchesNew/res_1/', 
        coordsfile='/data3/marrit/GrasslandProject/input/files/patches_grid.csv', 
        patch_size=32, classes=[638,659,654,650,770], resolution=1)

# split dataset in train + testset
train_test_split(coordspath + coordsfilename, tiles_cv_file, tiles_test_file, n_test_tiles=62)

#%% create dataset with larger patches
"""
Generate new dataset based on grid but with larger patches (res 1m, patch_size 480)
"""
from dataset import csv_to_patch, tile_to_csv_grid, enlarge_patches
from dataset import train_test_split, count_classes
from dataset import plot_patches_on_tile
from plots import plot_random_patches

# init
patch_size = 160
patch_size_padded = 480
coordsfile= '/data3/marrit/GrasslandProject/input/files/patches_large.csv'
tiles_cv_file= '/data3/marrit/GrasslandProject/input/files/folders_cv_large.npy'
tiles_test_file= '/data3/marrit/GrasslandProject/input/files/folders_test_large.npy'
patchespath= '/marrit1/GrasslandProject/PatchesLarge/res_1/'

# get starting location of patches: 
# without padding, thus use patch_size instead of patch_size_padded
tile_to_csv_grid(
        inputpath=inputpath, coordspath=coordsfile.split('patches')[0], 
        coordsfilename=coordsfile.split('/')[-1], 
        patch_size_padded=patch_size,  classes=classes)

# add padding to patches (padding == patch_sizes)
enlarge_patches(coordsfile,patch_size, patch_size)

# check with a plot
plot_patches_on_tile(coordsfile, inputpath, '081143w', patch_size*3)

# CAUTION: FOR 1M THE PATCH_SIZE AND PATCH_SIZE_PADDED SHOULD BE UPDATED!!!
patch_size = 32
patch_size_padded = 96
csv_to_patch(inputpath, dtmpath=dtmpath, patchespath=patchespath, 
        coordsfile=coordsfile, patch_size=patch_size, classes=classes, 
        resolution=resolution)

# check
plot_random_patches(patchespath, 5, classes, class_names)

# split dataset in train + testset
train_test_split(coordsfile,tiles_cv_file,tiles_test_file, n_test_tiles=50)

#%%
"""
Check some of the generated patches
"""
from plots import plot_patches_on_tile, plot_random_patches
import numpy as np

# patches
plot_random_patches(patchespath, 6, class_names, classes) # full available patch_size (e.g. 480x480 for 20x20cm resolution)

# patches on tiles
cv_tiles = np.load(tiles_cv_file, allow_pickle=True)
tiles = np.random.choice(cv_tiles,10,False)  
for i in range(5):
    tile = tiles[i]
    plot_patches_on_tile(coordspath+coordsfilename, inputpath, tile, patch_size_padded)

    
#%% Initialize
"""
Initialize model for Keras
"""
from models import models
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

# init 
image_size = (patch_size_padded, patch_size_padded, n_channels)

# model parameters
modelname ="UNet"
args = {
  'dropout_rate': 0.,
  'weight_decay':0., 
  'batch_momentum':0.9
}
lr= 1e-4
epsilon=1e-8

# init model
model = models.all_models[modelname](image_size, n_classes, **args)
optimizer = optimizers.Adam(lr, epsilon)
model.compile(optimizer=optimizer ,loss='categorical_crossentropy', 
              metrics=[metrics.categorical_accuracy])

#%% Train
""" 
Train model keras
"""
from time import localtime, strftime
from matplotlib import pyplot as plt
from dataset import train_val_split, train_test_split_random, train_val_split_random
from datagenerator import DataGen
from plots import plot_history
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import h5py

# init
folds = 5 
kth_fold = 0
output_model_path = model_savepath + modelname + \
    '_{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime())) + \
    '_res:' + str(resolution) +"_fold:" + str(kth_fold) + "_channels:"+ str(channels)+ "_epoch:.{epoch:02d}"+"_valloss:.{val_loss:.4f}.hdf5"

# training setup
batch_size = 64
epochs=200
checkpoint = ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True, mode='min')
stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, mode='min')
tensorboard = TensorBoard(log_dir = log_dir+modelname+'_{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime())))
pretrained_resnet50 = False
if modelname in ("Pretrained_ResNet", "Pretrained_VGG16_T", "pretrained_VGG16"):
    pretrained_resnet50 = True

# load train and validation set
index_train, index_val = train_val_split(tiles_cv_file, coordspath + coordsfilename, folds, kth_fold)
 
n_patches_train = len(index_train)
n_patches_val = len(index_val)

# init datagenerator
train_generator = DataGen(data_path=patchespath, n_patches = n_patches_train, 
                shuffle=True, augment=True, indices=index_train , 
                batch_size=batch_size, patch_size=patch_size_padded, 
                n_classes=n_classes, channels=channels, max_size=max_size,pretrained_resnet50=pretrained_resnet50)
val_generator = DataGen(data_path = patchespath, n_patches = n_patches_val, 
                shuffle=True, augment=False, indices=index_val , 
                batch_size=batch_size, patch_size=patch_size_padded, 
                n_classes=n_classes, channels=channels, max_size=max_size,pretrained_resnet50=pretrained_resnet50)

# run
result = model.fit_generator(generator=train_generator, validation_data=val_generator, 
                epochs=epochs,callbacks=[checkpoint,tensorboard, stop]) 

# plot training history
plot_history(result)


#%% Test
"""
Test the keras model on independent test set
"""
from dataset import load_test_indices
from datagenerator import DataGen
from tensorflow.keras.models import load_model
from models.BilinearUpSampling import BilinearUpSampling2D

# init
batch_size = 125

# load model
model_file = 'UNet_01102019_17:30:35_res:1_epoch:.60_valloss:.0.7414.hdf5'
model_path = model_savepath + model_file
model = load_model(model_path, custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D})

# get indices
index_test = load_test_indices(tiles_test_file, coordspath + coordsfilename)
n_patches_test = len(index_test)

test_generator = DataGen(data_path=patchespath, n_patches = n_patches_test, shuffle=True, 
                augment=False, indices=index_test , batch_size=batch_size, 
                patch_size=patch_size_padded, n_classes=n_classes, channels=channels, max_size=max_size,
                pretrained_resnet50=False)

evaluate = model.evaluate_generator(generator=test_generator)
print('test loss, test acc:', evaluate)

#%% 
""" 
Test keras model with loop as in pytorch
"""
import os
from dataset import load_test_indices, train_val_split
from evaluate_testset import test

# use gpu 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

#folds = 5
#kth_fold = 4
model_file = 'UNet_06112019_18:37:39_res:1_fold:full_channels:[0, 1, 2]__epoch:.20.hdf5'
channels = [0,1,2] 

# load model
model_path = model_savepath + model_file
results_path = results_dir + model_file +'/061032w/'
batch_size=1


if not os.path.isdir(results_path):
    os.makedirs(results_path)

# get indices 
index_test = load_test_indices(tiles_test_file, coordspath + coordsfilename)
#index_train, index_test = train_val_split(tiles_cv_file, coordspath + coordsfilename, folds, kth_fold) # test is validation set  
print('#samples: {}'.format(len(index_test)))

test(classes, index_test, patchespath, patch_size, patch_size_padded, 
     max_size, channels, resolution, model_path, results_path, class_names, visualize=True)

#%% Train model pytorch
"""
Train Pytorch model.
"""
# libs
import os
import random
from time import localtime, strftime
import torch
from dataset import train_val_split
from pytorch.train import train

# libs
import yaml

# load configuration file
cfg_file = 'config/grassland-hrnetv2_dataset-grid.yaml'
with open(cfg_file, 'r') as ymlfile: 
    cfg = yaml.load(ymlfile)
    
# info
#print("Loaded configuration file {}".format(cfg))
cfg

# init specific for pytorch
num_class= 5
arch_encoder= 'hrnetv2'
arch_decoder= 'c1'
fc_dim= 720
pretrained = False
batch_size_per_gpu= 64
num_epoch= 250
start_epoch= 0
epoch_iters= 1555
lr_encoder= 0.0001
lr_decoder= 0.0001
weight_decay= 0.0001
workers= 2
disp_iter= 200
seed= 304
kth_fold= 0
val_batch_size= 32
val_epoch_iters= 406
val_workers= 2

DIR= "/data3/marrit/GrasslandProject/GrasslandProject_pytorch/ckpt/grassland-hrnetv2-c1"
tb_DIR= "/data3/marrit/GrasslandProject/GrasslandProject_pytorch/tensorboard/grassland-hrnetv2-c1"

# Start from checkpoint
if start_epoch > 0:
    cfg['MODEL']['weights_encoder'] = os.path.join(
        DIR, 'encoder_epoch_{}.pth'.format(start_epoch))
    cfg['MODEL']['weights_decoder'] = os.path.join(
        DIR, 'decoder_epoch_{}.pth'.format(start_epoch))
    assert os.path.exists(cfg['MODEL']['weights_encoder']) and \
        os.path.exists(cfg['MODEL']['weights_decoder']), "checkpoint does not exitst!"

# gpu ids
gpus = [0]
gpus = [int(x) for x in gpus]
num_gpus = len(gpus)
batch_size = num_gpus *batch_size_per_gpu

# init
cfg['TRAIN']['max_iters'] = epoch_iters * num_epoch
cfg['TRAIN']['running_lr_encoder'] = lr_encoder
cfg['TRAIN']['running_lr_decoder'] = lr_decoder

random.seed(seed)
torch.manual_seed(seed)

# get indices for train and validation
index_train, index_val = train_val_split(tiles_cv_file,coordsfile,  
        folds, kth_fold)

# Output directory
outputtime = '_{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime()))
DIR = DIR + outputtime
test_dir = DIR
tb_DIR = tb_DIR + "/" + outputtime + "/"
if not os.path.isdir(DIR):
    os.makedirs(DIR)
print("Outputing checkpoints to: {}".format(DIR))

train(cfg, gpus, patchespath, index_train, index_val, patch_size, tb_DIR, 
          arch_encoder, arch_decoder,fc_dim, channels, num_class, pretrained,
          batch_size_per_gpu, val_batch_size, workers, val_workers, start_epoch,
          num_epoch, lr_encoder, lr_decoder, weight_decay, DIR, val_epoch_iters,
          epoch_iters, disp_iter)

#%% test pytorch model
"""
Test pytorch model
"""

import os
from dataset import get_test_indices, train_val_split
from pytorch.evaluate_testset import test
from plots import visualize_results
import pandas as pd

gpu = 0
#kth_fold = 4
test_dir = "/data3/marrit/GrasslandProject/GrasslandProject_pytorch/ckpt/grassland-hrnetv2-c1_07112019_18:51:56_07112019_18:54:10"
checkpoint = "epoch_140.pth"
visualize = True
pretrained = False

# absolute paths of model weights
weights_encoder = os.path.join(
    test_dir, 'encoder_' + checkpoint)
weights_decoder = os.path.join(
   test_dir, 'decoder_' + checkpoint)
assert os.path.exists(weights_encoder) and \
    os.path.exists(weights_decoder), "checkpoint does not exitst!"

results_dir = os.path.join(test_dir, "result")
if not os.path.isdir(os.path.join(test_dir, "result")):
    os.makedirs(os.path.join(test_dir, "result"))

index_test = get_test_indices(tiles_test_file, coordsfile)
# get indices for test in this case the validation set is used for testing!
index_train, index_test = train_val_split(
        tiles_cv_file,  
        coordsfile,  
        folds, 
        kth_fold)


test(cfg, gpu, arch_encoder, arch_decoder, fc_dim, channels, weights_encoder, 
         weights_decoder, num_class, class_names, pretrained, patchespath, patch_size, 
         patch_size_padded, index_test, visualize, results_dir)

summary = pd.read_csv(os.path.join(test_dir, 'result','summary_results.csv'))
print(summary.iloc[0])

#%% 
"""
Predict a full tile
"""
from dataset import find_patches_options_final_predict, enlarge_patches, read_patch_1m
from osgeo import gdal
import pandas as pd
import numpy as np

tile = '061032w'
coordsfilename = 'patches_' + tile +'.csv'
coordsfile = coordspath + coordsfilename
# always extract patches at original resolution of 20cm:
patch_size = 160
patch_size_padded = 480
padding = 32*5

# get alle patches on tile (even those outside classification task)
tile_to_csv_grid(inputpath, coordspath, coordsfilename, patch_size_padded=160, classes=[638,659,654,650,770],tile=tile, fraction=0,final=True)
# add padding
enlarge_patches(coordsfile, padding, patch_size)

## back to 1m resolution
coords = pd.read_csv(coordsfile, sep=',')
patch_size = 32
patch_size_padded = int(patch_size * 3)
patchespath = '/marrit1/GrasslandProject/Patches_' + tile + '/res_' + str(resolution) + '/'
imagespath = patchespath + 'images/'
labelspath = patchespath + 'labels/'

if not os.path.isdir(patchespath):
    os.makedirs(patchespath) 
if not os.path.isdir(imagespath):
    os.makedirs(imagespath)    
if not os.path.isdir(labelspath):
    os.makedirs(labelspath)
    
# extract patches
for idx, d in enumerate(coords['tiles']):
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

# save also ground truth
gt_resamp = gt_1m.GetRasterBand(1).ReadAsArray()
gt_resamp = np.uint16(gt_resamp)
tile_rows = coords['tile_rows'][0]
tile_cols = coords['tile_cols'][0]

### PREDICTION PART ###
import os
from dataset import load_test_indices, train_val_split
from evaluate_testset import test

# use gpu 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

model_file = 'UNet_06112019_18:37:39_res:1_fold:full_channels:[0, 1, 2]__epoch:.20.hdf5'
channels = [0,1,2] 

# load model
model_path = model_savepath + model_file
results_path = results_dir + model_file +'/061032w/'
batch_size=1
visualize = True

if not os.path.isdir(results_path):
    os.makedirs(results_path)

# get indices
coords = pd.read_csv(coordsfile)
index_test = [idx for idx in coords.index]    
print('#samples: {}'.format(len(index_test)))

test(classes, index_test, patchespath, patch_size, patch_size_padded, 
     max_size, channels, resolution, model_path, results_path, class_names, visualize=visualize)

### Ensamble part ###
from dataset import get_predictions

indices = index_test
padding = 32
patch_size = 32
#results_path = '/data3/marrit/GrasslandProject/GrasslandProject_pytorch/ckpt/grassland-hrnetv2-c1_07112019_18:51:56_07112019_18:54:10/result/061032w'

predictions = get_predictions(results_path, indices, patch_size)
nrows = ((tile_rows - 2*padding*5)//(patch_size*5))
ncols = ((tile_cols - 2*padding*5)//(patch_size*5))

full_tile = np.zeros((nrows*patch_size,ncols*patch_size), dtype=np.uint8)

for i, prediction in enumerate(predictions):
    col = i % ncols
    row = i // ncols
    full_tile[row*patch_size:row*patch_size+patch_size, col*patch_size:col*patch_size+patch_size] = prediction

gt_ = gt_resamp[padding+4:gt_resamp.shape[0]-(padding+4),43:gt_resamp.shape[1]-43]
gt_mask = gt_ == 0
full_tile[gt_mask] = 5

##### PLOT  ####
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
colors = np.array([(194/256,230/256,153/256),(120/256,198/256,121/256),
    (49/256,163/256,84/256),(0/256,76/256,38/256),(229/256,224/256,204/256),(0,0,0)])
cmap = ListedColormap(colors)

# Create figure and axes
fig,ax = plt.subplots(figsize=(10,10))

im = ax.imshow(full_tile, cmap=cmap, vmin=0, vmax=5)
ax.axis('off')

np.save('/data3/marrit/GrasslandProject/output/images/prediction_tile_061032w_1m_HRNET.npy',full_tile)
full_tile = np.load('/data3/marrit/GrasslandProject/output/images/prediction_tile_061032w_1m_RGB.npy')
#plt.savefig('/data3/marrit/GrasslandProject/output/images/prediction_tile_061032w_UNet_08112019_092937_RGBND_epoch160.png')

##### PLOT GT ########
# take care of classes
gt_resamp[np.where(gt_resamp == 656)] = 650
gt_resamp[np.where(gt_resamp == 780)] = 770
gt_resamp[np.isin(gt_resamp, classes)==False] = 0
gt_resamp[gt_resamp==0] =5
gt_resamp[gt_resamp==638] = 0
gt_resamp[gt_resamp==659] = 1
gt_resamp[gt_resamp==654] = 2  
gt_resamp[gt_resamp==650] = 3
gt_resamp[gt_resamp==770] = 4


fig,ax = plt.subplots(figsize=(10,10))

im = ax.imshow(gt_resamp, cmap=cmap, vmin=0, vmax=5)
ax.axis('off')

np.save('/data3/marrit/GrasslandProject/output/images/gt_tile_061032w_1m.npy',gt_resamp)

len(gt_resamp[gt_resamp == 4])

#plt.savefig('/data3/marrit/GrasslandProject/output/images/gt_tile_061032w_1m_noaxis.png')


#### PLOT DIFF #####

# get ground truth in exactly same shape
gt_r = gt_resamp[padding+4:gt_resamp.shape[0]-(padding+4),43:gt_resamp.shape[1]-43]

# calculate difference
diff = np.abs(np.subtract(full_tile.astype(np.int8), gt_r.astype(np.int8))).astype(np.float)  #full_tile - gt_r
# mask areas not of interest
diff[gt_mask] = np.nan

# create a colorbar of reds with white at the start
import matplotlib as mpl
reds = plt.cm.get_cmap('Reds', 128)
redsW = reds(np.linspace(0,1,128))
redsW[0] = [1.,1.,1.,1.]
cmapWR = mpl.colors.ListedColormap(redsW, name='RedsW')
current_cmap = plt.cm.get_cmap(cmapWR)
current_cmap.set_bad(color='black')

#plot
fig,ax = plt.subplots(figsize=(10,7.2))
im = ax.imshow(diff, cmap = current_cmap)
ax.set_yticklabels([])
ax.set_xticklabels([])
cbar = plt.colorbar(im)
cbar.set_label('Error in number of classes')
#plt.savefig('/data3/marrit/GrasslandProject/output/images/error_prediction_tile_061032w_UNet_08112019_092937_RGBND_epoch160.png')


