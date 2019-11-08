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
channels = [0,1,2] 
n_channels = len(channels)
n_classes = 5
resolution = 1 #1 for 1m; 20 for 20cm
if resolution == 1:
    max_size = 96
elif resolution == 20:
    max_size = 480


# set paths according to computer
if compute == "optimus":
    inputpath='/marrit2/Data/'
    dtmpath = 'marrit1/GrasslandProject/DTM/'
    coordspath='/data3/marrit/GrasslandProject/input/files/'
    #coordsfilename_old= 'patches.csv'
    #coordsfilename_grid = 'patches_grid.csv'
    coordsfilename = 'patches_grid_clean.csv'
    patchespath = '/marrit1/GrasslandProject/PatchesNew/'
    tiles_cv_file = '/data3/marrit/GrasslandProject/input/files/folders_cv_grid.npy'
    tiles_test_file = '/data3/marrit/GrasslandProject/input/files/folders_test_grid.npy'
    model_savepath = '/data3/marrit/GrasslandProject/output/models/'
    log_dir = '/data3/marrit/GrasslandProject/output/logs/scalars/'
    results_dir = '/data3/marrit/GrasslandProject/output/results/'
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
csv_to_patch(inputpath, dtmpath, patchespath='/marrit1/GrasslandProject/PatchesNew/res_20/', 
        coordsfile='/data3/marrit/GrasslandProject/input/files/patches_grid.csv', 
        patch_size=160, classes=[638,659,654,650,770], resolution=20)
csv_to_patch(inputpath, dtmpath, patchespath='/marrit1/GrasslandProject/PatchesNew/res_1/', 
        coordsfile='/data3/marrit/GrasslandProject/input/files/patches_grid.csv', 
        patch_size=32, classes=[638,659,654,650,770], resolution=1)


csv_to_patch(inputpath, dtmpath, patchespath='/marrit1/GrasslandProject/PatchesNew/test_res_20/', 
        coordsfile='/data3/marrit/GrasslandProject/input/files/patches_test_grid.csv', 
        patch_size=160, classes=[638,659,654,650,770], resolution=20)

# split dataset in train + testset
train_test_split(coordspath + coordsfilename, tiles_cv_file, tiles_test_file, n_test_tiles=62)

#%%
"""
Check some of the generated patches
"""
from dataset import get_patches
from plots import plot_patches, plot_patches_on_tile, plot_random_patches
import numpy as np

patches, gt = get_patches(patchespath, [100,107,200,300,400,500], patch_size_padded, [0,1,2,3,4], resolution)

plot_patches(patches, gt, 6)

plot_random_patches(patchespath, 6, class_names, classes) # full available patch_size (e.g. 480x480 for 20x20cm resolution)

cv_tiles = np.load(tiles_cv_file, allow_pickle=True)
tiles = np.random.choice(cv_tiles,10,False)
    
for i in range(5):
    tile = tiles[i]
    plot_patches_on_tile(coordspath+coordsfilename, inputpath, tile, patch_size_padded)
    
#%% Initialize
"""
Initialize model
"""
from models import models
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from metrics import weighted_categorical_crossentropy, dice_loss
#from keras.utils import multi_gpu_model


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
#model = multi_gpu_model(model, gpus=2)
optimizer = optimizers.Adam(lr, epsilon)
model.compile(optimizer=optimizer ,loss='categorical_crossentropy', 
              metrics=[metrics.categorical_accuracy])
# =============================================================================
# model.compile(optimizer=optimizer, loss = weighted_categorical_crossentropy(sample_weights),
#               metrics=[metrics.categorical_accuracy])
# =============================================================================

#%% Train with subset for model optimization
""" 
Train model with a smaller subset in order to optimize the hyperparameters
"""
from time import localtime, strftime
from matplotlib import pyplot as plt
from dataset import train_val_split_subset
from datagenerator import DataGen
from plots import plot_history
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import h5py

# init
folds = 7
kth_fold = 0
max_tiles = 500

output_model_path = model_savepath + modelname + \
    'subset_{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime())) + \
    '_res:' + str(resolution) +"_fold:"+str(kth_fold)+"_epoch:.{epoch:02d}"+"_valloss:.{val_loss:.4f}.hdf5"

# training setup
batch_size = 128
epochs=3
checkpoint = ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True, mode='min')
stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
tensorboard = TensorBoard(log_dir = log_dir+'scalars/'+'{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime())))

# load train and validation set
index_train, index_val = train_val_split_subset(tiles_cv_file, coordspath + coordsfilename , folds, kth_fold, max_tiles)
n_patches_train = 11*128#len(index_train)
n_patches_val = 5*128#len(index_val)

# init datagenerator
train_generator = DataGen(data_path=patchespath, n_patches = n_patches_train, 
                shuffle=True, augment=True, indices=index_train , 
                batch_size=batch_size, patch_size=patch_size_padded, 
                n_classes=n_classes, channels=channels, max_size=max_size)
val_generator = DataGen(data_path = patchespath, n_patches = n_patches_val, 
                shuffle=True, augment=False, indices=index_val , 
                batch_size=batch_size, patch_size=patch_size_padded, 
                n_classes=n_classes, channels=channels, max_size=max_size)

# run
result = model.fit_generator(generator=train_generator, validation_data=val_generator, 
                             epochs=epochs,callbacks=[checkpoint,stop,tensorboard]) 

# plot training history
plot_history(result)

#%% Train
""" 
Train model
"""
from time import localtime, strftime
from matplotlib import pyplot as plt
from dataset import train_val_split, train_test_split_random, train_val_split_random
from datagenerator import DataGen
from plots import plot_history
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
#from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import h5py

output_model_path = model_savepath + modelname + \
    '_{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime())) + \
    '_res:' + str(resolution) +"_fold:" + str(kth_fold) + "_channels:"+ str(channels)+ "_epoch:.{epoch:02d}"+"_valloss:.{val_loss:.4f}.hdf5"

# init
folds = 5 
kth_fold = 0

# training setup
batch_size = 128
epochs=200
checkpoint = ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True, mode='min')
stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, mode='min')
tensorboard = TensorBoard(log_dir = log_dir+modelname+'_{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime())))
pretrained_resnet50 = False
if modelname in ("Pretrained_ResNet", "Pretrained_VGG16_T", "pretrained_VGG16"):
    pretrained_resnet50 = True

# load train and validation set
index_train, index_val = train_val_split(tiles_cv_file, coordspath + coordsfilename, folds, kth_fold)
#index_train, index_val, index_test = train_test_split_random(coordspath + coordsfilename)
#index_train, index_val = train_val_split_random(tiles_cv_file, coordspath + coordsfilename, 15000)

 
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
Test the model on independent test set
"""
from dataset import load_test_indices
from datagenerator import DataGen
from tensorflow.keras.models import load_model
from models.BilinearUpSampling import BilinearUpSampling2D

# init
batch_size = 125
#model_file = 'UNet_01102019_17:30:35_res:1_epoch:.60_valloss:.0.7414.hdf5'

#model_path = model_savepath + model_file

index_test = load_test_indices(tiles_test_file, coordspath + coordsfilename)
n_patches_test = len(index_test)

test_generator = DataGen(data_path=patchespath, n_patches = n_patches_test, shuffle=True, 
                augment=False, indices=index_test , batch_size=batch_size, 
                patch_size=patch_size_padded, n_classes=n_classes, channels=channels, max_size=max_size,
                pretrained_resnet50=False)


#model = load_model(model_path, custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D})

evaluate = model.evaluate_generator(generator=test_generator)
print('test loss, test acc:', evaluate)

#%% 
""" 
Test the model with own loop as in pytorch
"""

import os
from dataset import load_test_indices, train_val_split
from evaluate_testset import test

# use gpu 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; 

folds = 5
kth_fold = 4
model_file = 'UNet_06112019_18:37:39_res:1_fold:full_channels:[0, 1, 2]_epoch:.120.hdf5'
channels = [0,1,2,3,4] 

# load model
model_path = model_savepath + model_file
results_path = results_dir + model_file +'/'
batch_size=1


if not os.path.isdir(results_path):
    os.makedirs(results_path)

# get indices
index_test = load_test_indices(tiles_test_file, coordspath + coordsfilename)
#index_train, index_test = train_val_split(tiles_cv_file, coordspath + coordsfilename, folds, kth_fold) # test is validation set  
print('#samples: {}'.format(len(index_test)))

test(classes, index_test, patchespath, patch_size, patch_size_padded, 
     max_size, channels, resolution, model_path, results_path, class_names, visualize=False)





#%% Predict
"""
Predict patches
"""
from dataset import load_test_indices, get_patches
from datagenerator import DataGen
from tensorflow.keras.models import load_model
from plots import plot_predicted_patches#, plot_confusion_matrix, plot_predicted_probabilities
from metrics import compute_confusion_matrix, compute_matthews_corrcoef
from dataset import to_categorical_classes
import numpy as np

# init
batch_size = 25
model_file = 'UNet_07102019_18:57:29_res:1_epoch:.39_valloss:.0.4390.hdf5'
model_path = model_savepath + model_file


index_test = load_test_indices(tiles_test_file, coordspath + coordsfilename)
n_patches_test = 1000#len(index_test)

test_generator = DataGen(data_path=patchespath, n_patches = n_patches_test, shuffle=False, 
                augment=False, indices=index_test , batch_size=batch_size, 
                patch_size=patch_size_padded, n_classes=n_classes, channels=channels, max_size=max_size,
                pretrained_resnet50=False)

model = load_model(model_path, custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D})

predictions = model.predict_generator(generator=test_generator)
patches, gt_patches = get_patches(patchespath, index_test[:1000], patch_size_padded, channels, resolution=resolution)

# binary case: only predict grass/no-grass
if n_classes == 2:
    gt_patches_full = gt_patches
    gt_patches = np.zeros((n_patches_test, patch_size_padded, patch_size_padded, 2), dtype=np.int8)
    for i in range(n_patches_test):
                gt_classes = np.argmax(gt_patches_full[i,], axis=2)
                gt_classes[(gt_classes == 1) | (gt_classes == 2)] = 0
                gt_classes[(gt_classes == 3) | (gt_classes == 4)] = 1
                gt_patches[i,] = to_categorical_classes(gt_classes, [0,1])  

# plots
plot_predicted_patches(predictions[100:200:10], gt_patches[100:200:10], patches[100:200:10])
mcc = compute_matthews_corrcoef(gt_patches, predictions[:1000])
mcc

cm = compute_confusion_matrix(gt_patches, predictions[:1000], classes=[0,1,2,3,4], class_names=class_names, user_producer=False, normalize=True, title='Confusion matrix {}'.format(modelname))
#cm = compute_confusion_matrix(gt_patches, predictions[:1000], classes=[0,1,2,3,4], class_names=class_names, user_producer=False, normalize=False)
cm

#%% Predict
"""
Predict small amount of patches
"""

from dataset import load_test_indices, get_patches
from tensorflow.keras.models import load_model
from plots import plot_predicted_patches, plot_confusion_matrix
from plots import plot_predicted_probabilities, plot_prediction_uncertainty
from models.BilinearUpSampling import BilinearUpSampling2D
import numpy as np
from metrics import compute_confusion_matrix, compute_matthews_corrcoef
from dataset import to_categorical_classes
from skimage.segmentation import find_boundaries
#%matplotlib qt

# init
n = 10
#model_file = 'UNet_04102019_15:19:54_res:1_epoch:.06_valloss:.0.0318.hdf5'
#model_path = model_savepath + model_file
#model = load_model(model_path, custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D, 'dice_loss':dice_loss})

# get n random patches from the testset
index_test = load_test_indices(tiles_test_file, coordspath + coordsfilename)
index_predict = np.random.choice(index_test, n)
index_predict = [11288, 10876,43659,18317, 25729,63385, 10073, 80736, 25751, 43424,9997, 73016,46201,48575,45781,7944]
index_predict = index_predict[10:15]
index_predict = [k for k in index_predict]
patches, gt_patches = get_patches(patchespath, index_predict, patch_size_padded, channels, resolution=resolution)

# binary case: only predict class boundaries
if n_classes == 2:
    gt_patches_full = gt_patches
    gt_patches = np.zeros((n, patch_size_padded, patch_size_padded, 2), dtype=np.int8)
    for i in range(n):
        gt_classes = np.argmax(gt_patches_full[i,], axis=2)
        edges = find_boundaries(gt_classes, mode='inner')
        gt_patches[i,] = to_categorical_classes(edges, [0,1])
# binary case: only predict grass/no-grass
if n_classes == 2:
    gt_patches_full = gt_patches
    gt_patches = np.zeros((n, patch_size_padded, patch_size_padded, 2), dtype=np.int8)
    for i in range(n):
                gt_classes = np.argmax(gt_patches_full[i,], axis=2)
                gt_classes[(gt_classes == 1) | (gt_classes == 2)] = 0
                gt_classes[(gt_classes == 3) | (gt_classes == 4)] = 1
                gt_patches[i,] = to_categorical_classes(gt_classes, [0,1])    

# predict
predictions = model.predict(patches)

# plots
#plot_predicted_probabilities(predictions[:6], gt_patches, n_classes, uncertainty=0.3)
plot_predicted_patches(predictions, gt_patches, patches)
#plot_prediction_uncertainty(predictions[:6], gt_patches, n_classes)

# metrics
mcc = compute_matthews_corrcoef(gt_patches, predictions)
mcc
cm = compute_confusion_matrix(gt_patches, predictions, classes=[0,1,2,3,4], class_names=class_names, user_producer=False, normalize=True)
cm

# =============================================================================
# plot_confusion_matrix(gt_patches, predictions, classes = [0,1,2,3,4], class_names=class_names, normalize=True,
#                       title='Normalized confusion matrix')
# =============================================================================

# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)
    
 	
# redefine model to output right after the first hidden layer
from tensorflow.keras import Model
import matplotlib.pyplot as plt

model2 = Model(inputs=model.inputs, outputs=model.layers[74].output)

featuremap = model2.predict(patches)

# plot all 64 maps in an 8x8 squares

#plot
rows = 4
cols = 4
    
# prepare
fig, ax = plt.subplots(rows,cols)
    
for i in range(featuremap.shape[3]):
    
    plt_im = featuremap[3][:, :, i].astype(np.float64)    
    
    # plot training image
    image = ax[int(i/4),int((i/4) % 1 * 4)].imshow(plt_im)


#%%
"""
some small test to get information about classes in tiles, work-in-progress
"""
# extact classes
tara0 = np.where(patch8000[:,:,5]==1)
tara20 = np.where(patch8000[:,:,6]==1)
tara50 = np.where(patch8000[:,:,7]==1)
woods = np.where(patch8000[:,:,8]==1)
no_colt = np.where(patch8000[:,:,9]==1)
patch8000_features = patch8000[:,:,0:5]
tara0_8000 = patch8000_features[tara0[0], tara0[1],]
tara20_8000 = patch8000_features[tara20[0], tara20[1],]
tara50_8000 = patch8000_features[tara50[0], tara50[1],]
woods_8000 = patch8000_features[woods[0], woods[1],]
no_colt_8000 = patch8000_features[no_colt[0], no_colt[1],]

# class statistics
features = np.arange(5)
classes = np.arange(5,10)
means = []
classesY= []
featY = []
stdY= []
colors = ['red', 'green', 'blue', 'darkred', 'yellow']
cmap = ListedColormap(colors)

for cl in classes:
    pix = np.where(patch8000[:,:,cl]==1)
    pix_cl =  patch8000_features[pix[0], pix[1],]
        
    for ft in features:
        ft_cl = pix_cl[:,ft]
        print("cl: " + str(cl-5) + " - ft: " + str(ft))
        print("mean: " + str(np.mean(ft_cl)))
        print("sd: " + str(np.std(ft_cl)))
        print("\n")
        means.append(ft_cl)
        classesY.append(cl)
        featY.append(ft)
        stdY.append(np.std(ft_cl))
 
plt.boxplot(means)  
plt.colorbar()     
tara0R = tara0_8000[:,0]
tara0G = tara0_8000[:,1]
tara0B = tara0_8000[:,2]
tara0NIR = tara0_8000[:,3]
tara0DEM = tara0_8000[:,4]

woodsR = woods_8000[:,0]
woodsG = woods_8000[:,1]
woods0B = woods_8000[:,2]
woodsNIR = woods_8000[:,3]
woodsDEM = woods_8000[:,4]


# edge detection    
from scipy import ndimage
input_result = np.argmax(gt[5], axis=2)
result = ndimage.sobel(input_result)
result[result !=0] = 1
plt.imshow(result)

#%% 
"""
Train models in a for loop
"""

"""
Initialize model
"""
# libs for initializing
from models import models
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
# libs for training
from time import localtime, strftime
from matplotlib import pyplot as plt
from dataset import train_val_split, train_test_split_random, train_val_split_random
from datagenerator import DataGen
from plots import plot_history
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import h5py

for channels in [[0,1,2],[0,1,2,3],[0,1,2,4],[0,1,2,3,4]]:    

    n_channels=len(channels)
        
    # init 
    image_size = (patch_size_padded, patch_size_padded, n_channels)
    
    # train different folds
    for kth_fold in [1,2,3,4]:
        
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
    
        output_model_path = model_savepath + modelname + \
            '_{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime())) + \
            '_res:' + str(resolution) +"_fold:" + str(kth_fold) + "_channels:"+ str(channels)+ "_epoch:.{epoch:02d}"+"_valloss:.{val_loss:.4f}.hdf5"
        print(output_model_path)
        
        # init
        folds = 5 
        
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
        #index_train, index_val, index_test = train_test_split_random(coordspath + coordsfilename)
        #index_train, index_val = train_val_split_random(tiles_cv_file, coordspath + coordsfilename, 15000)
        
         
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

#%% 
"""
Train models on full trainingset (train+val) in a for loop
"""
# libs for initializing
from models import models
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
# libs for training
from time import localtime, strftime
from matplotlib import pyplot as plt
from dataset import train_val_split, train_val_split_random
from datagenerator import DataGen
from plots import plot_history
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import h5py
import os

# use gpu 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; 

for channels in [[0,1,2,3,4]]:

    n_channels=len(channels)
        
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

    output_model_path = model_savepath + modelname + \
        '_{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime())) + \
        '_res:' + str(resolution) +"_fold:full" + "_channels:"+ str(channels)+ "_epoch:.{epoch:02d}.hdf5"
    print(output_model_path)
    
    # init
    folds = 1
    kth_fold = 0
    
    # training setup
    batch_size = 64
    epochs=200
    checkpoint = ModelCheckpoint(output_model_path,period=2)
    tensorboard = TensorBoard(log_dir = log_dir+modelname+'_{}'.format(strftime("%d%m%Y_%H:%M:%S", localtime())))
    pretrained_resnet50 = False
    if modelname in ("Pretrained_ResNet", "Pretrained_VGG16_T", "pretrained_VGG16"):
        pretrained_resnet50 = True
    
    # load train and validation set
    index_train, index_val = train_val_split(tiles_cv_file, coordspath + coordsfilename, folds, kth_fold)
    #index_train, index_val, index_test = train_test_split_random(coordspath + coordsfilename)
    #index_train, index_val = train_val_split_random(tiles_cv_file, coordspath + coordsfilename, 15000)
    index_train = index_val    
     
    n_patches_train = len(index_train)
    
    # init datagenerator
    train_generator = DataGen(data_path=patchespath, n_patches = n_patches_train, 
                    shuffle=True, augment=True, indices=index_train , 
                    batch_size=batch_size, patch_size=patch_size_padded, 
                    n_classes=n_classes, channels=channels, max_size=max_size,pretrained_resnet50=pretrained_resnet50)
    
    # run
    result = model.fit_generator(generator=train_generator, 
                    epochs=epochs,callbacks=[checkpoint,tensorboard]) 
    
    # plot training history
    plot_history(result)

