#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:29:59 2019

@author: MLeenstra
"""

compute = "personal"
patch_size=32
patch_size_padded = patch_size*3
classes = [638,659,654,650,770]
channels = [0,1,2,3] 
n_channels = len(channels)
n_classes = len(classes)
resolution = 20 #1 for 1m; 20 for 20cm


# set paths according to computer
if compute == "optimus":
    inputpath='/home/media/marrit/elements/Data/'
    dtmpath = 'home/media/marrit/marrit/GrasslandProject/DTM/'
    coordspath='/home/marrit/GrasslandProject/input/files/'
    coordsfilename= 'patches.csv'
    patchespath = '/media/marrit/marrit/GrasslandProject/Patches/'
    tiles_cv_file = '/home/marrit/GrasslandProject/input/files/folders_cv.npy'
    tiles_test_file = '/home/marrit/GrasslandProject/input/files/folders_test.npy'
    model_savepath = '/home/marrit/GrasslandProject/output/models/'
elif compute == "personal":
    inputpath='/media/cordolo/elements/Data/'
    dtmpath = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/DTM/'
    coordspath='/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/'
    coordsfilename= 'patches.csv'
    patchespath = '/media/cordolo/marrit/GrasslandProject/Patches/'
    tiles_cv_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/folders_cv.npy'
    tiles_test_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/folders_test.npy'
    model_savepath = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/output/models/'

patchespath = patchespath + 'res_'+str(resolution) + '/'



#%% Generate dataset
"""
Generate the dataset by extracting patches from the tiles. These patches are
divided over a training and test set, based on the original tiles.
"""

from dataset import tile_to_patch, train_test_split

# init
patches_per_tile = 200
n_test_tiles = 97 # to use all folders when using 4-fold CV


# generate dataset (patches) at 20cm resolution
tile_to_patch(inputpath, coordspath, coordsfilename, dtmpath, patchespath, 
              patch_size, patches_per_tile, classes, resolution)


# split dataset in train + testset
train_test_split(coordspath + coordsfilename, tiles_cv_file, tiles_test_file, n_test_tiles)

#%%
"""
Check some of the generated patches
"""
from dataset import get_patches
from plots import plot_patches
from plots import plot_random_patches

patches, gt = get_patches(patchespath, [0,1,2,3,4,5], patch_size_padded, [0,1,2,3,4])

plot_patches(patches, gt, 6)

plot_random_patches(patchespath, 6) # full available patch_size (i.e. 480x480)


#%% Initialize
"""
Initialize model
"""
from models import models
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

# init 
patch_size_padded = patch_size*3
image_size = (patch_size_padded, patch_size_padded, n_channels)

# model parameters
modelname ="UNet"
args = {
  'dropout_rate': 0.,
  'weight_decay':0., 
  'batch_momentum':0.9
}
lr= 1e-3
epsilon=1e-8

# init model
model = models.all_models[modelname](image_size, n_classes, **args)
optimizer = optimizers.Adam(lr, epsilon)
model.compile(optimizer=optimizer ,loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])


#%% Train
""" 
Train model
"""
from dataset import train_val_split
from datagenerator import DataGen
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py

output_model_path = model_savepath + modelname + '_res:' + str(resolution) +'.{epoch:02d}-{val_loss:.4f}.hdf5'

# init
folds = 4
kth_fold = 0

# training setup
batch_size = 128
epochs=30
checkpoint = ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True, mode='min')
stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')

# load train and validation set
index_train, index_val = train_val_split(tiles_cv_file, coordspath + coordsfilename, folds, kth_fold) 
n_patches_train = 1000#len(index_train)
n_patches_val = 1000#len(index_val)

# init datagenerator
train_generator = DataGen(data_path=patchespath, n_patches = n_patches_train, shuffle=True, 
                augment=True, indices=index_train , batch_size=batch_size, 
                patch_size=patch_size_padded, n_classes=n_classes, channels=channels)
val_generator = DataGen(data_path = patchespath, n_patches = n_patches_val, shuffle=True, 
                augment=False, indices=index_val , batch_size=batch_size, 
                patch_size=patch_size_padded, n_classes=n_classes, channels=channels)

# run
result = model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=epochs,callbacks=[checkpoint,stop]) 


#%% Test
"""
Test the model on independent test set
"""
from dataset import load_test_indices
from datagenerator import DataGen
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

# init
batch_size = 32
model_file = 'unet.02-1.5225.hdf5'
model_path = model_savepath + model_file

index_test = load_test_indices(tiles_test_file, coordspath + coordsfilename)
n_patches_test = len(index_test)

test_generator = DataGen(data_path=patchespath, n_patches = n_patches_test, shuffle=True, 
                augment=False, indices=index_test , batch_size=batch_size, 
                patch_size=patch_size_padded, n_classes=n_classes, channels=channels)

model = load_model(model_path)

evaluate = model.evaluate_generator(generator=test_generator,steps = 4)
print('test loss, test acc:', evaluate)

#%% Predict
"""
Predict patches
"""

from dataset import load_test_indices, get_patches
from tensorflow.keras.models import load_model
from plots import plot_predicted_patches#, plot_predicted_probabilities
import numpy as np
#%matplotlib qt

# init
patch_size_padded = patch_size*3
#patch_size_padded = 240
n = 6
model_file = 'unet.02-1.5225.hdf5'
model_path = model_savepath + model_file

index_test = load_test_indices(tiles_cv_file, coordspath + coordsfilename)
index_predict = np.random.choice(index_test, n)
index_predict = [k for k in index_predict]

patches, gt_patches = get_patches(patchespath, index_predict, patch_size_padded, channels)

model = load_model(model_path)

predictions = model.predict(patches)

#plot_predicted_probabilities(predictions[:6], gt_patches, 5)
plot_predicted_patches(predictions[:6], gt_patches)

#%%
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