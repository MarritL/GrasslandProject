#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:29:59 2019

@author: MLeenstra
"""


inputpath='/media/cordolo/elements/Data/'
dtmpath = 'home/media/marrit/marrit/GrasslandProject/DTM/'
coordspath='/home/marrit/GrasslandProject/input/files/'
coordsfilename= 'patches.csv'
patchespath = '/media/marrit/marrit/GrasslandProject/Patches/'
tiles_cv_file = '/home/marrit/GrasslandProject/input/files/folders_cv.npy'
tiles_test_file = '/home/marrit/GrasslandProject/input/files/folders_test.npy'
model_savepath = '/home/marrit/GrasslandProject/output/models/'

patch_size=80
classes = [638,659,654,650,770]
channels = [0,1,2,3] 
n_channels = len(channels)
n_classes = len(classes)


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
              patch_size, patches_per_tile, classes)


# split dataset in train + testset
train_test_split(coordspath + coordsfilename, tiles_cv_file, tiles_test_file, n_test_tiles)

#%% Initialize
"""
Initialize model
"""
from models.models import ModelsClass
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

# init 
patch_size_padded = patch_size*3
image_size = (patch_size_padded, patch_size_padded, n_channels)

# model parameters
lr= 1e-3
epsilon=1e-8
dropoutrate = 0.
modelname = "UNet"
reslution = "20cmx20cm"

# init model
models = ModelsClass(image_size, n_classes)
model = models.UNet(dropoutrate) #change model here
optimizer = optimizers.Adam(lr,epsilon)
model.compile(optimizer=optimizer ,loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])


#%% Train
""" 
Train model
"""
from dataset import train_val_split
from datagenerator import DataGen
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py

output_model_path = model_savepath + modelname + '_' + resolution +'.{epoch:02d}-{val_loss:.4f}.hdf5'

# init
folds = 4
kth_fold = 0

# training setup
batch_size = 64
epochs=30
checkpoint = ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True, mode='min')
stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')

# load train and validation set
index_train, index_val = train_val_split(tiles_cv_file, coordspath + coordsfilename, folds, kth_fold) 
n_patches_train = len(index_train)
n_patches_val = len(index_val)
n_patches_train = 1000
n_patches_val = 1000

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

# init
batch_size = 32
patch_size_padded = patch_size*3
#patch_size_padded = 240
model_file = 'unet_1epoch_lr1e03.h5'
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
from plots import plot_predicted_patches, plot_predicted_probabilities
import numpy as np
%matplotlib qt

# init
patch_size_padded = patch_size*3
#patch_size_padded = 240
n = 6
model_file = 'unet_1epoch_lr1e03.h5'
model_path = model_savepath + model_file

index_test = load_test_indices(tiles_cv_file, coordspath + coordsfilename)
index_predict = np.random.choice(index_test, n)
index_predict = [k for k in index_predict]

patches, gt_patches = get_patches(patchespath, index_predict, patch_size_padded, channels)

model = load_model(model_path)

predictions = model.predict(patches)

plot_predicted_probabilities(predictions[:6], gt_patches, 5)
plot_predicted_patches(predictions[:6], gt_patches)


