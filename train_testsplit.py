#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:45:11 2019

@author: MLeenstra
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
%matplotlib inline

save_path = '/media/cordolo/FREECOM HDD/GrasslandProject/Patches/'
images_path = save_path + 'images/'
labels_path = save_path + 'labels/'
coords_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/patches.csv'
folders_cv_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/folders_cv.npy'
folders_test_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Internship/folders_test.npy'
coords_df = pd.read_csv(coords_file, sep=',',header=None, names=['folder', 'row', 'col'])

# train / test split
folders = np.unique(coords_df.folder)
testfolders = np.random.choice(folders, 97, replace=False)
np.save(folders_test_file, testfolders)
coords_test = coords_df[np.isin(coords_df.folder, testfolders)]
folders = folders[np.isin(folders, testfolders) == False]

# shuffle
random.shuffle(folders)
folders_shuffled = folders
np.save(folders_cv_file, folders_shuffled)

# train / val split
cv = 0
testfolders = np.load(folders_test_file, allow_pickle=True)
folders = np.load(folders_cv_file, allow_pickle=True)
valfolders = folders_shuffled[cv*113:(cv+1)*113]
trainfolders = folders[np.isin(folders, valfolders) == False]

coords_test = coords_df[np.isin(coords_df.folder, testfolders)]
coords_val = coords_df[np.isin(coords_df.folder, valfolders)]
coords_train = coords_df[np.isin(coords_df.folder, trainfolders)]
