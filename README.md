# GrasslandProject
Author: Marrit Leenstra<br/> 
Date: 03-09-2019

## Task
Land cover classification of alpine pastures (5 classes image segmentation)

## Original dataset 
* 549 aerial images. 
* Imagesize: (17351, 15300, 5) 
* Channels: RGB + NIR + DTM 
* Ground truth: densely labeled images in 5 classes. 

## Generated dataset 
* 102433 patches 
* Imagesize = (480,480,5)

The dataset is generated by:
1. **create_dataset_csv2.py**: save locations of top-left corner-pixel of patch. n = 200 per original image. The top-left corner pixels are randomly chosen, but divided over the different classes in the same percentages as the original image. Saved in csv.
2. **save_patches.py**: extract the patches from the original images, normalize values and save on disk. Ground truth is converted to one-hot labels. Uses 2 functions from utils.py: read_patch and to_categorical_classes. 
3. **train-test-split.py**: divide the dataset in a train and test part based on the original image from which the data is generated (to avoid possibly partly overlapping patches to be divided over training and testset). The trainingset is then divided over 4 parts that can be used for 4-fold cross-validation again based on the original image (train: n=63326 and validation n=20979). 

## Generator for batch training
**datagenerator.py**: takes the list of patch numbers and loads these in batches. 
Optional: 
* crop images
* exclude channels
* implement data-augmentation (fliplr, flipud and rotations). 
