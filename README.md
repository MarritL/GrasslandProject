# GrasslandProject
Author: Marrit Leenstra<br/> 
Date: 03-09-2019

## Task
Land cover classification of alpine pastures (5 classes image segmentation)

## Documentation
### Original dataset 
* 549 aerial images. 
* Imagesize: (17351, 15300, 5) 
* Channels: RGB + NIR + DTM 
* Ground truth: densely labeled images in 5 classes. 

### Generated dataset 
* 102433 patches 
* Imagesize = (480,480,5)

**dataset.py**: contains all methods to generate the dataset and load patches
Mehodology to create dataset:
1. **tile_to_csv(...)**: save locations of top-left corner-pixel of patch. (In the project n = 200 per original image). The top-left corner pixels are randomly chosen, but divided over the different classes in the same percentages as the original image. Saved in csv.
2. **csv_to_patch(...)**: extract the patches from the original images, normalize values and save on disk. Ground truth is converted to one-hot labels. Uses 2 methods read_patch and to_categorical_classes. 
3. **train_test_split(...)**: divide the dataset in a train and test part based on the original images from which the data is generated (to avoid possibly partly overlapping patches to be divided over training and testset). 
4. **train_val_split(...)**: In the training phase the trainingset is divided over k - folds that can be used for cross-validation. This split is again based on the original image. (e.g. 4 folds: train: n=63326 and validation n=20979). 

### Models
**models.py**: models are organized in a class (ModelsClass), initialised with the imagesize and number of classes. Models can use blocks from **blocks.py**. A bilinearUpSampling class is saved separately in **BilinearUpSampling.py**.<\br>
At the moment the modelClass contains two models:
* A modified ResNet50
* UNet

### Generator for batch training
**datagenerator.py**: takes the list of patch numbers and loads these in batches.<br/>  
Optional: 
* crop images
* exclude channels
* implement data-augmentation (fliplr, flipud and rotations). 
