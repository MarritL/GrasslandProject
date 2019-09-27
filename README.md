# GrasslandProject
Author: Marrit Leenstra<br/> 
Date: 03-09-2019

## Task
Land cover classification of alpine pastures (5 classes image segmentation)

## Run
Use the **main.py** script to go through the pipeline of the project. 

## Documentation
### Original dataset 
* 549 aerial images (tiles). 
* Tilesize: (17351, 15300, 5) 
* Channels: RGB + NIR + DTM 
* Ground truth: densely labeled tiles in 5 classes. 

### Generated dataset 
* 81517 patches (out of 512 tiles, 37 tiles were not suitable) 
* Patchsize = (480,480,5)

**dataset.py**: contains all methods to generate the dataset and load patches<br/>
Mehodology to create dataset:
1. **tile_to_csv_grid(...)**: save locations of top-left corner-pixel of patch. (In the project n = 200 per original image). The top-left corner pixels are corner pixels from a grid layed over the tiles. This ensures there are no overlapping patches. Uses 1 method: find_patch_options().
2. **csv_to_patch(...)**: extract the patches from the original images, normalize values and save on disk. Ground truth is converted to one-hot labels. Uses 2 methods: to_categorical_classes and read_patch (read_patch has two variants: for 20cm resoltion and for 1m resolution). 
3. **train_test_split(...)**: divide the dataset in a train and test set based on the original tiles from which the data is generated.
4. **train_val_split(...)**: In the training phase the trainingset is divided over k - folds that can be used for cross-validation. This split is again based on the original tiles. <br/>
Additional training/validation/test split methods:
5. **train_val_split_random(...)**: divide the dataset in train, validation and test set in random manner. Takes no CV-folds and original tiles into account. Used only to compare datasets.
6. **train_val_split_subset(...)**: divide subset based on max. number of tiles in training and validation set. Used for quick testing of models. <br/>
Additional methods to load patches:
7. **load_test_indices(...)**: method returning the indices of the testpatches.
8. **get_patches(...)**: load patches in RAM, returns x (RGB, NIR, DTM) and y (ground truth).<br/>
Additional method to check class (in)balances:
9. **count_classes(...)**: count the number of pixels per class in all the patches, returns percentage per class.

**dataset_old.py**: contains all methods to generate the 'old' dataset (problem: overlapping patches). Number of patches in old dataset: 102433.<br/>
Mehodology to create dataset:
1. **tile_to_csv(...)**: save locations of top-left corner-pixel of patch. (In the project n = 200 per original image). The top-left corner pixels are randomly chosen, but divided over the different classes in the same percentages as the original image. Saved in csv. Uses 1 method: sample_patches_of_class().
2. **csv_to_patch(...)**: extract the patches from the original images, normalize values and save on disk. Ground truth is converted to one-hot labels. Uses 2 methods: to_categorical_classes and read_patch (read_patch has two variants: for 20cm resoltion and for 1m resolution).  
3. **train_test_split(...)**: divide the dataset in a train and test part based on the original images from which the data is generated (to avoid possibly partly overlapping patches to be divided over training and testset). 
4. **train_val_split(...)**: In the training phase the trainingset is divided over k - folds that can be used for cross-validation. This split is again based on the original image. (e.g. 4 folds: train: n=63326 and validation n=20979). 

### Models
**models.py**: script containing the intitialization of all models. Keras and Tensorflow are used to build the models. Models can use blocks from **blocks.py**. A bilinearUpSampling class is saved separately in **BilinearUpSampling.py**.<br/>
At the moment the modelClass contains five models:
* A modified ResNet50
* UNet
* Pretrained ResNet50 (has a problem with batch normalization)
* Pretrained VGG16 with bilinearUpSampling
* Pretrained VGG16 with transposed convolution

### Generator for batch training
**datagenerator.py**: takes the list of patch numbers and loads these in batches.<br/>  
Optional: 
* crop images
* exclude channels
* implement data-augmentation (fliplr, flipud and rotations). 
* perform different normalization as required by pretrained ResNet.

### Metrics
**metrics.py**: script with methods to compute validation metrics.<br/>
1. **compute_confusion_matrix**: computes confusion matrix based on given patches, optional plotting, normalization and user/producer accuracy. Calls: compute_user_producer_acc() and plot_confusion_matrix(). Has some problems when not all classes are present in patch and compute_user_producer_acc not working yet.
2.**compute_matthews_corrcoef**: computes the Matthews correlation coefficient based on given patches.

### Plots
**plots.py**: contains all plotting methods used in the project<br/>
1. **plot_random_patches**: plot random patches in 1st row and ground truth in second row.
2. **plot_predicted_paches**: plot predicted patches in 1st row and ground truth in second row.
3. **plot_patches**: plot chosen patches in 1st row and ground truth in second row.
4. **plot_predicted_probabilities**: plot predictionmaps (n_classes in rows, last row is ground truth), different columns represent different patches.
5. **plot_confustion_matrix**: method to plot confusion matrix already calculated in **compute_confustion_matrix**.
6. **umap_plot**: create a UMAP reduced dimensionaliy plot (calculation done in this method as well).
7. **pca_plot**: create PCA reduced dimensionality plot (calculation done done in this method as well).
8. **plot_patches_on_tile**: plot the patches on top of the original tile
9. **plot_patch_options**: plot the options for non-overlapping patches (grid-based) on top of tile. Based on method: **find_patches_options** in **dataset.py**.

