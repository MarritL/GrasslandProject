# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:00:36 2019

@author: lbergamasco + MLeenstra
"""

import numpy as np
import os,sys
from os.path import dirname
from os import listdir
import gc
from osgeo import gdal, ogr
import geopandas as gpd
from shapely.geometry import Polygon
import osr
import shutil
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import random
import csv
sys.path.append(dirname(__file__))

#################################################################################
def list_files(directory, extension):
    list = []
    for f in os.listdir(directory): 
        if f.endswith(extension):
            list.append(f)
    return list

def list_dir(directory):
    list = []
    for f in os.listdir(directory): 
        if os.path.isdir(directory+f):
            list.append(f)
    return list

def merge_dataset(path, output_filename):
    list_dataset = list_files(path,'.npy')
    dataset = None
    for d in list_dataset:
        if dataset is None:
            dataset = np.load(path + '/' + d)
        else:
            dataset = np.concatenate([dataset,np.load(path + '/' + d)], axis = 0)
            gc.collect()
    
    np.save(path + '/' + output_filename + '.npy', dataset)

####################################################################################################################
def rasterize_polygons(path, input_filename, output_filename, shadow_filename, no_data_value =-9999, fill_value=0):
    """Rasterize a shapefile, output is in integers (UInt16)
    
    Keyword arguments:
    path -- path to the files
    input_filename -- filename to the shapefile 
    output_filename -- filename for the output raster
    shadow_filename -- filename for the raster to borrow properties of (projection, celsize)
    fill_value -- value for blank no class cells (default: 0)
    no_data_value -- value for no_data class cells (default: -9999)
    
    To run e.g. 
    rasterize_polygons(path = 'Data/', input_filename = 'SHP/area_pascolo_fotointerprete_3044_clip.shp', 
                   output_filename = '/SHP/testraster.tif', shadow_filename='RGB/025164w_3044.tif')
    """
    
    # Filename of input OGR file
    vector_fn = path + input_filename
    #vector_fn = data_path + 'SHP/area_pascolo_fotointerprete_3044_clip.shp'
    
    # Filename of the raster Tiff that will be created
    raster_fn = path + output_filename
    #raster_fn = data_path + '/SHP/test2_tara_0.2.tif'
    
    # Filename for raster to borrow properties of
    shadow_fn = path + shadow_filename
    #path_RGB = data_path + 'RGB' + '/025164w_3044.tif' # testing with one image
    
    # Open the data source and read in the extent
    source_ds = ogr.Open(vector_fn)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    
    # get properties of shadow raster 
    shadow_im = gdal.Open(shadow_fn, gdal.GA_ReadOnly) 
    tileGeoTransformationParams = shadow_im.GetGeoTransform()
    projection = shadow_im.GetProjection()
    
    # get driver
    rasterDriver = gdal.GetDriverByName('GTiff')
    
    # Create the destination data source
    tempSource = rasterDriver.Create(raster_fn, 
                                     shadow_im.RasterXSize, 
                                     shadow_im.RasterYSize,
                                     1, #bandnumber
                                     gdal.GDT_UInt16)
    
    tempSource.SetGeoTransform(tileGeoTransformationParams)
    tempTile = tempSource.GetRasterBand(1)
    tempTile.Fill(fill_value)
    tempTile.SetNoDataValue(no_data_value)
    tempSource.SetProjection(projection)
    
    # rasterize
    gdal.RasterizeLayer(tempSource, # output to our new dataset
                        [1], #output to our new dataset's first band
                        source_layer, #rasterize this layer
                        options = ['ALL_TOUCHED=TRUE', # rasterize all pixels touched by polygons
                        'ATTRIBUTE=COD_RA'] # put raster values according to the 'COD_RA' field
                        )
    
    # clean
    source_ds = None
    tempSource = None
    shadow_fn = None  

################################################################################################
def resample_raster(path, input_filename, output_filename, xRes = 1, yRes = 1, resampleAlg=0):
    """ Resample a raster.
    
    Keyword arguments:
    path -- path to the files
    input_filename -- filename input file
    output_filename -- filename for the output raster
    xRes -- x-cellsize for the output raster
    yRes -- y-cellsize for the output raster
    resampleAlg -- 
        GRA_NearestNeighbour = 0
        GRA_Bilinear = 1
        GRA_Cubic = 2
        GRA_CubicSpline = 3
        GRA_Lanczos = 4
        GRA_Average = 5
        GRA_Mode = 6
        GRA_Gauss = 7 
        GRA_Max = 8
        GRA_Min = 9
        GRA_Med = 10
        GRA_Q1 = 11
        GRA_Q3 = 12
        
    To run e.g. 
    resample_raster('Data/', 'SHP/tara_20cm.tif', 'SHP/taratest.tif', 1,1,0)
    """
    input_raster = path + input_filename
    output_raster = path + output_filename
    src_ds = gdal.Open(input_raster)
    
    # resample
    dest_ds = gdal.Warp(output_raster, src_ds,
                      format = 'GTiff',
                      xRes = xRes, yRes = yRes,
                      resampleAlg = resampleAlg)
    
    # clean
    src_ds= None
    dest_ds = None

###############################################################3
def clip_vectorlayer(path, input_filename, clip_filename, output_filename):
    """Clip a shapefile to the extent of a raster. Coordinate system of raster is used. 
    
    Keyword arguments:
    path -- path to the files
    input_filename -- filename input file (shapefile)
    clip_filename -- filename of raster that is used for clipping
    output_filename -- filename for the output file (shapefile clipped)
    
    To run e.g. 
    clip_vectorlayer(data_path, 'SHP/suoli_amb_1_2_3_4_5_GB_def_2019_CLIP.shp', 'SHP/tara_1m.tif', 'SHP/test_tara.shp') 
    """
    
    vector_fn = path + input_filename    
    raster_fn = path + clip_filename
    clipped_fn = path + output_filename

    # open clip layer
    clip_ds = gdal.Open(raster_fn)
    
    # get projection
    proj = osr.SpatialReference(wkt=clip_ds.GetProjection())
    crs = '+init=epsg' + ':' + proj.GetAttrValue('AUTHORITY',1)
    # get extent
    GT = clip_ds.GetGeoTransform()
    xmin = GT[0]
    ymax = GT[3]
    xmax = GT[0] + GT[1]* clip_ds.RasterXSize
    ymin = GT[3] + GT[5]* clip_ds.RasterYSize
    
    # clean 
    clip_ds = None
    
    # create bounding box
    geometry = [Polygon(zip([xmin, xmin, xmax, xmax], [ymax, ymin, ymin, ymax]))]
    bbox = gpd.GeoDataFrame([0], geometry=geometry, columns=['x'])
    bbox.crs = crs
    
    # open vector layer
    source_ds= gpd.read_file(vector_fn)
    # reproject
    source_ds = source_ds.to_crs(crs)
    
    # clip
    clip = gpd.overlay(source_ds, bbox, how="intersection")
    
    # save
    try:
        clip.to_file(filename=clipped_fn, driver='ESRI Shapefile')
    except ValueError:
        print("ValueError in: " + output_filename)
        # move directory with ValueError
        no_gt_dir = 'no_gt/'
        if not os.path.isdir(path + no_gt_dir):
            os.mkdir(path + no_gt_dir)
            split = clip_filename.split('/')
        shutil.move(path+split[0]+ '/' + split[1], path+no_gt_dir+ split[1])

    # clean
    source_ds = None
    clip = None  

################################################################################
def add_band_to_names(path, dirs):
    """ Add the name of the folder to file (e.g. folder RGB, file ends on "_RGB.tif")
    
    Keyword arguments:
    path -- path to the files
    dirs -- list of directories to change
    
    to run e.g.            
    add_band_to_names(datawd, ['RGB', 'NIR'])
    """
    
    for d in dirs:
        tiffiles = list_files(path + d, '.tif')
    
        for f in tiffiles:
            if not f.endswith(d + '.tif'):
                split = f.split(".")
                new_name = split[0] + "_" + d + "." + split[1]
                os.rename(path + d + "/" + f, path + d + "/" + new_name)
                
        tfwfiles = list_files(path + d, '.tfw')
        
        for f in tfwfiles:
            if not f.endswith(d + '.tfw'):
                split = f.split(".")
                new_name = split[0] + "_" + d + "." + split[1]
                os.rename(path + d + "/" + f, path + d + "/" + new_name)

###############################################################################
def move_file(path, dirs):
    """ Move file to dir with tilename
    
    Keyword arguments:
    path -- path to the files
    dirs -- list of directories to change
    
    To run e.g. 
    move_file(datawd, ["RGB", "NIR"])
    """

                
    for d in dirs:
        #files = list_files(path + d, '.tif')
        files = listdir(path + d)
        
        for f in files:
            split = f.split("_")
            
            if not os.path.isdir(path + 'data_20cm/' + split[0] + "/"):
                os.makedirs(path + 'data_20cm/' + split[0] + "/")
                
            new_name = path + 'data_20cm/' + split[0] + "/" + f
            os.rename(path + d + "/" + f, new_name)
           
###############################################################################
def snap_tile(path, no_data_value =-9999, fill_value=0):
    """Snap NIR image to the location of the RGB image. 
    
    Keyword arguments:
    path -- path to the files (per tile a folder with a NIR and a RGB image. 
            RGB image is used as reference location for the NIR image)
    fill_value -- value for blank no class cells (default: 0)
    no_data_value -- value for no_data class cells (default: -9999)
    
    To run e.g.
    snap_tile(datawd)
    """
    
    # check how many folder there are
    ndirs = len(listdir(path))
    n = 0
    
    # create filenames
    for d in listdir(path):
        n += 1
        for f in list_files(path + d, '.tif'):
            if f.endswith('NIR.tif'):
                input_filename = path + d + '/' + f
                output_filename =  path + d + '/' + f
            if f.endswith('RGB.tif'):
                reference_filename = path + d + '/' + f
        
        print(str(n) + '/' + str(ndirs))
                
        # open files
        ref_ds = gdal.Open(reference_filename)
        input_ds = gdal.Open(input_filename,1)
        
        if ref_ds == None or input_ds == None:
            print("error in: " + d)
            shutil.rmtree(path + d)
            continue
        # get the right location
        ref_GT = ref_ds.GetGeoTransform()
        
        # remember metadata
        projection = input_ds.GetProjection()
        nRow = input_ds.RasterXSize
        nCol = input_ds.RasterYSize
        
        # save the data you need
        NIR = np.array(input_ds.GetRasterBand(1).ReadAsArray())
        
        # remove old (wrong) NIR file
        os.remove(input_filename)

        # Create the destination data source
        rasterDriver = gdal.GetDriverByName('GTiff')
        tempSource = rasterDriver.Create(output_filename, 
                                     nRow, 
                                     nCol,
                                     1, #bandnumber
                                     gdal.GDT_UInt16)
        tempSource.SetGeoTransform(ref_GT)
        tempSource.SetProjection(projection)
        tempTile = tempSource.GetRasterBand(1)
        tempTile.Fill(0)
        tempTile.SetNoDataValue(-9999)
        tempTile.WriteArray(NIR)
        tempSource.SetProjection(projection)
        
        # clean
        ref_ds = None
        input_ds = None
        tempSource = None


###############################################################################
def umap_plot(pixel_values, labels, n_neighbors=15, min_dist=0.2, metric='euclidean', title=''):
    """ Create a UMAP reduced dimensionlity plot 
    
    eyword arguments:
    pixel_values -- np.array with shape (nrows, ncols) 
                    (e.g. create with im.reshape(im.shape[0]*im.shape[1], im.shape[2])
                    in case of multiple bands)  
    labels -- np.array with shape (nrows,)
              e.g. if shape is (x,1) squeeze will create the right shape.  
    
    To run e.g.
    umap_plot(umap_im, umap_gt)
    """    
    # UMAP 
    reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            min_dist = min_dist,
            metric = metric)
    embedding = reducer.fit_transform(pixel_values)
    
    # make sure the shape of the labels array is right
    #umap_gt_sq = np.squeeze(labels)
    umap_gt_sq = labels
    
    #plot
    colors = ['red','green','blue','purple','yellow']
    colors_map = umap_gt_sq[:,]
    #tare = [770,659,654,690,650]
    #for i, cl in enumerate(tare):
    for cl in range(5):
        indices = np.where(colors_map==cl)
        plt.scatter(embedding[indices,0], embedding[indices, 1], c=colors[cl], label=[cl])
    plt.legend()
    plt.title(title)
    plt.show() 

    
def pca_plot(pixel_values, labels):
    """ Create a PCA reduced dimensionlity plot 
    
    Keyword arguments:
    pixel_values -- np.array with shape (nrows, ncols) 
                    (e.g. create with im.reshape(im.shape[0]*im.shape[1], im.shape[2])
                    in case of multiple bands)  
    labels -- np.array with shape (nrows,)
              e.g. if shape is (x,1) squeeze will create the right shape.  
    
    To run e.g.
    pca_plot(umap_im, umap_gt)
    """
    # pca
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(pixel_values)
    
     # make sure the shape of the labels array is right
    umap_gt_sq = np.squeeze(labels)
    
    # plot
    colors = ['red','green','blue','purple','yellow']
    colors_map = umap_gt_sq[:,]
    tare = [770,659,654,690,650]
    #for i, cl in enumerate(tare):
    for cl in range(5):
        indices = np.where(colors_map==cl)
        plt.scatter(principalComponents[indices,0], principalComponents[indices, 1], c=colors[cl], label=[cl])
    plt.legend()
    plt.show()

####################################################################################################################
def sample_patches_of_class(population, n_samples, cl, gt, patch_size, output_file, d):
    """ sample origin coordinates of patches and save in csv file. 
    
    Keyword arguments:
        population -- numpy.ndarray with coordinates containing class cl
        n_samples -- number of samples to take= n_samples_no_coltivable
        cl -- ground truth class= classes[4]
        gt -- ground truth image= gt
        patch_size -- size must be squared --> input 1 int. 
        
    e.g. 
    """    
    n_loops = 0
        
    with open(output_file, 'a') as file:
        writer = csv.writer(file, delimiter =',')     
        while n_samples > 0 and len(population) > n_samples:
            idx = random.sample(range(len(population)),n_samples)
            gt_idx = population[idx]
            population = np.delete(population,idx,axis=0)
            for r,c in gt_idx:
                patch = gt[r:(r+patch_size),c:(c+patch_size)]
                mask_past = np.uint16(patch==cl)
                mask_el = np.uint16(patch==0) 
                if (np.sum(mask_past)>=1000 and np.sum(mask_el)==0 and patch.shape[0]==patch_size and patch.shape[1]==patch_size) or n_loops>=1000:
                    writer.writerow([d, r, c])
                    n_samples -= 1
                    n_loops =0
                    continue
                n_loops += 1
                if n_loops==100:
                    n_samples==0
                    break


def to_categorical_classes(y, n_classes=None, dtype=np.int8, classes = [638,659,654,650,770]):
  """Converts a class vector to binary class matrix.

  E.g. for use with categorical_crossentropy.

  Arguments:
      y: class vector to be converted into a matrix (classes not int 0-num-classes)
          (if integers from 0 to num_classes use keras.utils.to_categorical instead).
      n_classes: total number of classes.
      dtype: The data type expected by the input. Default: `'float32'`.
      classes: list with classes

  Returns:
      A binary matrix representation of the input. The classes axis is placed
      last.
  """

  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  n_classes = len(classes)
  n = y.shape[0]
  categorical = np.zeros((n, n_classes), dtype=dtype)
  for i, cl in enumerate(classes):
    y[np.where(y == cl)]=i
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (n_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


def read_patch(data_path, dtm_path, coords, patch_size, idx, classes):
    """ load patch based on left-top coordinate
    
    Arguments:
      data_path: path to the folder containing dataset
      coords: file with coordinates in the format [subfolder, r (=left), c (=top)]
      patch_size: size of patch to be extracted in number of pixels (patch will be squared)
      idx: index of row in coords file to be used
      classes: list with classes

    Returns:
      patch: numpy array of size (patch_size, patch_size, 5) with normalized RGB, NIR, DTM 
      gt: numpy array of size (patch_size, patch_size, n_classes) with one-hot encoded ground truth
    """

    n_features = 5
    
    folder, r, c = coords.iloc[idx]
    r = int(r)*5
    c = int(c)*5

    patch = np.zeros([patch_size,patch_size,n_features],dtype=np.float16)
    
    # RGB
    ds = gdal.Open(data_path + folder + '/' + folder + '_RGB.tif',gdal.GA_ReadOnly)    
    # needed for resampling of dtm 
    for x in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(x)
        patch[:,:,x-1] = band.ReadAsArray(c,r,patch_size, patch_size)

    # NIR
    ds = gdal.Open(data_path + folder + '/' + folder + '_NIR.tif' ,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    patch[:,:,3] = band.ReadAsArray(c, r, patch_size, patch_size)

    # DTM
    ds = gdal.Open(dtm_path + folder + '/dtm135_20cm.tif' ,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    patch[:,:,4] = band.ReadAsArray(c, r, patch_size, patch_size)

    #normalization
    patch = np.divide(patch,255)
    
    # load ground truth
    ds = gdal.Open(data_path + folder + '/tare.tif' ,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    gt = band.ReadAsArray(c, r, patch_size, patch_size)
    
    # take care of classes
    gt[np.where(gt == 656)] = 650
    gt[np.where(gt == 780)] = 770
    gt[np.isin(gt, classes)==False] = 0
    
    gt = to_categorical_classes(gt, classes)
    
    return((patch,gt))

###########################################################################################
# =============================================================================
# def read_patch_rasterio(data_path, coords, patch_size, idx, classes):
# """ same as read_patch but using rasterio"""
# 
#     folder, r, c = coords.iloc[idx]
#     r = int(r)*5
#     c = int(c)*5
# 
#     # load features
#     with rasterio.open(data_path + folder + '/' + folder + '_RGB.tif') as ds:
#         patch_RGB = ds.read([1,2,3], window=Window(c, r, patch_size, patch_size)).astype(np.float16)
#     
#     with rasterio.open(data_path + folder + '/' + folder + '_NIR.tif') as ds:
#         patch_NIR = ds.read(1, window=Window(c, r, patch_size, patch_size)).reshape(1,patch_size, patch_size).astype(np.float16)
# 
#     im = np.vstack([patch_RGB, patch_NIR])
#     im = np.moveaxis(im, 0, -1)
#     
#     #normalization
#     im[:,:,0:4] = np.divide(im[:,:,0:4],255)
# 
#     # load ground truth
#     with rasterio.open(data_path + folder + '/tare.tif') as ds:
#         gt = ds.read(1, window=Window(c, r, patch_size, patch_size))
#         
#     # take care of classes
#     gt[np.where(gt == 656)] = 650
#     gt[np.where(gt == 780)] = 770
#     gt[np.isin(gt, classes)==False] = 0
#     
#     gt = to_categorical_classes(gt, classes)
#     
#     return((im,gt))
# =============================================================================