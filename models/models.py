#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:13:08 2019

@author: cordolo
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Conv2D,BatchNormalization,Activation,MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from models.BilinearUpSampling import BilinearUpSampling2D
from models.blocks import conv_block, identity_block, atrous_conv_block, atrous_identity_block

def UNet(input_shape, n_classes, dropout_rate=0.5,weight_decay=0., batch_momentum=0.9, bn_axis=3):
    """ Create a UNet with 4 times downsampling and 4 times upsampling
    
    source: https://github.com/zhixuhao/unet/blob/master/model.py
    
    arguments
    ---------
        dropout_rate: float between 0 and 1. default=0.5
            fraction of the input units to drop    
        weight_decay: float between 0 and 1
            !! NOT USED IN THIS FUNCTION!! l2 weight regularization penalty
        batch_momentum: float between 0 and 1
            !! NOT USED IN THIS FUNCTION!! momentum in the computation of the exponential average of the 
            mean and standard deviation of the data, for feature-wise normalization.
            
    returns
    -------
        model: Keras Model
    """
    inputs = Input(input_shape)

    c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    c1 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same') (c1)
    c1 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    c1 = Dropout(dropout_rate) (c1)
    
    c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (c2)
    c2 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    c2 = Dropout(dropout_rate) (c2)
    
    c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (c3)
    c3 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    c3 = Dropout(dropout_rate) (c3)
    
    c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (p3)
    c4 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (c4)
    c4 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    c4 = Dropout(dropout_rate) (c4)
    
    c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same') (p4)
    c5 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same') (c5)
    c5 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c5)
    c5 = Activation('relu')(c5)
    c5 = Dropout(dropout_rate) (c5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (u6)
    c6 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (c6)
    c6 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c6)
    c6 = Activation('relu')(c6)
    c6 = Dropout(dropout_rate) (c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (u7)
    c7 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (c7)
    c7 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c7)
    c7 = Activation('relu')(c7)
    c7 = Dropout(dropout_rate) (c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (c8)
    c8 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c8)
    c8 = Activation('relu')(c8)
    c8 = Dropout(dropout_rate) (c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same') (u9)
    c9 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c9)
    c9 = Activation('relu')(c9)
    c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same') (c9)
    c9 = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(c9)
    c9 = Activation('relu')(c9)
    c9 = Dropout(dropout_rate) (c9)
    
    outputs = Conv2D(n_classes, (1, 1), activation='softmax') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return(model)
    
# =============================================================================
# def UNet(input_shape, n_classes, dropout_rate=0.5,weight_decay=0., batch_momentum=0.9):
#     """ Create a UNet with 4 times downsampling and 4 times upsampling
#     
#     source: https://github.com/zhixuhao/unet/blob/master/model.py
#     
#     arguments
#     ---------
#         dropout_rate: float between 0 and 1. default=0.5
#             fraction of the input units to drop    
#         weight_decay: float between 0 and 1
#             !! NOT USED IN THIS FUNCTION!! l2 weight regularization penalty
#         batch_momentum: float between 0 and 1
#             !! NOT USED IN THIS FUNCTION!! momentum in the computation of the exponential average of the 
#             mean and standard deviation of the data, for feature-wise normalization.
#             
#     returns
#     -------
#         model: Keras Model
#     """
#     inputs = Input(input_shape)
# 
#     c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
#     c1 = Dropout(dropout_rate) (c1)
#     c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
#     p1 = MaxPooling2D((2, 2)) (c1)
#     
#     c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
#     c2 = Dropout(dropout_rate) (c2)
#     c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
#     p2 = MaxPooling2D((2, 2)) (c2)
#     
#     c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
#     c3 = Dropout(dropout_rate) (c3)
#     c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
#     p3 = MaxPooling2D((2, 2)) (c3)
#     
#     c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
#     c4 = Dropout(dropout_rate) (c4)
#     c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
#     p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
#     
#     c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
#     c5 = Dropout(dropout_rate) (c5)
#     c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
#     
#     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
#     u6 = concatenate([u6, c4])
#     c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
#     c6 = Dropout(dropout_rate) (c6)
#     c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
#     
#     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
#     u7 = concatenate([u7, c3])
#     c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
#     c7 = Dropout(dropout_rate) (c7)
#     c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
#     
#     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
#     u8 = concatenate([u8, c2])
#     c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
#     c8 = Dropout(dropout_rate) (c8)
#     c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
#     
#     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
#     u9 = concatenate([u9, c1], axis=3)
#     c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
#     c9 = Dropout(dropout_rate) (c9)
#     c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
#     
#     outputs = Conv2D(n_classes, (1, 1), activation='softmax') (c9)
#     
#     model = Model(inputs=[inputs], outputs=[outputs])
#     return(model)
# =============================================================================


def AtrousFCN_Resnet53_16s(input_shape, n_classes, weight_decay=0., batch_momentum=0.9,dropout_rate=0.5):
    """ Create a 53 layer ResNet
    
    arguments
    ---------
        weight_decay: float between 0 and 1
            l2 weight regularization penalty
        batch_momentum: float between 0 and 1
            momentum in the computation of the exponential average of the 
            mean and standard deviation of the data, for feature-wise normalization.
        dropout_rate: float between 0 and 1. default=0.5
            !! NOT USED IN THIS FUNCTION!! fraction of the input units to drop 
        
    
    returns
    -------
        model: Keras Model
    """            
    inputs = Input(input_shape)

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, strides=(2, 2), atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    
    x = atrous_conv_block(3, [1024, 1024, 4096], stage=6, block='a', weight_decay=weight_decay, atrous_rate=(4, 4), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [1024, 1024, 4096], stage=6, block='b', weight_decay=weight_decay, atrous_rate=(4, 4), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [1024, 1024, 4096], stage=6, block='c', weight_decay=weight_decay, atrous_rate=(4, 4), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = BilinearUpSampling2D(target_size=tuple(input_shape[0:2]))(x)

    model = Model(inputs, x)
#    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
#    model.load_weights(weights_path, by_name=True)
    return model

def pretrained_Resnet50(input_shape, n_classes, weight_decay=0., batch_momentum=0.9,dropout_rate=0.5):
    """ Create a pretrained ResNet 50 and add upsampling layers for dense prediction
    
    arguments
    ---------
        input_shape: tuple
        
        n_classes: int
    
        weight_decay: float between 0 and 1
            l2 weight regularization penalty
        batch_momentum: float between 0 and 1
            momentum in the computation of the exponential average of the 
            mean and standard deviation of the data, for feature-wise normalization.
        dropout_rate: float between 0 and 1. default=0.5
            !! NOT USED IN THIS FUNCTION!! fraction of the input units to drop 
        
    
    returns
    -------
        model: Keras Model
    """  
    
    # Create the base model from the pre-trained model ResNet50
    base_model = ResNet50(input_shape=(input_shape[0], input_shape[1],3),include_top=False, weights='imagenet')
# =============================================================================
#     for layer in base_model.layers[:-10]: 
#         layer.trainable = False
# =============================================================================
    for layer in base_model.layers:
        layer.trainable = False
    
    inputs = Input(input_shape)
    
    # map inputs to 3 layers
    c1 = Conv2D(3, (1,1))(inputs)
    c1 = BatchNormalization(axis=3)(c1)
    
    # resnet
    resnet = base_model(c1)
    
    # dense classification
    head = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(resnet)
    
    #upsampling
    outputs = BilinearUpSampling2D(target_size=tuple(input_shape[0:2]))(head)

    model = Model(inputs, outputs)

    return model

def pretrained_VGG16(input_shape, n_classes, weight_decay=0., batch_momentum=0.9,dropout_rate=0.5):
    """ Create a pretrained VGG16 and add upsampling layers for dense prediction
    
    arguments
    ---------
        input_shape: tuple
        
        n_classes: int
    
        weight_decay: float between 0 and 1
            l2 weight regularization penalty
        batch_momentum: float between 0 and 1
            momentum in the computation of the exponential average of the 
            mean and standard deviation of the data, for feature-wise normalization.
        dropout_rate: float between 0 and 1. default=0.5
            !! NOT USED IN THIS FUNCTION!! fraction of the input units to drop 
        
    
    returns
    -------
        model: Keras Model
    """  
    
    # Create the base model from the pre-trained model ResNet50
    base_model = VGG16(input_shape=(input_shape[0], input_shape[1],3),include_top=False, weights='imagenet')
# =============================================================================
#     for layer in base_model.layers[:-10]: 
#         layer.trainable = False
# =============================================================================
    for layer in base_model.layers:
        layer.trainable = False
    
    inputs = Input(input_shape)
    
    # map inputs to 3 layers
    c1 = Conv2D(3, (1,1))(inputs)
    c1 = BatchNormalization(axis=3)(c1)
    
    # resnet
    vgg = base_model(c1)
    
    # dense classification
    head = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(vgg)
    
    #upsampling
    outputs = BilinearUpSampling2D(target_size=tuple(input_shape[0:2]))(head)

    model = Model(inputs, outputs)

    return model

def pretrained_VGG16_transpose(input_shape, n_classes, weight_decay=0., batch_momentum=0.9,dropout_rate=0.5):
    """ Create a pretrained VGG16 and add upsampling layers for dense prediction
    
    arguments
    ---------
        input_shape: tuple
        
        n_classes: int
    
        weight_decay: float between 0 and 1
            l2 weight regularization penalty
        batch_momentum: float between 0 and 1
            momentum in the computation of the exponential average of the 
            mean and standard deviation of the data, for feature-wise normalization.
        dropout_rate: float between 0 and 1. default=0.5
            !! NOT USED IN THIS FUNCTION!! fraction of the input units to drop 
        
    
    returns
    -------
        model: Keras Model
    """  
    
    # Create the base model from the pre-trained model ResNet50
    base_model = VGG16(input_shape=(input_shape[0], input_shape[1],3),include_top=False, weights='imagenet')
    
# =============================================================================
#     for layer in base_model.layers[:-10]: 
#         layer.trainable = False
# =============================================================================
    for layer in base_model.layers:
        layer.trainable = False
    
    inputs = Input(input_shape)
    
    # map inputs to 3 layers
    c1 = Conv2D(3, (1,1))(inputs)
    c1 = BatchNormalization(axis=3)(c1)
    
    # resnet
    vgg = base_model(c1)
      
    # upsampling
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (vgg)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(dropout_rate) (c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(dropout_rate) (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(dropout_rate) (c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(dropout_rate) (c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(dropout_rate) (c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
    
    # dense classification
    outputs = Conv2D(n_classes, (1, 1), activation='softmax') (c9)

    model = Model(inputs, outputs)

    return model


##### Try Xception net 


all_models = {
    "UNet": UNet,
    "ResNet": AtrousFCN_Resnet53_16s,
    "Pretrained_ResNet": pretrained_Resnet50,
    "Pretrained_VGG16": pretrained_VGG16,
    "Pretrained_VGG16_T":pretrained_VGG16_transpose
}
        
