#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:02:18 2019

@author: cordolo
"""

import torch.nn as nn
import torch

__all__ = ['unet']


class UNet(nn.Module):
    
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        
        self.conv1a = nn.Conv2d(n_channels, 16, stride=1,kernel_size=3, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(16)
        self.relu1a = nn.ReLU(inplace=True)
        self.conv1b = nn.Conv2d(16, 16,kernel_size=3,padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(16)
        self.relu1b = nn.ReLU(inplace=True)
        self.conv1c = nn.Conv2d(16, 16, stride=1,kernel_size=3, padding=1, bias=False)
        self.bn1c = nn.BatchNorm2d(16)
        self.relu1c = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2a = nn.Conv2d(16, 32,kernel_size=3, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(32)
        self.relu2a = nn.ReLU(inplace=True)
        self.conv2b = nn.Conv2d(32, 32,kernel_size=3, padding=1, bias=False)
        self.bn2b = nn.BatchNorm2d(32)
        self.relu2b = nn.ReLU(inplace=True)  
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3a = nn.Conv2d(32, 64,kernel_size=3, padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(64)
        self.relu3a = nn.ReLU(inplace=True)
        self.conv3b = nn.Conv2d(64, 64,kernel_size=3, padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(64)
        self.relu3b = nn.ReLU(inplace=True)  
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4a = nn.Conv2d(64, 128,kernel_size=3, padding=1, bias=False)
        self.bn4a = nn.BatchNorm2d(128)
        self.relu4a = nn.ReLU(inplace=True)
        self.conv4b = nn.Conv2d(128, 128,kernel_size=3, padding=1, bias=False)
        self.bn4b = nn.BatchNorm2d(128)
        self.relu4b = nn.ReLU(inplace=True)   
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv5a = nn.Conv2d(128, 256,kernel_size=3, padding=1, bias=False)
        self.bn5a = nn.BatchNorm2d(256)
        self.relu5a = nn.ReLU(inplace=True)
        self.conv5b = nn.Conv2d(256, 256,kernel_size=3, padding=1, bias=False)
        self.bn5b = nn.BatchNorm2d(256)
        self.relu5b = nn.ReLU(inplace=True)  
        
        self.convT6a = nn.ConvTranspose2d(256, 128,kernel_size=2, stride=2,padding=0, bias=False)
        
        self.conv6a = nn.Conv2d(256,128, kernel_size=3, padding=1, bias=False)
        self.bn6a = nn.BatchNorm2d(128)
        self.relu6a = nn.ReLU(inplace=True)
        self.conv6b = nn.Conv2d(128, 128,kernel_size=3, padding=1, bias=False)
        self.bn6b = nn.BatchNorm2d(128)
        self.relu6b = nn.ReLU(inplace=True)   

        self.convT7a = nn.ConvTranspose2d(128, 64,kernel_size=2, stride=2,padding=0, bias=False)
        
        self.conv7a = nn.Conv2d(128,64, kernel_size=3, padding=1, bias=False)
        self.bn7a = nn.BatchNorm2d(64)
        self.relu7a = nn.ReLU(inplace=True)
        self.conv7b = nn.Conv2d(64, 64,kernel_size=3, padding=1, bias=False)
        self.bn7b = nn.BatchNorm2d(64)
        self.relu7b = nn.ReLU(inplace=True)   
        
        self.convT8a = nn.ConvTranspose2d(64, 32,kernel_size=2, stride=2,padding=0, bias=False)
        
        self.conv8a = nn.Conv2d(64,32, kernel_size=3, padding=1, bias=False)
        self.bn8a = nn.BatchNorm2d(32)
        self.relu8a = nn.ReLU(inplace=True)
        self.conv8b = nn.Conv2d(32, 32,kernel_size=3, padding=1, bias=False)
        self.bn8b = nn.BatchNorm2d(32)
        self.relu8b = nn.ReLU(inplace=True)  
        
        self.convT9a = nn.ConvTranspose2d(32, 16,kernel_size=2, stride=2,padding=0, bias=False)
        
        self.conv9a = nn.Conv2d(32,16, kernel_size=3, padding=1, bias=False)
        self.bn9a = nn.BatchNorm2d(16)
        self.relu9a = nn.ReLU(inplace=True)
        self.conv9b = nn.Conv2d(16, 16,kernel_size=3, padding=1, bias=False)
        self.bn9b = nn.BatchNorm2d(16)
        self.relu9b = nn.ReLU(inplace=True)  


    def forward(self, x, return_feature_maps=False):
        x = self.relu1a(self.bn1a(self.conv1a(x)))
        x = self.relu1b(self.bn1b(self.conv1b(x)))
        layer1 = self.relu1c(self.bn1c(self.conv1c(x)))
        x = self.maxpool1(layer1)
        x = self.relu2a(self.bn2a(self.conv2a(x)))
        layer2 = self.relu2b(self.bn2b(self.conv2b(x)))
        x = self.maxpool2(layer2)
        x = self.relu3a(self.bn3a(self.conv3a(x)))
        layer3 = self.relu3b(self.bn3b(self.conv3b(x)))
        x = self.maxpool3(layer3)
        x = self.relu4a(self.bn4a(self.conv4a(x)))
        layer4 = self.relu4b(self.bn4b(self.conv4b(x)))
        x = self.maxpool4(layer4)
        x = self.relu5a(self.bn5a(self.conv5a(x)))
        x = self.relu5b(self.bn5b(self.conv5b(x)))
        
        x = self.convT6a(x)
        x = torch.cat([x, layer4], dim=1)
        x = self.relu6a(self.bn6a(self.conv6a(x)))
        x = self.relu6b(self.bn6b(self.conv6b(x)))
        
        x = self.convT7a(x)
        x = torch.cat([x, layer3], dim=1)
        x = self.relu7a(self.bn7a(self.conv7a(x)))
        x = self.relu7b(self.bn7b(self.conv7b(x)))      
        
        x = self.convT8a(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.relu8a(self.bn8a(self.conv8a(x)))
        x = self.relu8b(self.bn8b(self.conv8b(x))) 
        
        x = self.convT9a(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.relu9a(self.bn9a(self.conv9a(x)))
        x = self.relu9b(self.bn9b(self.conv9b(x))) 
        
        return x
    
def unet(n_channels=5, **kwargs):
    model = UNet(n_channels=n_channels, **kwargs)

    return model