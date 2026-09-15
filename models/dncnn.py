import torch
import torch.nn as nn
from .common import *

import torch.nn as nn

def DnCNN(
        num_input_channels=2, 
        num_output_channels=3, 
        num_layers=17, 
        num_features=128, 
        kernel_size=3, 
        padding =1,
        need_bias=True,
        need_sigmoid=False):
    """DnCNN (Denoising Convolutional Neural Network) architecture.
    
    Arguments:
        num_layers: Number of layers in the network (default: 17 for DnCNN-S)
        num_features: Number of feature maps in each convolutional layer
        kernel_size: Size of convolutional kernels
        padding: Padding for convolutional layers
        act_fun: Activation function ('ReLU' for DnCNN)
        need_sigmoid: Whether to use sigmoid at the output (usually False for denoising)
    """
    
    model = nn.Sequential()
    # First layer: input to feature space
    model.add(nn.Conv3d(num_input_channels,num_features, kernel_size, padding=padding,bias=need_bias))
    model.add(nn.ReLU(inplace=True))
    
    # Intermediate layers: feature extraction with BN and ReLU
    for _ in range(num_layers - 2):
        model.add(nn.Conv3d(num_features, num_features, kernel_size=kernel_size, 
                           padding=padding, bias=need_bias))
        model.add(nn.BatchNorm3d(num_features))
        model.add(nn.ReLU(inplace=True))
    
    # Last layer: feature space to output
    model.add(nn.Conv3d(num_features, num_output_channels, kernel_size=kernel_size, 
                       padding=padding, bias=need_bias))
    
    # DnCNN typically doesn't use sigmoid for denoising tasks
    # as it learns the residual (noise) rather than the clean image directly
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model

