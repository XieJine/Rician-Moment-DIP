import torch
import torch.nn as nn
from .common import *

class ResidualBlock3D(nn.Module):
    """3D Residual Block with BatchNorm and ReLU"""
    
    def __init__(self, num_features, kernel_size=3, padding=1, need_bias=True):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(num_features, num_features, kernel_size, 
                              padding=padding, bias=need_bias)
        self.bn1 = nn.BatchNorm3d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(num_features, num_features, kernel_size, 
                              padding=padding, bias=need_bias)
        self.bn2 = nn.BatchNorm3d(num_features)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out

def ResNet3D(
        num_input_channels=2, 
        num_output_channels=3, 
        num_blocks=8,           # Number of residual blocks
        num_features=128, 
        kernel_size=3, 
        padding=1,
        need_bias=True,
        need_sigmoid=False):
    """3D ResNet architecture for image restoration tasks.
    
    Arguments:
        num_blocks: Number of residual blocks in the network
        num_features: Number of feature maps in each convolutional layer
        kernel_size: Size of convolutional kernels
        padding: Padding for convolutional layers
        need_bias: Whether to use bias in convolutional layers
        need_sigmoid: Whether to use sigmoid at the output
    """
    
    model = nn.Sequential()
    
    # Initial convolution: input to feature space
    model.add_module('initial_conv', 
                    nn.Conv3d(num_input_channels, num_features, kernel_size, 
                             padding=padding, bias=need_bias))
    model.add_module('initial_relu', nn.ReLU(inplace=True))
    
    # Residual blocks
    for i in range(num_blocks-2):
        model.add_module(f'resblock_{i}', 
                        ResidualBlock3D(num_features, kernel_size, padding, need_bias))
    
    # Final convolution: feature space to output
    model.add_module('final_conv', 
                    nn.Conv3d(num_features, num_output_channels, kernel_size, 
                             padding=padding, bias=need_bias))
    
    # Optional sigmoid activation
    if need_sigmoid:
        model.add_module('sigmoid', nn.Sigmoid())
    
    return model

# 可选：带有跳跃连接的ResNet变体
def ResNetWithSkip3D(
        num_input_channels=2, 
        num_output_channels=3, 
        num_blocks=8,
        num_features=128, 
        kernel_size=3, 
        padding=1,
        need_bias=True,
        need_sigmoid=False):
    """3D ResNet with global skip connection (similar to DnCNN's residual learning)"""
    
    # 输入卷积
    initial_conv = nn.Conv3d(num_input_channels, num_features, kernel_size, 
                           padding=padding, bias=need_bias)
    initial_relu = nn.ReLU(inplace=True)
    
    # 残差块
    res_blocks = nn.Sequential()
    for i in range(num_blocks):
        res_blocks.add_module(f'resblock_{i}', 
                             ResidualBlock3D(num_features, kernel_size, padding, need_bias))
    
    # 输出卷积
    final_conv = nn.Conv3d(num_features, num_output_channels, kernel_size, 
                         padding=padding, bias=need_bias)
    
    # 可选sigmoid
    if need_sigmoid:
        sigmoid = nn.Sigmoid()
    
    def forward(x):
        identity = x
        
        out = initial_conv(x)
        out = initial_relu(out)
        
        out = res_blocks(out)
        
        out = final_conv(out)
        
        # 全局跳跃连接：学习残差
        out += identity
        
        if need_sigmoid:
            out = sigmoid(out)
        
        return out
    
   