"""Shared Deep Image Prior utility functions.

This module was retained from the submitted implementation and cleaned for publication.
"""

import torch
import torch.nn as nn
import torchvision
import sys
import scipy.io as scio
import numpy as np
from PIL import Image   #PIL python
import PIL
import numpy as np

import matplotlib.pyplot as plt

def crop_image(img, d=32):                    #crop_image   PILcrop
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)   # (384,256)  (400,261)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),   #bbox (snail)  [8,2,392,258]
    ]

    img_cropped = img.crop(bbox)
    return img_cropped        #

def get_params(opt_over, net, net_input, downsampler=None):          #get_params 
    '''Returns parameters that we want to optimize over.   #

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"            input
        net: network                                          net
        net_input: torch.Tensor that stores input `z`                   z  
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]   # a list of images all of the same size 
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)   #make_grid nrow8 #WX
    
    return torch_grid.numpy()    

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):  #lanczos 
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure                             #factor     figure
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)           ##!!!  transpose  images_np (3,384,256)  max3
    
    #assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"  # 3 
                                             
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np] #np.concatenate 

    grid = get_image_grid(images_np, nrow)    #  grid get_image_grid make_grid 
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor)) #  figsize:figure，；
    
    #if images_np[0].shape[0] == 1:          ##!!!  transpose  images_np (3,384,256)  max3
    plt.imshow(grid[0], cmap='gray', interpolation=interpolation)  #  transpose
    #else:
    #plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)  # W H C img PIL
    
    #plt.show()                 #imshow()，，。   plt.show()。
    return grid

def get_image(path):                       ########  
    """Load an image and resize to a cpecific size. 
    Args: 
        path: path to image  X z X   X∈(1,96)
    """
    denoised = scio.loadmat(path)
    img_np = denoised['ims_denoised'].astype(np.float32)#       b=0     0   14   28  42 56
    
    img_np = img_np[:, :, : , :]            #3D   z  69   140*140*96*69 (z=50)

    
    b0 = [14,28,42,56]
    for z in range(img_np.shape[3]):
           if z in b0:
                img_np[:, :, :, z] = img_np[:, :, :, 50]  #b0
    #print(img_np.max())
    img_np = (img_np-img_np[:,:,50,:].min())/(img_np[:,:,50,:].max()-img_np[:,:,50,:].min())   #
    #print(img_np.max())
    
    #for z in range(img_np.shape[2]):
    #       if z in b0:
    #            img_np[:, :, z] = img_np[:, :, 50]  #b0
    #img_np = (img_np-img_np.min())/(img_np.max()-img_np.min())   #
    return img_np #img_np  <class 'numpy.ndarray'>
   

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`.""" #x
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor         #
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid     #`noise`noise；`meshgrid`np.
        spatial_size: spatial size of the tensor to initialize          #
        noise_type: 'u' for uniform; 'n' for normal         #u     n 
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
                                                             #，。。
    """
   # if isinstance(spatial_size, int):
       # spatial_size = (spatial_size, spatial_size)
    if method == 'noise':            #'noise'
        
        shape = [1, input_depth, spatial_size[0], spatial_size[1],spatial_size[2]]
        
        net_input = torch.zeros(shape)   # size,torch.dtype，0tensor
        
        fill_noise(net_input, noise_type)
        net_input *= var          #？  
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input     #？？

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.       #PIL  np  
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)     #arimg_PIL 

    if len(ar.shape) == 3:      #ar.shape  ar  (261,400,3)     140*140*69
        ar = ar.transpose(2,0,1)              # W--0 H--1 C--2  (0,1,2) ()    (2,0,1)---> C--2 W--0 H--1 ()
    else:
        ar = ar[None, ...]   #

    return ar.astype(np.float32) / 255.   # <class 'numpy.ndarray'> 

    #######float32，255，（0，255）（0，1）

def np_to_pil(img_np):        ## pil_to_np np  C * W * H   transpose
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8) #np.clip ,denoising  img_np(0,1) 
    
    if img_np.shape[0] == 1:
        ar = ar[0]              #    
    else:
        ar = ar.transpose(1, 2, 0) # C--0 W--1 H--2 ---->  (1,2,0)  W--1 H--2 C--0  PIL W H C

    return Image.fromarray(ar)   #，arrayimage   

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.  

    From C x W x H [0..1] to  C x W x H [0..1] 
    '''
    return torch.from_numpy(img_np)[None, :]  ###  ，，，

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]   #np 


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'                   #
        parameters: list of Tensors to optimize over            # get_param
        closure: function, that returns loss variable           #，
        LR: learning rate                              #Learning rate.             
        num_iter: number of iterations                     #
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)  #Optimizer   Learning rate.0.001
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')  
        optimizer = torch.optim.Adam(parameters, lr=LR)   #Optimizer   Learning rate.lrLR=0.01
        
        for j in range(num_iter):     #3000
            optimizer.zero_grad()        #
            closure()                # 
            optimizer.step()           #optimizer.step()         ... 
    elif optimizer_type == 'SGD':
        print('Starting optimization with SGD')  
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        for j in range(num_iter):     
            optimizer.zero_grad()        
            closure()               
            optimizer.step()                  
    else:
        assert False