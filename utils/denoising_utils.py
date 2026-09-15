"""Shared Deep Image Prior utility functions.

This module was retained from the submitted implementation and cleaned for publication.
"""

import os
from .common_utils import *    
import numpy
import matplotlib.pyplot as plt
import random

def get_noisy_image(img_np, sigma):   # np.ndarray    sigma ()
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1     np0-1     
        sigma: std of the noise
    """
    #real = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape) ,0 ,1).astype(np.float32)#
    
    #im = np.clip(np.random.normal(scale=sigma, size=img_np.shape) ,0 ,1).astype(np.float32)    #    
   # real = img_np + sigma * np.random.standard_normal(img_np.shape)
    
    #im = sigma * np.random.standard_normal(img_np.shape)
    
    #img_noisy = np.sqrt(real**2 + im**2) 
    
   # img_noisy_np = np.clip(img_noisy, 0, 1).astype(np.float32)

#     channels, length, width = img_np.shape   #69 140 140
    
#     num_p =channels * length * width
    
#     data_noise_free = np.reshape(img_np,[-1,1]) 
    
#     real = np.reshape(sigma*np.random.standard_normal(num_p),[-1,1]) + data_noise_free
    
#     im = np.reshape(sigma*np.random.standard_normal(num_p),[-1,1])
    
#     img_noisy = np.sqrt(real**2 + im**2) 
    
#     img_noisy_np = np.reshape(img_noisy,[channels, length, width])
    
#     img_noisy_np = np.clip(img_noisy_np, 0, 1).astype(np.float32)
    
#     np.random.seed(0)
    real = img_np + sigma * np.random.standard_normal(img_np.shape)
#     print(np.random.standard_normal(img_np.shape)[50,40,40:45])
    
#     np.random.seed(1)   #set seed make sure the same random numbers
    im = sigma * np.random.standard_normal(img_np.shape)
#     print(np.random.standard_normal(img_np.shape)[50,40,40:45])
    
#     print('################Without seed#############')
#     print(np.random.standard_normal(img_np.shape)[50,40,40:45])
    img_noisy = np.sqrt(real**2 + im**2) 
    
    img_noisy_np = np.clip(img_noisy, 0, 1).astype(np.float32)

    
    return img_noisy_np   # ()

#np.random.normal(loc=0.0, scale=1.0, size=None)   

def findSmallest(arr):
    smallest = arr[0]      #
    smallest_index = 0     #
    for i in range(1,len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index
#
def selectionSort(arr):   #
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))   #，
    return newArr
