from __future__ import print_function
"""
Generate denoised in-vivo DWI results from a checkpoint trained by
`train_in_vivo.py`.

This script loads a selected DIP-M1-W checkpoint obtained by optimizing the
model on the corresponding in-vivo DWI dataset. The loaded model is applied to
the same DIP input used during training to generate and export the final
denoised DWI volume. It can also save representative visualization figures.

Before running, ensure that the following settings are consistent with
`train_in_vivo.py`:
    - Model architecture and input_depth.
    - In-vivo noisy DWI path, brain-mask path, and DIP input path.
    - Spatial crop range and DWI-volume order.
    - Noise level or noise-map setting used during optimization.
    - Checkpoint path and selected epoch.

Key parameters to edit:
    - CUDA_VISIBLE_DEVICES
    - epoch and net_name
    - noisy_path, input_path, and mask_path
    - save_data_path and save_name
    - slice for visualization

Relative paths assume that the script is run from the repository root.
"""
import numpy as np
from models import * 
import torch
import torch.optim
from utils.denoising_utils import * 
from PIL import Image   
import PIL
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import math
import math
import scipy
import os
from dipy.io.image import load_nifti, save_nifti

def SSIM(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    ssim = structural_similarity(y_pred,y_true)
    return ssim

torch.backends.cudnn.enabled = True  
torch.backends.cudnn.benchmark = True 
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
slice = 20
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

epoch = 194120
net_name = 'outputs/DIP_M1_W/in_vivo/trained_model/epoch_'+str(epoch)+'.pt'
save_name = 'outputs/DIP_M1_W/in_vivo/model_result/fig_'+str(epoch)+'/'
# Output directory for the denoised NIfTI volume.
save_data_path  =  "outputs/DIP_M1_W/in_vivo/denoised/"

if not os.path.exists(save_name):
    os.makedirs(save_name)

if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)


noisy_path ="data/in_vivo/noisy_data.nii.gz"
input_path = "data/in_vivo/noisy_input.nii.gz"
net_input,_ = load_nifti(input_path)
net_input = net_input[:,:,:,:]
noisy_data,affine = load_nifti(noisy_path)
noisy_data = noisy_data[:,:,:,:]
noisy_data = noisy_data.transpose(3,0,1,2)

mask_path ="data/in_vivo/mask.nii.gz"
mask,_ = load_nifti(mask_path)
#mask = mask[:,:,:]
mask_2D = mask[:,:,slice]

#Configure the network.
INPUT = 'noise' 
OPT_OVER = 'net'
#Use reflection padding.
pad = 'reflection'
#Number of input/output channels (DWI volumes). 
input_depth = 17
# Initialize the network.
net = get_net(input_depth, 'skip', pad,skip_n33d=32,skip_n33u=32,skip_n11=4,num_scales=4,upsample_mode='trilinear',n_channels=input_depth).type(dtype)

# Prepare the five-dimensional DIP input tensor.
net_input = net_input.transpose(3,0,1,2)
input_fig = net_input
net_input = np.expand_dims(net_input,axis=0)
net_input = net_input.astype(np.float32)
net_input = torch.from_numpy(net_input).cuda() 
net.load_state_dict(torch.load(net_name))
out = net(net_input)
out_np = out.detach().cpu().numpy()[0]


save=True
if save:
    img_np_mask = mask
    out_np_ = out_np * img_np_mask
    #out_np_ = out_np
    out_np_ = out_np_.transpose(1,2,3,0) 
    save_nifti(save_data_path + 'denoise_' + str(epoch) + '.nii.gz', out_np_, affine=affine)


fig = True
if fig:
    mask_row = np.sum(mask_2D,axis=1)
    pp_y = np.where(mask_row!=0)
    x_idx_1 = pp_y[0][0]
    x_idx_2 = pp_y[0][-1] + 1
    x_length = x_idx_2-x_idx_1

    mask_col = np.sum(mask_2D,axis=0)
    pp_x = np.where(mask_col!=0)
    y_idx_1 = pp_x[0][0]
    y_idx_2 = pp_x[0][-1] + 1       
    y_length = y_idx_2-y_idx_1

    noisy_0 = noisy_data[0,:, :, slice]*mask_2D
    out_np_0 = out_np[0, :, :, slice]*mask_2D

    noisy_dwi = noisy_data[1,:, :, slice]*mask_2D
    out_np_dwi = out_np[1, :, :, slice]*mask_2D


    fig_save = save_name
    if not os.path.exists(fig_save):
        os.makedirs(fig_save)
    fig_name =fig_save+'noisy_data_b0.tif'
    plt.figure(dpi=600, figsize=(x_length/50,y_length/50))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(noisy_0[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.6), plt.axis('off')
    plt.savefig(fig_name,dpi=600) 
   

    fig_name =fig_save+'epoch_'+str(epoch)+'_b0.tif'
    plt.figure(dpi=600, figsize=(x_length/50,y_length/50))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(out_np_0[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.6), plt.axis('off')
    plt.savefig(fig_name,dpi=600)

    fig_name =fig_save+'noisy_data.tif'
    plt.figure(dpi=600, figsize=(x_length/50,y_length/50))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(noisy_dwi[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.2), plt.axis('off')
    plt.savefig(fig_name,dpi=600) 
   

    fig_name =fig_save+'epoch_'+str(epoch)+'.tif'
    plt.figure(dpi=600, figsize=(x_length/50,y_length/50))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(out_np_dwi[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.2), plt.axis('off')
    plt.savefig(fig_name,dpi=600) 
    
    print('finish')
