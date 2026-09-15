from __future__ import print_function

"""Export results and metrics for a noise-level ablation experiment.

User configuration:
    Key parameters to edit: CUDA_VISIBLE_DEVICES, epoch, checkpoint path, input/mask/reference paths, crop and display slice, input_depth, and result/figure output paths.
    Relative paths assume execution from the repository root.
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
import csv

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
slice = 43
noise_level = 5
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

epoch = 46360
net_name = 'outputs/DIP_M1_W/sigma_Ablation/best_train_model/level_5_sigma6/epoch_'+str(epoch)+'.pt'
save_name = "outputs/DIP_M1_W/sigma_Ablation/best_train_model/level_5_sigma6/denoise/fig/"
save_data_path  =  "outputs/DIP_M1_W/sigma_Ablation/best_train_model/level_5_sigma6/denoise/"

if not os.path.exists(save_name):
    os.makedirs(save_name)

if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)


noisy_path = 'data/generate_data/dwi_level_' + str(noise_level) +'.nii.gz'
input_path = 'data/generate_data/noisy_input.nii.gz'
img_np,affine1 = load_nifti('data/generate_data/dwi_reference.nii.gz')
net_input,_ = load_nifti(input_path)
net_input = net_input[:,:,20:,:]
noisy_data,affine = load_nifti(noisy_path)
noisy_data = noisy_data[:,:,20:,:]
noisy_data = noisy_data.transpose(3,0,1,2)
img_np = img_np[:,:,20:,:]
img_np = img_np.transpose(3,0,1,2)

mask_path = 'data/generate_data/mask.nii.gz'
mask,_ = load_nifti(mask_path)
mask = mask[:,:,20:]
mask_2D = mask[:,:,slice]

#Configure the network.
INPUT = 'noise' 
OPT_OVER = 'net'
#Use reflection padding.
pad = 'reflection'
#Number of input/output channels (DWI volumes). 
input_depth = 31
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
    save_nifti(save_data_path + 'denoise_' + str(epoch) + '.nii.gz', out_np_, affine=affine1)

Matrix=True
if Matrix:
    img_np_mask = mask
    mask_num_npy = np.sum(img_np_mask)
    noisy_np_nonskull = noisy_data*img_np_mask
    img_np_nonskull = img_np * img_np_mask
    out_np_nonskull = out_np * img_np_mask
    print(img_np_nonskull.shape)
    print(out_np_nonskull.shape)
    RMSE_out = np.sqrt((np.sum((out_np_nonskull-img_np_nonskull)**2))/(mask_num_npy*input_depth))
    PSNR_out = 20*np.log10(1/RMSE_out)
    SSIM_out = SSIM(img_np_nonskull,out_np_nonskull)

    RMSE_out_b0 = np.sqrt((np.sum((out_np_nonskull[0,:,:,:]-img_np_nonskull[0,:,:,:])**2))/(mask_num_npy*1))
    PSNR_out_b0 = 20*np.log10(1/RMSE_out_b0)
    SSIM_out_b0 = SSIM(img_np_nonskull[0,:,:,:],out_np_nonskull[0,:,:,:])

    RMSE_out_dwi = np.sqrt((np.sum((out_np_nonskull[1:,:,:,:]-img_np_nonskull[1:,:,:,:])**2))/(mask_num_npy*30))
    PSNR_out_dwi = 20*np.log10(1/RMSE_out_dwi)
    SSIM_out_dwi = SSIM(img_np_nonskull[1:,:,:,:],out_np_nonskull[1:,:,:,:])

    RMSE_out_noisy_b0 = np.sqrt((np.sum((noisy_np_nonskull[0,:,:,:]-img_np_nonskull[0,:,:,:])**2))/(mask_num_npy*1))
    PSNR_out_noisy_b0 = 20*np.log10(1/RMSE_out_noisy_b0)
    SSIM_out_noisy_b0 = SSIM(img_np_nonskull[0,:,:,:],noisy_np_nonskull[0,:,:,:])

    RMSE_out_noisy_dwi = np.sqrt((np.sum((noisy_np_nonskull[1:,:,:,:]-img_np_nonskull[1:,:,:,:])**2))/(mask_num_npy*30))
    PSNR_out_noisy_dwi = 20*np.log10(1/RMSE_out_noisy_dwi)
    SSIM_out_noisy_dwi = SSIM(img_np_nonskull[1:,:,:,:],noisy_np_nonskull[1:,:,:,:])

    print(f'PSNR:{PSNR_out}')
    print(f'PSNR_B0:{PSNR_out_b0}')
    print(f'PSNR_DWI:{PSNR_out_dwi}')
    print(f'Noisy_PSNR_B0:{PSNR_out_noisy_b0}')
    print(f'Noisy_PSNR_DWI:{PSNR_out_noisy_dwi}')
   

    print(f'SSIM:{SSIM_out}')
    print(f'SSIM_B0:{SSIM_out_b0}')
    print(f'SSIM_DWI:{SSIM_out_dwi}')
    print(f'Noisy_SSIM_B0:{SSIM_out_noisy_b0}')
    print(f'Noisy_SSIM_DWI:{SSIM_out_noisy_dwi}')

    Mrtrix_output_path = save_name
    if not os.path.exists(Mrtrix_output_path):
        os.makedirs(Mrtrix_output_path)
    Mrtrix_output_file= Mrtrix_output_path +'SSIM_PSNR_results.csv'
    headers = ["Metric","all" , "b0", "dwi"]
    rows = [
            ["PSNR", PSNR_out, PSNR_out_b0,PSNR_out_dwi],
            ["SSIM",  SSIM_out,SSIM_out_b0,SSIM_out_dwi]
            ]
            
                        # Write the metrics to CSV.
    with open(Mrtrix_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # Write the header.
        writer.writerows(rows)    # Write data rows.

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

    Ref_X = img_np[0, :, :, slice]*mask_2D
    Ref_Y = img_np[1, :, :, slice]*mask_2D
    noisy_X = noisy_data[0,:, :, slice]*mask_2D
    noisy_Y = noisy_data[1,:, :, slice]*mask_2D
    out_np_X = out_np[0, :, :, slice]*mask_2D
    out_np_Y = out_np[1, :, :, slice]*mask_2D
    Input_X = input_fig[0, :, :, slice]
    Input_Y = input_fig[1, :, :, slice]

    fig_save = save_name
    if not os.path.exists(fig_save):
        os.makedirs(fig_save)
    fig_name =fig_save+'ref_data_b0.tif'
    plt.figure(dpi=600, figsize=(x_length/100,y_length/100))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(Ref_X[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.5), plt.axis('off')
    plt.savefig(fig_name,dpi=600) 

    fig_name =fig_save+'ref_data_dwi.tif'
    plt.figure(dpi=600, figsize=(x_length/100,y_length/100))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(Ref_Y[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.3), plt.axis('off')
    plt.savefig(fig_name,dpi=600)
    
    fig_name =fig_save+'noisy_data_b0.tif'
    plt.figure(dpi=600, figsize=(x_length/100,y_length/100))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(noisy_X[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.5), plt.axis('off')
    plt.savefig(fig_name,dpi=600) 

    fig_name =fig_save+'noisy_data_dwi.tif'
    plt.figure(dpi=600, figsize=(x_length/100,y_length/100))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(noisy_Y[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.3), plt.axis('off')
    plt.savefig(fig_name,dpi=600) 

    fig_name =fig_save+'epoch_'+str(epoch)+'_b0.tif'
    plt.figure(dpi=600, figsize=(x_length/100,y_length/100))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(out_np_X[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.5), plt.axis('off')
    plt.savefig(fig_name,dpi=600) 

    fig_name =fig_save+'epoch_'+str(epoch)+'_dwi.tif'
    plt.figure(dpi=600, figsize=(x_length/100,y_length/100))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(out_np_Y[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.3), plt.axis('off')
    plt.savefig(fig_name,dpi=600) 

    fig_name =fig_save+'Noise_input_b0.tif'
    plt.figure(dpi=600, figsize=(x_length/100,y_length/100))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(Input_X[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.5), plt.axis('off')
    plt.savefig(fig_name,dpi=600) 

    fig_name =fig_save+'Noise_input_dwi.tif'
    plt.figure(dpi=600, figsize=(x_length/100,y_length/100))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(1,1,1), plt.imshow(np.rot90(Input_Y[x_idx_1:x_idx_2,y_idx_1:y_idx_2], k=3), cmap='gray', vmin=0, vmax=0.3), plt.axis('off')
    plt.savefig(fig_name,dpi=600)


    

    print('finish')
