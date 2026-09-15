from __future__ import print_function

"""Train the weighted DIP model on an in-vivo DWI dataset.

User configuration:
    Key parameters to edit: CUDA_VISIBLE_DEVICES, input NIfTI paths, crop ranges, noise_level or sigma_, LR, num_iter, input_depth, show_every, checkpoint-resume settings, and all output directories.
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
import os
from dipy.io.image import load_nifti


torch.backends.cudnn.enabled = True  
torch.backends.cudnn.benchmark = True 
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

show_every = 40


# load data
# mask,_ = load_nifti('data/2015_simulated_data/mask_.nii.gz')
# mask_np = mask.astype(np.float32)
# Crop the volume to reduce memory use and computation.
# img_np_mask = mask_np
# Check the array shape.
# print(img_np_mask.shape)
# load noise free image
# img_np,_ = load_nifti('data/nordic/subj1/reacq_new/raw/noisy_data.nii.gz')
# img_np = img_np.astype(np.float32)
# Noise level; edit for the experiment.
# noise_level = 5.14

# Crop the volume to reduce memory use and computation.

# Check the array shape. 
# print(img_np.shape)
# Transpose data to channel-first network layout.
# img_np = img_np.transpose(3,0,1,2)


# Load noisy data.
img_noisy_np,_ = load_nifti("data/in_vivo/1p5T/b1500/noisy_data.nii.gz")  #b1kb3k[0:66]  b5k[66,131] b10k[131,260]
img_noisy_np = img_noisy_np[:,:,:,:]
# img_noisy_np = img_noisy_np[:,:,:,1:]
img_noisy_np = img_noisy_np.astype(np.float32)
img_noisy_np = img_noisy_np.transpose(3,0,1,2)
print(img_noisy_np.shape)

#sigma_, _ = load_nifti("data/MGH-HCP/new/noise_map.nii.gz")
##sigma_ = sigma_[31:225,31:225,30:120]
#sigma_ = np.expand_dims(sigma_,axis=0)
#sigma_ = np.clip(sigma_, a_min=0.001, a_max=None)
#sigma_ = np_to_torch(sigma_).type(dtype).cuda()

sigma_ = 0.028
#print(sigma_.shape)
# Apply the brain mask to noisy data.
img_noisy_np_nonskull = img_noisy_np
print(img_noisy_np_nonskull.shape)
# Apply the brain mask to reference data.
# img_np_nonskull = img_np


#Configure the network.
INPUT = 'noise' 
OPT_OVER = 'net'
#Use reflection padding.
pad = 'reflection'
i = 0


# Learning rate.
LR = 0.01

# Optimizer.
OPTIMIZER='adam'
#Maximum number of iterations.
num_iter = 200000
#Number of input/output channels (DWI volumes). 
input_depth = 17
# Initialize the network.
net = get_net(input_depth, 'skip', pad,skip_n33d=32,skip_n33u=32,skip_n11=4,num_scales=4,upsample_mode='trilinear',n_channels=input_depth).type(dtype)


# Prepare the five-dimensional DIP input tensor.
net_input,_ = load_nifti("data/in_vivo/1p5T/b1500/noisy_input.nii.gz")
net_input = net_input.astype(np.float32)
net_input = net_input[:,:,:,:]

# net_input = net_input[:,:,1:91,:]
net_input = net_input.transpose(3,0,1,2)
net_input = np.expand_dims(net_input,axis=0)
net_input = net_input.astype(np.float32)
net_input = torch.from_numpy(net_input).cuda() 
# Report device and tensor shape.
print(net_input.device)
print(type(net_input))
print(net_input.shape)
# Mean-squared-error helper.
mse = torch.nn.MSELoss().type(dtype)
# Convert arrays to PyTorch tensors.
img_noisy_torch = np_to_torch(img_noisy_np).type(dtype) # Convert the noisy array to a tensor. 

# normal
psrn_noisy_list = []  
psrn_out_list = []  
rmse_out_list = []
total_loss_list = [] 

# # Optional checkpoint resume block.
# bp = True
# ii = np.load('outputs/DIP_M1_W/2015_simulated/iterations/level_3/i.npy')
# i = np.load('outputs/DIP_M1_W/2015_simulated/iterations/level_3/i.npy')
# i = i-(i%40)
# LR=0.01*0.9**(i//2000)

# trained_model_name = 'outputs/DIP_M1_W/2015_simulated/trained_model/level_3/epoch_' + str(i) +'.pt'
# trained_model_name_last = 'outputs/DIP_M1_W/2015_simulated/trained_model/level_3/epoch_' + str(i-1) +'.pt'
# net.load_state_dict(torch.load(trained_model_name_last))
# out_bp = net(net_input).data
# net.load_state_dict(torch.load(trained_model_name))

# psrn_out_list = list(np.load('outputs/DIP_M1_W/2015_simulated/psnr_rmse/level_3/psnr_level_3.npy'))
# total_loss_list = list(np.load('outputs/DIP_M1_W/2015_simulated/psnr_rmse/level_3/loss_level_3.npy'))
# rmse_out_list = list(np.load('outputs/DIP_M1_W/2015_simulated/psnr_rmse/level_3/rmse_level_3.npy'))

# for k in range(i,ii+1):
#     del total_loss_list[i]
#     del psrn_out_list[i]
#     del rmse_out_list[i]

net_input_saved = net_input.detach().clone()   
noise = net_input.detach().clone()              
# print(type(net_input))
out_avg = None
last_net = None
psrn_noisy_last = 0              
total_loss_last = 9999
psrn_gt_last = 0
t=0

# mask_num_npy = np.sum(img_np_mask)
# mask_num = np.sum(img_np_mask)
# mask_num=70*86*90/mask_num
# mask_num = torch.tensor(mask_num,dtype=torch.float32,device='cuda',requires_grad=False)  
# mask_num_sqrt = torch.sqrt(mask_num)
path_main_loss = 'outputs/DIP_M1_W/1p5T/loss/b1k5/'
if not os.path.exists(path_main_loss):
     os.makedirs(path_main_loss)
path_main_iteration = 'outputs/DIP_M1_W/1p5T/iterations/b1k5/'
if not os.path.exists(path_main_iteration):
    os.makedirs(path_main_iteration)
path_main_result = 'outputs/DIP_M1_W/1p5T/result/b1k5/'
if not os.path.exists(path_main_result):
    os.makedirs(path_main_result)
path_main_train_model = 'outputs/DIP_M1_W/1p5T/trained_model/b1k5/'
if not os.path.exists(path_main_train_model):
    os.makedirs(path_main_train_model)
path_main_model = 'outputs/DIP_M1_W/1p5T/model/b1k5/'
if not os.path.exists(path_main_model):
    os.makedirs(path_main_model)

        
last_i=0
def closure():   #######！！！！！#####
    
    global last_i, total_loss_last,num_iter,t,psrn_gt_last,LR,i, out_avg, psrn_noisy_last, last_net, net_input, psrn_noisy_list, psrn_gt_list,total_loss_list   #Denoising PSNR state.  
    out = net(net_input)    # <class 'torch.Tensor'>
# Output from the previous iteration.
    alpha1 = (out / (2*sigma_))**2   
    pi = torch.tensor(np.pi,dtype=torch.float32,device='cuda',requires_grad=False) 
    SNR = out.data/sigma_
    # ep = (SNR**2+2)-(pi/8)*(((2+SNR**2)*torch.special.i0e((SNR**2)/4)+(SNR**2)*torch.special.i1e((SNR**2)/4)))**2
    ep = (2 + SNR**2) - (pi/8)*(((2+SNR**2)*torch.special.i0e((SNR**2)/4) + (SNR**2)*torch.special.i1e((SNR**2)/4))**2)
    M1_weight = sigma_*torch.sqrt(ep)
    temp = torch.sqrt((pi*(sigma_**2))/2)*((1+2*alpha1)*torch.special.i0e(alpha1) + 2*alpha1*torch.special.i1e(alpha1))
    total_loss = mse(temp/M1_weight, img_noisy_torch/M1_weight)
    total_loss.backward() 
    total_loss_ = total_loss.data.cpu().numpy()
    total_loss_ = float(total_loss_)
    out_np = out.detach().cpu().numpy()[0]  #Convert the output to a NumPy array.
    out_np_nonskull = out_np 
    # out_np_nonskull_gt = out_np_nonskull
    
    # RMSE_out = np.sqrt((np.sum((out_np_nonskull-img_np_nonskull)**2))/(mask_num_npy*67))
    # MSE_out = (np.sum((out_np_nonskull-img_np_nonskull)**2))/(mask_num_npy*67)
    # PSNR_out = 20*np.log10(1/RMSE_out)

    # RMSE_noisy = np.sqrt((np.sum((out_np_nonskull-img_noisy_np_nonskull)**2))/(mask_num_npy*67))
    # PSNR_noisy = 20*np.log10(1/RMSE_noisy)
    # psrn_noisy_list.append(PSNR_noisy)
    # psnr_noisy_array = np.array(psrn_noisy_list)

    # psrn_out_list.append(PSNR_out)
    # # mse_out_list.append(MSE_out)   
    # rmse_out_list.append(RMSE_out)   
    total_loss_list.append(total_loss_)  
    # psnr_array = np.array(psrn_out_list)
    # psnr_noisy_name = 'outputs/DIP_M1_W/2015_simulated/psnr_rmse/level_3/psnr_noisy_level_3.npy'
    # psnr_name = 'outputs/DIP_M1_W/2015_simulated/psnr_rmse/level_3/psnr_level_3.npy'
    # mse_name = 'outputs/DIP_M1_W/2015_simulated/psnr_rmse/level_3/mse_level_3.npy'
    # rmse_name = 'outputs/DIP_M1_W/2015_simulated/psnr_rmse/level_3/rmse_level_3.npy'
    loss_name = path_main_loss+'loss.npy'
    # np.save(psnr_name, np.array(psrn_out_list))
    # np.save(psnr_noisy_name, psnr_noisy_array)
    # np.save(rmse_name, np.array(rmse_out_list))   
    np.save(loss_name, np.array(total_loss_list))
    iteration_name = path_main_iteration+'i.npy'  
    np.save(iteration_name, i)
    #Save the metric history.

    # if (i+1)%show_every == 0:
    #     model_name_last = 'outputs/DIP_M1_W/2015_simulated/trained_model/level_3/epoch_' + str(i) +'.pt'
    #     torch.save(net.state_dict(),model_name_last)
    print ('Iteration %05d  Loss %f' % (i, total_loss.item()), '\n', end='')
    if  i % show_every == 0:
        #Periodically visualize the denoising result.
        if i% (show_every) ==0:
            out_np = torch_to_np(out)
            X = 25
            out_np_X = out_np[0, :, :, X]
            out_np_Y = out_np[1, :, :, X]
           


            fig_name = path_main_result+'epoch_' + str(i) +'.png'
            plt.figure(figsize=(5,3))
            plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1), plt.axis('off')
            plt.subplot(3,2,1), plt.imshow(img_noisy_np_nonskull[0, :, :, X],vmin=0,vmax=0.3, cmap='gray'), plt.axis('off') 
            plt.subplot(3,2,2), plt.imshow(img_noisy_np_nonskull[1, :, :, X],vmin=0,vmax=0.2, cmap='gray'), plt.axis('off')
          
        
            plt.subplot(3,2,3), plt.imshow(out_np_X,vmin=0,vmax=0.3, cmap='gray'), plt.axis('off') 
            plt.subplot(3,2,4), plt.imshow(out_np_Y,vmin=0,vmax=0.2, cmap='gray'), plt.axis('off')
           

            plt.subplot(3,2,5), plt.imshow(out_np_X-img_noisy_np_nonskull[0, :, :, X],vmin=0,vmax=0.01, cmap='gray'), plt.axis('off') 
            plt.subplot(3,2,6), plt.imshow(out_np_Y-img_noisy_np_nonskull[1, :, :, X],vmin=0,vmax=0.01, cmap='gray'), plt.axis('off')
            

            plt.savefig(fig_name)

        loss_name = path_main_loss+'loss_' + str(i%(show_every*2)) +'.pt'
        model_name = path_main_train_model+'epoch_' + str(i) +'.pt'
        model_name_ = path_main_model+'epoch_' + str(i%(show_every*2)) +'.pt'
        torch.save(total_loss,loss_name)#Save the checkpoint.
        torch.save(net.state_dict(),model_name_)
        torch.save(net.state_dict(),model_name)#Save the checkpoint.

        
        # if  i!=0 and i!=show_every and psnr_noisy_array[i-show_every+1:i+1].min() - psnr_noisy_array[i-2*show_every+1:i-show_every+1].min() < -0.05:  
        #     if last_i != i:
        #         t = 5
        #     if last_i == i:
        #         t = t-1
        #     last_i =i
        #     if t==0:
        #         if psnr_noisy_array[i].min() - psnr_noisy_array[i-show_every].min()<-0.5:
        #             t=1
        #     if t >0:
        #         i = i-show_every
        #         net.load_state_dict(torch.load('outputs/DIP_M1_W/2015_simulated/model/level_3/epoch_' + str(i%(show_every*2)) +'.pt'))
        #         total_loss = torch.load('outputs/DIP_M1_W/2015_simulated/loss/level_3/epoch_' + str(i%(show_every*2)) +'.pt')
        #         total_loss.backward()
        #         for k in range(i+1,i+show_every+1):
        #             del total_loss_list[i+1]
        #             del psrn_out_list[i+1]
        #             del rmse_out_list[i+1]
        #             del psrn_noisy_list[i+1]
        #         total_loss_last = total_loss_list[i]
        #         out = net(net_input)
        #         out_inside_last = out.data
        #     else:
        #         total_loss_last = total_loss
        # else:
        #     total_loss_last = total_loss
    LR=0.01*0.9**(i//2000)
    i += 1      
    return total_loss_list,psrn_out_list,rmse_out_list

p = get_params(OPT_OVER, net, net_input) 
optimizer = torch.optim.Adam(p, lr=LR)
for j in range(num_iter):
        optimizer.zero_grad()
        optimizer.step(closure)
