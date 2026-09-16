from __future__ import print_function

"""Train the Rician second Moment DIP model on simulated data with a spatially varying noise map.

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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

show_every = 40


# load data
mask,_ = load_nifti("data/generate_data/mask.nii.gz")
#mask = mask[:,:,20:]
mask_np = mask.astype(np.float32)
# Crop the volume to reduce memory use and computation.
img_np_mask = mask_np
# Check the array shape.
print(img_np_mask.shape)
# load noise free image
img_np,_ = load_nifti('data/generate_data/dwi_reference.nii.gz')
#img_np = img_np[:,:,20:,:]
img_np = img_np.astype(np.float32)
# Noise level; edit for the experiment.
noise_level = 5
# Crop the volume to reduce memory use and computation.

# Check the array shape. 
print(img_np.shape)
# Transpose data to channel-first network layout.
img_np = img_np.transpose(3,0,1,2)


# Load noisy data.
img_noisy_np,_ = load_nifti('data/generate_data/dwi_level_3_5.nii.gz')
img_noisy_np = img_noisy_np.astype(np.float32)
#img_noisy_np = img_noisy_np[:,:,20:]
img_noisy_np = img_noisy_np.transpose(3,0,1,2)
print(img_noisy_np.shape)

sigma_, _ = load_nifti('data/generate_data/noise_map_level_3_5.nii.gz')
#sigma_ = sigma_[:,:,20:]
sigma_ = np.expand_dims(sigma_,axis=0)
sigma_ = np_to_torch(sigma_).type(dtype).cuda()
#sigma_ = np_to_torch(sigma_).type(dtype)
#sigma_ = torch.from_numpy(sigma_).cuda() 
#sigma_ = sigma_.cuda()
#sigma_ = noise_level /100


print(sigma_.shape)
# Apply the brain mask to noisy data.
img_noisy_np_nonskull = img_noisy_np * img_np_mask
print(img_noisy_np_nonskull.shape)
# Apply the brain mask to reference data.
img_np_nonskull = img_np * img_np_mask


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
input_depth = 31
# Initialize the network.
net = get_net(input_depth, 'skip', pad,skip_n33d=32,skip_n33u=32,skip_n11=4,num_scales=4,upsample_mode='trilinear',n_channels=31).type(dtype)




# Prepare the five-dimensional DIP input tensor.
net_input,_ = load_nifti('data/generate_data/noisy_input.nii.gz')
net_input = net_input[:,:,20:,:]
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
bp = False
if bp:
    ii = np.load('outputs/DIP_M2_W/generate_data_2/noisemap/iterations/level_5/i.npy')
    i = np.load('outputs/DIP_M2_W/generate_data_2/noisemap/iterations/level_5/i.npy')
    i = i-(i%40)
    LR=0.01*0.9**(i//2000)

    trained_model_name = 'outputs/DIP_M2_W/generate_data_2/noisemap/trained_model/level_5/epoch_' + str(i) +'.pt'
    #trained_model_name_last = 'outputs/DIP_M2_W/sigma_Ablation/trained_model/level_5_sigma3/epoch_' + str(i-1) +'.pt'
    #net.load_state_dict(torch.load(trained_model_name_last))
    #out_bp = net(net_input).data
    net.load_state_dict(torch.load(trained_model_name))

    psrn_out_list = list(np.load('outputs/DIP_M2_W/generate_data_2/noisemap/psnr_rmse/level_5/psnr_level_5.npy'))
    total_loss_list = list(np.load('outputs/DIP_M2_W/generate_data_2/noisemap/psnr_rmse/level_5/loss_level_5.npy'))
    rmse_out_list = list(np.load('outputs/DIP_M2_W/generate_data_2/noisemap/psnr_rmse/level_5/rmse_level_5.npy'))

    for k in range(i,ii+1):
        del total_loss_list[i]
        del psrn_out_list[i]
        del rmse_out_list[i]

net_input_saved = net_input.detach().clone()   
noise = net_input.detach().clone()              
# print(type(net_input))
out_avg = None
last_net = None
psrn_noisy_last = 0              
total_loss_last = 9999
psrn_gt_last = 0
t=0

mask_num_npy = np.sum(img_np_mask)
mask_num = np.sum(img_np_mask)
mask_num=128*160*111/mask_num
mask_num = torch.tensor(mask_num,dtype=torch.float32,device='cuda',requires_grad=False)  
mask_num_sqrt = torch.sqrt(mask_num)

last_i=0
def closure():   #######！！！！！#####
    
    global last_i,total_loss_last,num_iter,t,psrn_gt_last,LR,i, out_avg, psrn_noisy_last, last_net, net_input, psrn_noisy_list, psrn_gt_list,total_loss_list   #Denoising PSNR state.  
    out = net(net_input)    # <class 'torch.Tensor'>
# Output from the previous iteration.
    out_inside_last = out.data
    M2_weight = torch.sqrt(out_inside_last**2 + sigma_**2)*2*sigma_
    total_loss = mse((out**2+2*(sigma_**2))/M2_weight, (img_noisy_torch**2/M2_weight))*mask_num_sqrt
    total_loss.backward() 
    total_loss_ = total_loss.data.cpu().numpy()
    total_loss_ = float(total_loss_)
    out_np = out.detach().cpu().numpy()[0]  #Convert the output to a NumPy array.
    out_np_nonskull = out_np * img_np_mask
    # out_np_nonskull_gt = out_np_nonskull
    
    RMSE_out = np.sqrt((np.sum((out_np_nonskull-img_np_nonskull)**2))/(mask_num_npy*input_depth))
    MSE_out = (np.sum((out_np_nonskull-img_np_nonskull)**2))/(mask_num_npy*input_depth)
    PSNR_out = 20*np.log10(1/RMSE_out)

    RMSE_noisy = np.sqrt((np.sum((out_np_nonskull-img_noisy_np_nonskull)**2))/(mask_num_npy*input_depth))
    PSNR_noisy = 20*np.log10(1/RMSE_noisy)
    psrn_noisy_list.append(PSNR_noisy)
    psnr_noisy_array = np.array(psrn_noisy_list)

    psrn_out_list.append(PSNR_out)
    # mse_out_list.append(MSE_out)   
    rmse_out_list.append(RMSE_out)   
    total_loss_list.append(total_loss_)   #Record the loss value.  

    path_main_1 = 'outputs/DIP_M2_W/generate_data_2/noisemap/psnr_rmse/level_5/'
    if not os.path.exists(path_main_1):
        os.makedirs(path_main_1)
    path_main_2 = 'outputs/DIP_M2_W/generate_data_2/noisemap/iterations/level_5/'
    if not os.path.exists(path_main_2):
        os.makedirs(path_main_2)
    path_main_3 = 'outputs/DIP_M2_W/generate_data_2/noisemap/result/level_5/'
    if not os.path.exists(path_main_3):
        os.makedirs(path_main_3)
    path_main_4= 'outputs/DIP_M2_W/generate_data_2/noisemap/loss/level_5/'
    if not os.path.exists(path_main_4):
        os.makedirs(path_main_4)
    path_main_5 = 'outputs/DIP_M2_W/generate_data_2/noisemap/trained_model/level_5/'
    if not os.path.exists(path_main_5):
        os.makedirs(path_main_5)
    path_main_6 = 'outputs/DIP_M2_W/generate_data_2/noisemap/model/level_5/'
    if not os.path.exists(path_main_6):
        os.makedirs(path_main_6)

    psnr_name =  path_main_1+'psnr_level_5.npy'
    psnr_noisy_name =  path_main_1+'psnr_noise_level_5.npy'
    # mse_name = 'outputs/DIP_M1_W/generate_data_2/psnr_rmse/level_5/mse_level_5.npy'
    rmse_name =  path_main_1+'rmse_level_5.npy'
    loss_name =  path_main_1+'loss_level_5.npy'
    psnr_array = np.array(psrn_out_list)
    np.save(psnr_name, np.array(psrn_out_list))
    np.save(psnr_noisy_name, np.array(psrn_noisy_list))
    np.save(rmse_name, np.array(rmse_out_list))   
    np.save(loss_name, np.array(total_loss_list))
    iteration_name = path_main_2+'i.npy'  
    np.save(iteration_name, i)
    #Save the metric history.

    # if (i+1)%show_every == 0:
    #     model_name_last = 'outputs/DIP_M2_W/generate_data/trained_model/level_5/epoch_' + str(i) +'.pt'
    #     torch.save(net.state_dict(),model_name_last)
    print ('Iteration %05d  Loss %f  PSNR_out: %f  PSNR_noisy_out: %f   RMSE_out: %f' % (i, total_loss.item(), PSNR_out, PSNR_noisy, RMSE_out), '\n', end='')
    if  i % show_every == 0:
        #Periodically visualize the denoising result.
        if i% (4*show_every) ==0:
            out_np = torch_to_np(out)
            X = 63
            
            out_np_X = out_np[0, :, :, X]
            out_np_Y = out_np[1, :, :, X]
            fig_name = path_main_3+'epoch_' + str(i) +'.png'
            plt.figure(figsize=(8,1))
            plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1), plt.axis('off')
            plt.subplot(1,8,1), plt.imshow(img_np_nonskull[0, :, :, 63],vmin=0,vmax=0.5, cmap='gray'), plt.axis('off')
            plt.subplot(1,8,2), plt.imshow(img_np_nonskull[1, :, :, 63],vmin=0,vmax=0.3, cmap='gray'), plt.axis('off')
            plt.subplot(1,8,3), plt.imshow(img_noisy_np_nonskull[0, :, :, 63],vmin=0,vmax=0.5, cmap='gray'), plt.axis('off') 
            plt.subplot(1,8,4), plt.imshow(img_noisy_np_nonskull[1, :, :, 63],vmin=0,vmax=0.3, cmap='gray'), plt.axis('off')
            plt.subplot(1,8,5), plt.imshow(out_np_X,vmin=0,vmax=0.5,cmap='gray'), plt.axis('off')
            plt.subplot(1,8,6), plt.imshow(out_np_Y,vmin=0,vmax=0.3,cmap='gray') , plt.axis('off') 
            plt.subplot(1,8,7), plt.imshow((abs(img_np_nonskull[0, :, :, 63] - out_np_X)),vmin=0,vmax=0.15,cmap='gray') , plt.axis('off')
            plt.subplot(1,8,8), plt.imshow((abs(img_np_nonskull[1, :, :, 63] - out_np_Y)),vmin=0,vmax=0.09,cmap='gray') , plt.axis('off')
            plt.savefig(fig_name)
        loss_name =path_main_4+ 'epoch_' + str(i%(show_every*2)) +'.pt'
        model_name = path_main_5+'epoch_' + str(i) +'.pt'
        model_name_ = path_main_6+'epoch_' + str(i%(show_every*2)) +'.pt'
        torch.save(total_loss,loss_name)#Save the checkpoint.
        torch.save(net.state_dict(),model_name_)
        torch.save(net.state_dict(),model_name)#Save the checkpoint.

        
        #if  i!=0 and i!=show_every and psnr_noisy_array[i-show_every+1:i+1].min() - psnr_noisy_array[i-2*show_every+1:i-show_every+1].min() < -0.05:  
        #    if last_i != i:
        #        t = 5
        #    if last_i == i:
        #        t = t-1
        #    last_i =i
        #    if t==0:
        #        if psnr_noisy_array[i].min() - psnr_noisy_array[i-show_every].min()<-0.5:
        #            t=1
        #    if t >0:
        #        i = i-show_every
        #        net.load_state_dict(torch.load('outputs/DIP_M2_W/sigma_Ablation/model/level_5_sigma3/epoch_' + str(i%(show_every*2)) +'.pt'))
        #        total_loss = torch.load('outputs/DIP_M2_W/sigma_Ablation/loss/level_5_sigma3/epoch_' + str(i%(show_every*2)) +'.pt')
        #        total_loss.backward()
        #        for k in range(i+1,i+show_every+1):
        #            del total_loss_list[i+1]
        #            del psrn_out_list[i+1]
        #            del rmse_out_list[i+1]
        #            del psrn_noisy_list[i+1]
        #        total_loss_last = total_loss_list[i]
        #        out = net(net_input)
        #        out_inside_last = out.data
        #    else:
        #        total_loss_last = total_loss
        #else:
        #    total_loss_last = total_loss
        
    LR=0.01*0.9**(i//2000)
    i += 1      
    return total_loss_list,psrn_out_list,rmse_out_list

p = get_params(OPT_OVER, net, net_input) 
optimizer = torch.optim.Adam(p, lr=LR)
for j in range(num_iter):
        optimizer.zero_grad()
        optimizer.step(closure)
