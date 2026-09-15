"""Estimate background noise from a user-selected patch in one DWI volume.

User configuration:
    Key parameters to edit: input/output paths, crop or background-patch indices, volume index, and normalization/noise-estimation settings.
    Relative paths assume execution from the repository root.
"""

import numpy as np
from dipy.io.image import load_nifti,save_nifti

data, affine = load_nifti("data/MGH-HCP/raw/dwi_preproc_nor.nii.gz")  #0.035  #0.021

data_patch = data[10:45,90:120,89,0]
print(data_patch.shape)
data_patch_2 = data_patch**2
# data_patch_2 = np.reshape(data_patch_2,[data_patch_2.shape[0]*data_patch_2.shape[1],data_patch_2.shape[2]*data_patch_2.shape[3]])
data_patch_2_mean = np.mean(data_patch_2)
data_patch_2_var = np.var(data_patch_2)
noise_level = np.sqrt(data_patch_2_mean/2)
noise_level_2 = np.sqrt(np.sqrt(data_patch_2_var/4))
print(noise_level)
print(noise_level_2)
# print(data_patch_2_mean.shape)
# data_patch_2_mean_ = np.sqrt(data_patch_2_mean/2)
# noise_level = np.std(data_patch_2_mean_)
# print(noise_level)
# print(np.mean(data_patch_2_mean_))
