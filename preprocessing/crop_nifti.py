"""Optionally crop NIfTI data to reduce storage and GPU memory use.

User configuration:
    Key parameters to edit: input/output paths, crop or background-patch indices, volume index, and normalization/noise-estimation settings.
    Relative paths assume execution from the repository root.
"""

import numpy as np
from dipy.io.image import load_nifti,save_nifti

#data,affine = load_nifti("data/MGH-HCP/new/noisy_data.nii.gz")
#input_data,affine = load_nifti("data/MGH-HCP/new/noisy_input.nii.gz")
mask,affine = load_nifti("data/MGH-HCP/new/mask.nii.gz")
#data_crop = data[5:133,5:133,18:,:] $slice=48

#data_crop = data[19:113,5:115,20:40,:] #2026.7.14 #slice=20
#noisy_input = input_data[19:113,5:115,20:40,:]
mask_crop = mask[19:113,5:115,20:40]
#data_crop = data_crop.astype(np.float32)
#noisy_input = np.random.standard_normal(data_crop.shape).astype(np.float32)
#save_nifti("data/MGH-HCP/new/new_new/noisy_data.nii.gz",data_crop,affine)
#save_nifti('data/MGH-HCP/new/new_new/noisy_input.nii.gz',noisy_input,affine)
save_nifti('data/MGH-HCP/new/new_new/mask.nii.gz',mask_crop,affine)
