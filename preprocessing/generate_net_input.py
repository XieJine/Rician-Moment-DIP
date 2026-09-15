"""
Optionally crop the in-vivo DWI data to reduce GPU memory use and
computational cost during Deep Image Prior optimization.

Parameters to update before running:
    - Input DWI and mask paths.
    - Crop ranges for the x, y, and z dimensions.
    - Output file paths.

The generated `noisy_input.nii.gz` is a fixed Gaussian random tensor
used as the network input for DIP optimization.
"""

import numpy as np
from dipy.io.image import load_nifti,save_nifti

data,affine = load_nifti("data/in-vivo/noisy_data_raw.nii.gz")
mask,affine = load_nifti("data/in-vivo/mask.nii.gz")

data_crop = data[19:113,5:115,20:40,:] 
mask_crop = mask[19:113,5:115,20:40]
data_crop = data_crop.astype(np.float32)
noisy_input = np.random.standard_normal(data_crop.shape).astype(np.float32)
save_nifti("data/in-vivo/noisy_data.nii.gz",data_crop,affine)
save_nifti('data/in-vivo/noisy_input.nii.gz',noisy_input,affine)
save_nifti('data/in-vivo/mask.nii.gz',mask_crop,affine)
