"""
Apply global min-max normalization to a 4D in-vivo DWI dataset.

This preprocessing step rescales all voxel intensities to the range [0, 1].
It is suitable for datasets without substantial outliers. If the data contain
outliers, NaN values, or infinite values, use `clip_and_normalize.py` instead.

Before running, update:
    - Input and output NIfTI file paths.
    - The normalization range if values other than [0, 1] are required.

Relative paths assume that the script is run from the repository root.
"""

import numpy as np
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti

data,affine = load_nifti("data/in-vivo/raw/data.nii.gz")
data_nor = (data - np.min(data))/(np.max(data)-np.min(data))
data_nor = data_nor.astype(np.float32)
save_nifti('data/in-vivo/raw/data_nor.nii.gz',data_nor,affine)
