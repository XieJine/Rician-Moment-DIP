"""Apply global min-max normalization to a 4D DWI dataset.

User configuration:
    Key parameters to edit: input/output paths, crop or background-patch indices, volume index, and normalization/noise-estimation settings.
    Relative paths assume execution from the repository root.
"""

import numpy as np
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti

data,affine = load_nifti("data/MGH-HCP/raw/diff_32dir_b1k3k_64dir_b5k_128dir_b10k.nii.gz")
data_nor = (data - np.min(data))/(np.max(data)-np.min(data))
data_nor = data_nor.astype(np.float32)
save_nifti('data/MGH-HCP/raw/diff_32dir_b1k3k_64dir_b5k_128dir_b10k_nor.nii.gz',data_nor,affine)
