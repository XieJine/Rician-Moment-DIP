"""Estimate noise sigma from a user-selected Rayleigh background patch.

User configuration:
    Key parameters to edit: input/output paths, crop or background-patch indices, volume index, and normalization/noise-estimation settings.
    Relative paths assume execution from the repository root.
"""

import numpy as np
from dipy.io.image import load_nifti


def main():

    data, affine = load_nifti(
        "data/MGH-HCP/raw/dwi_preproc_nor.nii.gz"
    )

    data_patch = data[10:45, 90:120, 89, :]

    background = data_patch.reshape(-1)
    background = background[np.isfinite(background)]

    patch_mean = np.mean(background)
    patch_std = np.std(background)
    patch_rms = np.sqrt(np.mean(background ** 2))

    # Valid only for zero-signal Rayleigh background.
    sigma_from_second_moment = np.sqrt(
        np.mean(background ** 2) / 2.0
    )

    # Estimate sigma from the Rayleigh standard deviation.
    sigma_from_rayleigh_std = patch_std / np.sqrt(
        (4.0 - np.pi) / 2.0
    )

    print("Data shape:", data.shape)
    print("Patch shape:", data_patch.shape)
    print("Patch mean:", patch_mean)
    print("Patch std:", patch_std)
    print("Patch RMS:", patch_rms)

    print("\nMean / std:", patch_mean / patch_std)

    print(
        "Sigma from E[M^2]:",
        sigma_from_second_moment
    )

    print(
        "Sigma from Rayleigh std:",
        sigma_from_rayleigh_std
    )

    print(
        "Sigma using patch std:",
        patch_std
    )


if __name__ == "__main__":
    main()
