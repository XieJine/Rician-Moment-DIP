"""
Clip extreme intensity values in each DWI volume and then apply global
min-max normalization.

This preprocessing step is optional. It is recommended when the in-vivo
data contain outliers, NaN values, or infinite values that may adversely
affect subsequent DIP denoising.

Before running, update:
    - Input and output file paths.
    - LOW_PERCENTILE and HIGH_PERCENTILE for outlier clipping.
    - Normalization settings, if a different intensity range is required.

Relative paths assume that the script is run from the repository root.
"""

import os
import numpy as np
from dipy.io.image import load_nifti, save_nifti
# ============================================================
# 1. File paths (edit before running).
# ============================================================
data_path = (
    "data/in-vivo/raw/"
    "data.nii.gz"
)

save_clean_path = (
    "data/in-vivo/raw/"
    "dwi_preproc_clean.nii.gz"
)

save_nor_path = (
    "data/in-vivo/raw/"
    "dwi_preproc_nor.nii.gz"
)


# ============================================================
# 2. Processing parameters (edit before running).
# ============================================================
# Retain the central 99% of values.
# Clip 0.5% from each tail.
low_percentile = 0.01
high_percentile = 99.99


# ============================================================
# 3. Load the input data.
# ============================================================
data, affine, img = load_nifti(
    data_path,
    return_img=True
)

data = data.astype(np.float32)

if data.ndim != 4:
    raise ValueError(
        f"Input DWI must be 4D; received shape: {data.shape}"
    )

print("=" * 70)
print("DWI shape:", data.shape)
print("Raw global min:", np.nanmin(data))
print("Raw global max:", np.nanmax(data))
print("=" * 70)


# ============================================================
# 4. Clip outliers independently in each volume.
# ============================================================
data_clean = np.empty_like(data, dtype=np.float32)

for v in range(data.shape[-1]):

    vol = data[..., v]

    # Use finite values to compute percentiles.
    valid_values = vol[np.isfinite(vol)]

    if valid_values.size == 0:
        raise ValueError(
            f"Volume {v} contains no finite values."
        )

    # Retain the central 99% of values.
    low = np.percentile(valid_values, low_percentile)
    high = np.percentile(valid_values, high_percentile)

    # Replace NaN and infinite values first.
    vol_clean = np.nan_to_num(
        vol,
        nan=low,
        posinf=high,
        neginf=low
    )

    # Clip both tails to their percentile thresholds.
    vol_clean = np.clip(vol_clean, low, high)

    data_clean[..., v] = vol_clean.astype(np.float32)

    print(
        f"Volume {v:03d}: "
        f"raw min={np.nanmin(vol):.6f}, "
        f"raw max={np.nanmax(vol):.6f}, "
        f"clip low={low:.6f}, "
        f"clip high={high:.6f}, "
        f"clean min={vol_clean.min():.6f}, "
        f"clean max={vol_clean.max():.6f}"
    )


# ============================================================
# 5. Save the clipped data.
# ============================================================
os.makedirs(
    os.path.dirname(save_clean_path),
    exist_ok=True
)

save_nifti(
    save_clean_path,
    data_clean,
    affine,
    hdr=img.header
)

print("\n" + "=" * 70)
print("After outlier clipping")
print("Clean global min:", data_clean.min())
print("Clean global max:", data_clean.max())
print("Clean global mean:", data_clean.mean())
print("Clean global std:", data_clean.std())
print("=" * 70)


# ============================================================
# 6. Apply global min-max normalization after clipping.
# ============================================================
data_min = data_clean.min()
data_max = data_clean.max()
data_range = data_max - data_min

if data_range <= 1e-8:
    raise ValueError(
        "Data range is too small for min-max normalization: "
        f"min={data_min}, max={data_max}"
    )

data_nor = (
    data_clean - data_min
) / data_range

data_nor = np.clip(
    data_nor,
    0.0,
    1.0
).astype(np.float32)


# ============================================================
# 7. Report normalization statistics.
# ============================================================
print("\n" + "=" * 70)
print("Min-Max normalization")
print("Normalization min:", data_min)
print("Normalization max:", data_max)
print("Normalization range:", data_range)
print("Normalized global min:", data_nor.min())
print("Normalized global max:", data_nor.max())
print("Normalized global mean:", data_nor.mean())
print("Normalized global std:", data_nor.std())
print("=" * 70)


# ============================================================
# 8. Save normalized data.
# ============================================================
os.makedirs(
    os.path.dirname(save_nor_path),
    exist_ok=True
)

save_nifti(
    save_nor_path,
    data_nor,
    affine,
    hdr=img.header
)

print("\nProcessing complete:")
print("Clean data saved to:")
print(save_clean_path)

print("Normalized data saved to:")
print(save_nor_path)
