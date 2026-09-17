# Unsupervised Denoising of Diffusion-Weighted Images with Bias and Variance Corrected Noise Modeling

This repository contains the official implementation of our proposed method. 
The complete source code and configuration files will be released upon acceptance of the paper.
# Rician-Aware Deep Image Prior for dMRI Denoising

This repository contains the implementation used for Rician-aware, variance-weighted Deep Image Prior (DIP) denoising of diffusion MRI data. Two loss variants are provided:

- `DIP_M1_W`: first-moment Rician bias correction with variance weighting.
- `DIP_M2_W`: second-moment Rician bias correction with variance weighting.

## Important prerequisite

The submitted archive does **not** include the `models/` package, although every training and result-export script imports `get_net` from it. Before running the code, add the exact 3D DIP network implementation used for the study at:

```text
models/
    __init__.py
    ... files that define get_net(...)
```

Do not substitute a different architecture when reproducing reported results. The `utils/` package is included and has been consolidated from the two identical copies in the submitted archive.

## Installation

Create a Python environment with CUDA-enabled PyTorch appropriate for the local GPU, then install the remaining packages:

```bash
pip install -r requirements.txt
```

Run all scripts from the repository root so that imports such as `from utils.denoising_utils import *` resolve correctly.

## Repository layout

| Directory | Contents | Scripts to use |
| --- | --- | --- |
| `preprocessing/` | In-vivo data preparation and background-noise estimation. Cropping is optional. | `crop_nifti.py`, `normalize_minmax.py`, `clip_and_normalize.py`, `estimate_background_noise.py`, `estimate_noise_single_volume.py` |
| `DIP_M1_W/` | DIP-M1-W training and checkpoint-to-NIfTI export. | `train_simulation_uniform.py`, `train_simulation_spatial.py`, `train_in_vivo.py`, `make_result_*.py` |
| `DIP_M2_W/` | DIP-M2-W training and checkpoint-to-NIfTI export. | `train_simulation_uniform.py`, `train_simulation_spatial.py`, `train_in_vivo.py`, `make_result_*.py` |
| `select_model/` | Simulation training-curve plots and in-vivo offline SSIM checkpoint selection. | `plot_simulation_training_curve.py`, `inspect_simulation_training.py`, `compute_in_vivo_ssim.py`, `plot_in_vivo_ssim.py` |
| `evaluate/` | Reserved for final evaluation scripts and saved evaluation configurations. | See `evaluate/README.md` |
| `utils/` | Shared DIP helper functions retained from the supplied code. | Imported by training and result-export scripts |

## Data and output layout

All personal absolute paths were replaced with relative paths. Edit the configuration settings near the start of each script to match your local data. The default conventions are:

```text
data/
    generate_data/                 # Simulation data, masks, reference, DIP input
    in_vivo/                       # In-vivo data
    MGH-HCP/                       # Example preprocessing paths
outputs/
    DIP_M1_W/                      # M1 checkpoints, figures, metrics
    DIP_M2_W/                      # M2 checkpoints, figures, metrics
```

Data and outputs are ignored by Git to prevent publication of participant data, checkpoints, and large intermediate files.

## Recommended workflow

1. Put de-identified NIfTI data, masks, and DIP input tensors under `data/`.
2. Optionally run `preprocessing/crop_nifti.py` to reduce the spatial extent.
3. Run `clip_and_normalize.py` or `normalize_minmax.py`; select one normalization workflow consistent with the experiment.
4. Estimate the background noise standard deviation with `estimate_background_noise.py` or `estimate_noise_single_volume.py` and record the selected patch.
5. In the desired training script, update data paths, crop ranges, `noise_level`/`sigma_`, GPU index, `input_depth`, `num_iter`, and output directories.
6. Train one case at a time, then set the selected checkpoint epoch in the corresponding `make_result_*.py` script to export the denoised NIfTI image.
7. For simulations, inspect PSNR/loss curves. For in-vivo data, compute and plot offline SSIM curves to select a checkpoint.

## Parameters that must be checked before every run

Each executable script begins with an English description and configuration checklist. At minimum, verify:

- NIfTI/mask/reference/input paths and output directories.
- `CUDA_VISIBLE_DEVICES` and GPU availability.
- Crop range and display-slice index.
- Number and order of DWI volumes (`input_depth` / `n_channels`).
- Noise setting: scalar `noise_level`/`sigma_` for uniform noise or the noise-map path for spatially varying noise.
- Learning rate, iteration count, checkpoint interval, and optional resume settings.
- Selected checkpoint `epoch` when running `make_result_*.py`.
- b-values, mask behavior, metric directory, and selection windows for offline in-vivo SSIM.

## Notes for a public GitHub release

- Do not upload raw or derived participant data, trained model weights, output NIfTI files, or machine-specific paths.
- Add the missing `models/` source code before publishing.
- Add a license only after confirming the intended reuse terms and the provenance of the retained utility/network code.

## Simulated Data

The simulated data used in the experiments are available in the 
[GitHub Release](https://github.com/XieJine/Rician-DIP-dMRI-Denoising/releases/latest).

simulated_data.zip

├── dwi_reference.nii.gz

├── dwi_level_5.nii.gz  # noise level(sigma)=0.05 

├── dwi_level_3_5.nii.gz  #  non-uniform noise distributions 

├── noise_map_level_3_5.nii.gz  # 

├── noisy_input.nii.gz

└── mask.nii.gz
