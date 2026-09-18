# Rician-Moment-DIP

## Unsupervised Denoising of Diffusion-Weighted Images with Bias- and Variance-Corrected Noise Modeling

Rician-Moment-DIP is an image-specific, unsupervised diffusion MRI (dMRI) denoising framework based on Deep Image Prior (DIP). It is designed for magnitude diffusion-weighted images (DWIs), where Rician noise can introduce signal bias and heteroscedastic noise variance, particularly at low signal-to-noise ratios. The framework incorporates two Rician noise-aware loss functions: a first-moment loss for bias correction and a second-moment loss based on squared-signal statistics. Both objectives further use adaptive variance weighting to account for the signal-dependent nature of Rician noise, without modifying the underlying DIP network architecture.

The repository provides two Rician noise-aware, variance-weighted loss functions within the DIP framework:

- **DIP-M1-W** (`DIP_M1_W/`): a first-moment Rician loss that uses the conditional expectation of magnitude data to correct Rician-induced signal bias. The residual is adaptively weighted using the signal-dependent Rician noise variance.
- **DIP-M2-W** (`DIP_M2_W/`): a second-moment Rician loss based on \(\mathbb{E}[M^2] = S^2 + 2\sigma^2\), which performs bias correction in the squared-signal domain and applies second-moment variance weighting.

Both methods perform image-specific, unsupervised optimization for each DWI acquisition. They require neither clean reference images nor pretrained denoising models, and they do not modify the underlying DIP network architecture.

## Requirements

The implementation was tested with Python 3.8, PyTorch 2.0.0, and CUDA 11.7.

For a CUDA-enabled GPU installation, Conda is recommended:

```bash
conda env create -f environment.yml
conda activate rician-moment-dip
```

Alternatively, install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

`requirements.txt` may install a CPU-only PyTorch build, depending on the platform. Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) if a GPU build is required.

Run all commands from the repository root:

```bash
cd Rician-Moment-DIP
```

## Simulated data

The simulated data used in the experiments are available from the [GitHub Release](https://github.com/XieJine/Rician-DIP-dMRI-Denoising/releases/latest).

After extraction, the following files are expected by the default simulation scripts:

```text
data/generate_data/
├── dwi_reference.nii.gz          # Noise-free reference DWI
├── dwi_level_5.nii.gz            # Uniform-noise DWI (sigma = 0.05)
├── dwi_level_3_5.nii.gz          # Spatially varying-noise DWI
├── noise_map_level_3_5.nii.gz    # Spatial noise standard-deviation map
├── noisy_input.nii.gz            # Fixed Gaussian DIP input
└── mask.nii.gz                   # Brain mask
```

The supplied simulation scripts assume normalized image intensities. Update the data paths, crop ranges, number of volumes, and noise settings before running a new experiment.

## In-vivo data preparation

Raw in-vivo data are not included. Use de-identified data only.

The default preprocessing workflow is:

1. `preprocessing/normalize_minmax.py` rescales a 4D DWI dataset to \([0, 1]\). Use it when the data do not contain substantial outliers.
2. `preprocessing/clip_and_normalize.py` clips extreme values in each volume before normalization. Use it instead when outliers, NaN values, or infinite values are present.
3. `preprocessing/estimate_background_noise.py` estimates the background noise standard deviation from a user-selected Rayleigh background patch.
4. `preprocessing/generate_net_input.py` optionally crops the DWI and mask and generates the fixed Gaussian network input used by DIP.

Before execution, edit the paths and crop/background-patch indices at the top of each preprocessing script. The DWI, mask, and DIP input must have matching spatial dimensions.

## Training and result generation

Each training script saves model checkpoints and intermediate metrics under `outputs/`. Each corresponding `make_result_*.py` script loads one selected checkpoint and exports a denoised NIfTI volume.

| Experiment | DIP-M1-W | DIP-M2-W |
| --- | --- | --- |
| Simulation: uniform Rician noise | `train_simulation_uniform.py` → `make_result_simulation_uniform.py` | `train_simulation_uniform.py` → `make_result_simulation_uniform.py` |
| Simulation: spatially varying Rician noise | `train_simulation_spatial.py` → `make_result_simulation_spatial.py` | `train_simulation_spatial.py` → `make_result_simulation_spatial.py` |
| In-vivo DWI | `train_in_vivo.py` → `make_result_in_vivo.py` | `train_in_vivo.py` → `make_result_in_vivo.py` |

For example, to train DIP-M1-W on the uniform-noise simulation:

```bash
python DIP_M1_W/train_simulation_uniform.py
```

After choosing a checkpoint epoch, set `epoch` and `net_name` in the paired result-generation script, then run:

```bash
python DIP_M1_W/make_result_simulation_uniform.py
```

## Checkpoint selection

### Simulated data

For simulated data, a clean reference is available. Use PSNR and loss curves to select the checkpoint:

```bash
python select_model/plot_simulation_training_curve.py
python select_model/select_model_simulation.py
```

### In-vivo data

For in-vivo data, `select_model/compute_in_vivo_SSIM-DA.py` calculates offline SSIM-DA values across saved checkpoints. `select_model/plot_in_vivo_SSIMM_AD.py` plots these curves and identifies candidate checkpoints.

SSIM-DA compares direction-averaged reconstructed and noisy DWIs within each b-value shell. Update `CHECKPOINT_DIR`, `SAVE_DIR`, `NOISY_PATH`, `INPUT_PATH`, `MASK_PATH`, and `BVAL_PATH` before running either script.

## Important configuration checks

The scripts are deliberately explicit rather than command-line driven. Before every run, check the following settings in the selected script:

- Input DWI, reference DWI, mask, fixed DIP input, noise-map, and output paths.
- GPU index in `CUDA_VISIBLE_DEVICES`.
- Number and order of DWI volumes (`input_depth` and `n_channels`).
- Scalar \(\sigma\) for uniform noise or the NIfTI noise-map path for spatially varying noise.
- Learning rate, number of iterations, checkpoint interval, and optional resume settings.
- Spatial crop range and display slice.
- The selected checkpoint epoch and its exact output directory.

The model architecture, number of channels, data crop, and DWI ordering must be identical between a training script and its paired `make_result_*.py` script; otherwise the checkpoint cannot be loaded or the exported result will be invalid.

## Repository structure

```text
Rician-Moment-DIP/
├── preprocessing/       # Normalization, noise estimation, and DIP-input generation
├── DIP_M1_W/            # First-moment Rician DIP training and result export
├── DIP_M2_W/            # Second-moment Rician DIP training and result export
├── select_model/        # Simulation and in-vivo checkpoint-selection utilities
├── models/              # 3D DIP skip-network and additional architectures
├── utils/               # Shared image, optimization, and denoising utilities
├── requirements.txt
└── environment.yml
```

## Reproducibility and data privacy

- Do not commit raw participant data, derived NIfTI images, checkpoints, or experiment outputs.
- Add `data/`, `outputs/`, `*.nii.gz`, `*.pt`, and Python cache directories to `.gitignore` before running new experiments.
- The supplied script defaults are experiment-specific examples. They must be adapted to the local dataset before use.

## Citation

If you use this code, please cite the associated manuscript. Citation details will be added after publication.

## License

No license is currently specified. Please contact the authors before reusing the code outside the scope permitted by the associated work.
