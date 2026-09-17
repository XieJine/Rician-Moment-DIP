from __future__ import print_function

"""Compute offline in-vivo SSIM-DA curves for saved checkpoints.

User configuration:
    Key parameters to edit: metric/checkpoint paths, save directory, checkpoint interval, iteration range, b-value metadata, mask options, plot limits, and selection window.
    Relative paths assume execution from the repository root.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from dipy.io.image import load_nifti
from nibabel.processing import resample_from_to
from skimage.metrics import structural_similarity

from models import *


# =====================================================================
# 1. Configuration
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------
# Folder containing all epoch_*.pt checkpoint files.
CHECKPOINT_DIR = Path(
    "/root/xje/DIP_M2_W_3D/mgh_hcp2/"
    "trained_model"
)


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
SAVE_DIR = Path(
    "/root/xje/DIP_M2_W_3D/mgh_hcp2/"
    "offline_metrics_in_vivo_ssim_da"
)


# ---------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------
NOISY_PATH = Path(
    "/root/xje/DIP_data/MGH-HCP/new/"
    "new_new/noisy_data.nii.gz"
)

INPUT_PATH = Path(
    "/root/xje/DIP_data/MGH-HCP/new/"
    "new_new/noisy_input.nii.gz"
)

MASK_PATH = Path(
    "/root/xje/DIP_data/MGH-HCP/new/"
    "new_new/mask.nii.gz"
)

BVAL_PATH = Path(
    "/root/xje/DIP_data/MGH-HCP/new/"
    "bval.txt"
)


# ---------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------
AUTO_RESAMPLE_MASK = True
MASK_THRESHOLD = 0.5


# ---------------------------------------------------------------------
# b-value grouping
# ---------------------------------------------------------------------
B0_THRESHOLD = 50.0

# Values such as 1499.8 and 1501.2 are grouped into b=1500.
BVALUE_ROUND_TO = 50.0


# ---------------------------------------------------------------------
# Network architecture
# Must exactly match the training configuration.
# ---------------------------------------------------------------------
PAD = "reflection"
SKIP_N33D = 261
SKIP_N33U = 261
SKIP_N11 = 4
NUM_SCALES = 4
UPSAMPLE_MODE = "trilinear"


# ---------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------
# Data are assumed to be normalized approximately to [0, 1].
DATA_RANGE = 1.0


# Save intermediate results every N checkpoints.
SAVE_EVERY_N_CHECKPOINTS = 10


# =====================================================================
# 2. Checkpoint utilities
# =====================================================================

def extract_epoch(path: Path) -> int:
    """Extract iteration number from epoch_XXXX.pt."""
    match = re.search(r"epoch_(\d+)\.pt$", path.name)

    if match is None:
        raise ValueError(
            f"Cannot parse epoch from checkpoint name: {path.name}"
        )

    return int(match.group(1))


def list_checkpoints(folder: Path):
    """Find and sort all epoch_*.pt checkpoints."""

    checkpoints = []

    for path in folder.glob("epoch_*.pt"):
        try:
            checkpoints.append(
                (extract_epoch(path), path)
            )
        except ValueError:
            continue

    checkpoints.sort(
        key=lambda item: item[0]
    )

    return checkpoints


def load_state_dict_robust(path: Path, device):
    """Load a checkpoint saved in several common PyTorch formats."""

    try:
        checkpoint = torch.load(
            str(path),
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            str(path),
            map_location=device,
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Unsupported checkpoint format: {path}"
        )

    # Remove DataParallel prefix if necessary.
    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


# =====================================================================
# 3. b-value grouping
# =====================================================================

def load_bvalues(path: Path, expected_count: int):
    """Load one b-value for each DWI volume."""

    if not path.exists():
        raise FileNotFoundError(
            f"b-value file not found: {path}"
        )

    bvalues = np.asarray(
        np.loadtxt(str(path)),
        dtype=np.float64,
    ).reshape(-1)

    if bvalues.size != expected_count:
        raise ValueError(
            f"The bval file contains {bvalues.size} values, "
            f"but the image contains {expected_count} volumes."
        )

    if not np.all(np.isfinite(bvalues)):
        raise ValueError(
            "The bval file contains NaN or infinite values."
        )

    return bvalues


def make_bvalue_groups(
    bvalues,
    b0_threshold=50.0,
    round_to=50.0,
):
    """
    Group diffusion volumes according to b-value.

    Returns
    -------
    shell_labels : ndarray [K]
        Representative b-values:
        [b0, b-shell1, b-shell2, ...]

    shell_indices : list of ndarrays
        Volume indices belonging to each shell.
    """

    bvalues = np.asarray(
        bvalues,
        dtype=np.float64,
    ).reshape(-1)

    # b0
    b0_idx = np.flatnonzero(
        bvalues <= float(b0_threshold)
    ).astype(np.int64)

    # non-zero diffusion-weighted volumes
    dwi_idx = np.flatnonzero(
        bvalues > float(b0_threshold)
    ).astype(np.int64)

    if b0_idx.size == 0:
        raise ValueError(
            f"No b0 volumes found using "
            f"B0_THRESHOLD={b0_threshold}."
        )

    if dwi_idx.size == 0:
        raise ValueError(
            "No diffusion-weighted volumes found."
        )

    rounded_dwi = (
        np.rint(
            bvalues[dwi_idx] / float(round_to)
        )
        * float(round_to)
    )

    dwi_shells = np.unique(rounded_dwi)
    dwi_shells.sort()

    shell_labels = [0.0]
    shell_indices = [b0_idx]

    for shell in dwi_shells:

        indices = dwi_idx[
            np.isclose(
                rounded_dwi,
                shell,
            )
        ]

        if indices.size == 0:
            continue

        shell_labels.append(
            float(shell)
        )

        shell_indices.append(
            indices.astype(np.int64)
        )

    return (
        np.asarray(
            shell_labels,
            dtype=np.float64,
        ),
        shell_indices,
    )


# =====================================================================
# 4. Brain mask
# =====================================================================

def prepare_mask_for_image(
    mask_data,
    mask_affine,
    target_spatial_shape,
    target_affine,
):
    """
    Prepare a 3D binary brain mask on the same voxel grid as the DWI data.
    """

    mask_array = np.asarray(
        mask_data,
        dtype=np.float32,
    )

    mask_array = np.squeeze(mask_array)

    if mask_array.ndim != 3:
        raise ValueError(
            "Mask must be 3D after squeezing, "
            f"but got {mask_array.shape}."
        )

    shape_matches = (
        tuple(mask_array.shape)
        == tuple(target_spatial_shape)
    )

    affine_matches = np.allclose(
        np.asarray(
            mask_affine,
            dtype=np.float64,
        ),
        np.asarray(
            target_affine,
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-4,
    )

    if not shape_matches or not affine_matches:

        if not AUTO_RESAMPLE_MASK:
            raise ValueError(
                "Mask and DWI image are on different grids."
            )

        print(
            "[Info] Resampling mask to DWI grid..."
        )

        source_mask = nib.Nifti1Image(
            mask_array,
            np.asarray(
                mask_affine,
                dtype=np.float64,
            ),
        )

        resampled_mask = resample_from_to(
            source_mask,
            (
                tuple(
                    int(v)
                    for v in target_spatial_shape
                ),
                np.asarray(
                    target_affine,
                    dtype=np.float64,
                ),
            ),
            order=0,
        )

        mask_array = np.asarray(
            resampled_mask.get_fdata(
                dtype=np.float32
            ),
            dtype=np.float32,
        )

    mask_bool = (
        mask_array > float(MASK_THRESHOLD)
    )

    if not np.any(mask_bool):
        raise ValueError(
            "Prepared brain mask is empty."
        )

    print(
        "Mask voxel count :",
        np.count_nonzero(mask_bool),
    )

    return mask_bool


def masked_bbox(mask_bool):
    """Return the bounding box of the brain mask."""

    coords = np.argwhere(mask_bool)

    if coords.size == 0:
        raise ValueError(
            "The brain mask is empty."
        )

    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1

    return tuple(
        slice(
            int(lo[d]),
            int(hi[d]),
        )
        for d in range(3)
    )


# =====================================================================
# 5. Direction averaging
# =====================================================================

def build_shell_mean_multichannel(
    data_chwd,
    shell_indices,
):
    """
    Convert original diffusion volumes into shell-wise
    direction-averaged images.

    Input
    -----
    data_chwd : [C, H, W, D]

    Output
    ------
    shell_mean_data : [H, W, D, K]

    Channel order:
        [b0 mean,
         first non-zero b-value mean,
         second non-zero b-value mean,
         ...]
    """

    data_chwd = np.asarray(
        data_chwd,
        dtype=np.float32,
    )

    channels = []

    for indices in shell_indices:

        indices = np.asarray(
            indices,
            dtype=np.int64,
        ).reshape(-1)

        if indices.size == 0:
            raise ValueError(
                "An empty b-value group was encountered."
            )

        # -------------------------------------------------------------
        # Direction averaging within one b-value shell
        # -------------------------------------------------------------
        shell_mean = np.mean(
            data_chwd[indices],
            axis=0,
            dtype=np.float32,
        )

        channels.append(
            shell_mean
        )

    return np.stack(
        channels,
        axis=-1,
    ).astype(
        np.float32,
        copy=False,
    )


# =====================================================================
# 6. SSIM-DA
# =====================================================================

def calculate_ssim_da(
    noisy_da,
    reconstructed_da,
    mask_bool,
    bbox,
    data_range=1.0,
):
    """
    Calculate SSIM-DA between:

        noisy direction-averaged images

    and

        reconstructed direction-averaged images.

    Both inputs have shape:

        [H, W, D, K]

    where K is the number of b-value shells.
    """

    noisy_da = np.asarray(
        noisy_da,
        dtype=np.float32,
    )

    reconstructed_da = np.asarray(
        reconstructed_da,
        dtype=np.float32,
    )

    if noisy_da.shape != reconstructed_da.shape:
        raise ValueError(
            "SSIM-DA input shapes differ: "
            f"{noisy_da.shape} vs "
            f"{reconstructed_da.shape}"
        )

    if noisy_da.ndim != 4:
        raise ValueError(
            "Expected shell-mean data with shape "
            f"[H,W,D,K], got {noisy_da.shape}."
        )

    # Apply brain mask.
    mask_4d = (
        mask_bool[..., None]
        .astype(np.float32)
    )

    full_slices = (
        bbox + (slice(None),)
    )

    noisy_crop = (
        noisy_da * mask_4d
    )[full_slices]

    reconstructed_crop = (
        reconstructed_da * mask_4d
    )[full_slices]

    # SSIM window size.
    min_spatial_dim = min(
        noisy_crop.shape[:3]
    )

    if min_spatial_dim < 3:
        return np.nan

    win_size = min(
        7,
        min_spatial_dim,
    )

    if win_size % 2 == 0:
        win_size -= 1

    kwargs = {
        "data_range": float(data_range),
        "win_size": win_size,
    }

    try:
        value = structural_similarity(
            noisy_crop,
            reconstructed_crop,
            channel_axis=-1,
            **kwargs,
        )

    except TypeError:
        # Compatibility with older scikit-image versions.
        value = structural_similarity(
            noisy_crop,
            reconstructed_crop,
            multichannel=True,
            **kwargs,
        )

    return float(value)


# =====================================================================
# 7. Save results
# =====================================================================

def save_results(
    epochs,
    ssim_da_values,
    save_dir,
):
    """Save SSIM-DA values for all processed checkpoints."""

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        save_dir / "epochs.npy",
        np.asarray(
            epochs,
            dtype=np.int64,
        ),
    )

    np.save(
        save_dir / "ssim_da.npy",
        np.asarray(
            ssim_da_values,
            dtype=np.float64,
        ),
    )


# =====================================================================
# 8. Main
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Check files
    # -----------------------------------------------------------------

    if not NOISY_PATH.exists():
        raise FileNotFoundError(
            f"Noisy data not found: {NOISY_PATH}"
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Network input not found: {INPUT_PATH}"
        )

    if not MASK_PATH.exists():
        raise FileNotFoundError(
            f"Brain mask not found: {MASK_PATH}"
        )

    if not BVAL_PATH.exists():
        raise FileNotFoundError(
            f"b-value file not found: {BVAL_PATH}"
        )


    # -----------------------------------------------------------------
    # Checkpoints
    # -----------------------------------------------------------------

    checkpoints = list_checkpoints(
        CHECKPOINT_DIR
    )

    if len(checkpoints) == 0:
        raise RuntimeError(
            "No epoch_*.pt checkpoints were found in "
            f"{CHECKPOINT_DIR}"
        )

    print(
        f"Device          : {DEVICE}"
    )

    print(
        f"Checkpoint count: {len(checkpoints)}"
    )

    print(
        f"First/last epoch: "
        f"{checkpoints[0][0]} / "
        f"{checkpoints[-1][0]}"
    )

    print(
        f"Save directory  : {SAVE_DIR}"
    )


    # -----------------------------------------------------------------
    # Load DWI data
    # -----------------------------------------------------------------

    noisy_hwdc, affine = load_nifti(
        str(NOISY_PATH)
    )

    input_hwdc, input_affine = load_nifti(
        str(INPUT_PATH)
    )

    mask_data, mask_affine = load_nifti(
        str(MASK_PATH)
    )

    noisy_hwdc = np.asarray(
        noisy_hwdc,
        dtype=np.float32,
    )

    input_hwdc = np.asarray(
        input_hwdc,
        dtype=np.float32,
    )

    if noisy_hwdc.ndim != 4:
        raise ValueError(
            f"Noisy data must be 4D: "
            f"{noisy_hwdc.shape}"
        )

    if input_hwdc.shape != noisy_hwdc.shape:
        raise ValueError(
            "Network input and noisy data have "
            "different shapes."
        )

    if not np.allclose(
        input_affine,
        affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Network input and noisy data "
            "use different voxel grids."
        )

    input_depth = int(
        noisy_hwdc.shape[-1]
    )

    print(
        f"Input depth      : {input_depth}"
    )


    # -----------------------------------------------------------------
    # Brain mask
    # -----------------------------------------------------------------

    mask_bool = prepare_mask_for_image(
        mask_data=mask_data,
        mask_affine=mask_affine,
        target_spatial_shape=noisy_hwdc.shape[:3],
        target_affine=affine,
    )

    bbox = masked_bbox(
        mask_bool
    )


    # -----------------------------------------------------------------
    # Rearrange:
    #
    # [H,W,D,C] -> [C,H,W,D]
    # -----------------------------------------------------------------

    noisy = noisy_hwdc.transpose(
        3, 0, 1, 2
    )

    net_input_np = input_hwdc.transpose(
        3, 0, 1, 2
    )[None]

    net_input = torch.from_numpy(
        net_input_np
    ).to(
        DEVICE,
        dtype=DTYPE,
    )


    # -----------------------------------------------------------------
    # b-value groups
    # -----------------------------------------------------------------

    bvalues = load_bvalues(
        BVAL_PATH,
        input_depth,
    )

    shell_labels, shell_indices = (
        make_bvalue_groups(
            bvalues,
            b0_threshold=B0_THRESHOLD,
            round_to=BVALUE_ROUND_TO,
        )
    )

    print(
        "\nSSIM-DA shell groups:"
    )

    for label, indices in zip(
        shell_labels,
        shell_indices,
    ):

        print(
            f"  b={label:g}: "
            f"{len(indices)} volumes, "
            f"indices={indices.tolist()}"
        )


    # -----------------------------------------------------------------
    # Noisy direction-averaged reference
    # -----------------------------------------------------------------

    noisy_masked = (
        noisy * mask_bool[None]
    )

    noisy_da = build_shell_mean_multichannel(
        noisy_masked,
        shell_indices,
    )

    print(
        "\nNoisy DA shape:",
        noisy_da.shape,
    )


    # -----------------------------------------------------------------
    # Build network
    # -----------------------------------------------------------------

    net = get_net(
        input_depth,
        "skip",
        PAD,
        skip_n33d=SKIP_N33D,
        skip_n33u=SKIP_N33U,
        skip_n11=SKIP_N11,
        num_scales=NUM_SCALES,
        upsample_mode=UPSAMPLE_MODE,
        n_channels=input_depth,
    ).to(
        DEVICE,
        dtype=DTYPE,
    )

    # Match original inference behavior.
    net.train()


    # -----------------------------------------------------------------
    # Calculate SSIM-DA for every checkpoint
    # -----------------------------------------------------------------

    epochs = []
    ssim_da_values = []

    print(
        "\n================ SSIM-DA calculation ================"
    )

    for number, (
        epoch,
        checkpoint_path,
    ) in enumerate(
        checkpoints,
        start=1,
    ):

        # Load checkpoint.
        state_dict = load_state_dict_robust(
            checkpoint_path,
            DEVICE,
        )

        net.load_state_dict(
            state_dict,
            strict=True,
        )

        net.train()

        # Network reconstruction.
        with torch.inference_mode():
            output = net(
                net_input
            )

        output_np = (
            output[0]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        # Brain mask.
        output_masked = (
            output_np
            * mask_bool[None]
        )

        # -------------------------------------------------------------
        # Reconstructed direction-averaged images
        # -------------------------------------------------------------
        reconstructed_da = (
            build_shell_mean_multichannel(
                output_masked,
                shell_indices,
            )
        )

        # -------------------------------------------------------------
        # SSIM-DA:
        #
        # reconstructed DA
        #       vs
        # noisy DA
        # -------------------------------------------------------------
        ssim_da = calculate_ssim_da(
            noisy_da,
            reconstructed_da,
            mask_bool,
            bbox,
            DATA_RANGE,
        )

        epochs.append(
            epoch
        )

        ssim_da_values.append(
            ssim_da
        )

        print(
            f"[{number:04d}/"
            f"{len(checkpoints):04d}] "
            f"epoch={epoch:6d}  "
            f"SSIM-DA={ssim_da:.6f}"
        )

        # Save intermediate results.
        if (
            number
            % SAVE_EVERY_N_CHECKPOINTS
            == 0
        ):
            save_results(
                epochs,
                ssim_da_values,
                SAVE_DIR,
            )

        del (
            state_dict,
            output,
            output_np,
            output_masked,
            reconstructed_da,
        )

        if (
            DEVICE.type == "cuda"
            and number % 50 == 0
        ):
            torch.cuda.empty_cache()


    # -----------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------

    save_results(
        epochs,
        ssim_da_values,
        SAVE_DIR,
    )

    np.save(
        SAVE_DIR
        / "ssim_da_channel_bvalues.npy",
        shell_labels,
    )


    print(
        "\n================ SSIM-DA completed ================"
    )

    print(
        "Saved iterations : "
        f"{SAVE_DIR / 'epochs.npy'}"
    )

    print(
        "Saved SSIM-DA    : "
        f"{SAVE_DIR / 'ssim_da.npy'}"
    )

    print(
        "Saved b-values   : "
        f"{SAVE_DIR / 'ssim_da_channel_bvalues.npy'}"
    )


if __name__ == "__main__":
    main()

