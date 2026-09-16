from __future__ import print_function

"""Compute offline in-vivo SSIM-DA curves for saved checkpoints.

User configuration:
    Key parameters to edit: metric/checkpoint paths, save directory, checkpoint interval, iteration range, b-value metadata, mask options, plot limits, and selection window.
    Relative paths assume execution from the repository root.
"""


from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity


# =====================================================================
# Configuration
# =====================================================================
NOISY_PATH = Path(
    "data/three_resolution/noedic/"
    "p9/noisy_data.nii.gz"
)
MASK_PATH = Path(
    "data/three_resolution/noedic/"
    "p9/mask_new.nii.gz"
)
BVAL_PATH = Path(
    "data/three_resolution/noedic/"
    "p9/AP.bval"
)
SAVE_DIR = Path(
    "outputs/DIP_M1_W/three_different_resolution2/"
    "p9/offline_metrics_in_vivo_ssim_only"
)

INPUT_DEPTH = 99
DATA_RANGE = 1.0
B0_THRESHOLD = 50.0
BVAL_GROUP_ROUND = 50.0

AUTO_RESAMPLE_MASK = True
MASK_THRESHOLD = 0.5

# Your 4D mask channels are not identical. To reproduce the evaluation
# currently used in SSIM_new.py, use the first channel as the 3D mask.
MASK_4D_MODE = "first"
MASK_MAJORITY_FRACTION = 0.5
# =====================================================================


def format_bvalue_label(b_value):
    b_value = float(b_value)
    if abs(b_value) <= B0_THRESHOLD:
        return "b0"
    if abs(b_value - round(b_value)) < 1e-6:
        return "b%d" % int(round(b_value))
    return ("b%.1f" % b_value).replace(".", "p")


def resolve_volume_groups(input_depth):
    if not BVAL_PATH.exists():
        raise FileNotFoundError("Missing b-value file: %s" % BVAL_PATH)

    bvals = np.asarray(
        np.loadtxt(str(BVAL_PATH)), dtype=np.float32
    ).reshape(-1)

    if len(bvals) != input_depth:
        raise ValueError(
            "b-values contain %d entries, but INPUT_DEPTH=%d."
            % (len(bvals), input_depth)
        )

    b0_idx = np.flatnonzero(bvals <= B0_THRESHOLD).astype(np.int64)
    dwi_idx = np.flatnonzero(bvals > B0_THRESHOLD).astype(np.int64)

    groups = []
    if len(b0_idx) > 0:
        groups.append(("b0", 0.0, b0_idx))

    dwi_bvals = bvals[dwi_idx]
    grouped_bvals = (
        np.round(dwi_bvals / float(BVAL_GROUP_ROUND))
        * float(BVAL_GROUP_ROUND)
    )

    for grouped_b in np.sort(np.unique(grouped_bvals)):
        member_mask = np.isclose(grouped_bvals, grouped_b, atol=1e-6)
        indices = dwi_idx[member_mask].astype(np.int64)
        groups.append(
            (format_bvalue_label(grouped_b), float(grouped_b), indices)
        )

    used_indices = np.concatenate([indices for _, _, indices in groups])
    if len(used_indices) != input_depth:
        raise ValueError(
            "The b-value groups contain %d volumes, but INPUT_DEPTH=%d."
            % (len(used_indices), input_depth)
        )
    if len(np.unique(used_indices)) != input_depth:
        raise ValueError("Some volumes occur in more than one b-value group.")

    print("b-value groups:")
    for label, b_value, indices in groups:
        print(
            "  %s: representative b=%g, volumes=%d, indices=%s"
            % (label, b_value, len(indices), indices.tolist())
        )

    return groups


def prepare_mask_for_image(
    mask_data,
    mask_affine,
    target_spatial_shape,
    target_affine,
):
    mask_array = np.asarray(mask_data, dtype=np.float32)
    original_shape = tuple(mask_array.shape)
    mask_array = np.squeeze(mask_array)

    print("Noisy spatial shape : %s" % (tuple(target_spatial_shape),))
    print("Original mask shape : %s" % (original_shape,))
    print("Squeezed mask shape : %s" % (tuple(mask_array.shape),))

    if mask_array.ndim == 4:
        if tuple(mask_array.shape[:3]) != tuple(target_spatial_shape):
            raise ValueError(
                "The 4D mask spatial shape does not match the noisy image: "
                "mask=%s, image=%s."
                % (mask_array.shape[:3], tuple(target_spatial_shape))
            )

        mask_channels = mask_array > float(MASK_THRESHOLD)
        channel_counts = np.count_nonzero(mask_channels, axis=(0, 1, 2))
        print(
            "4D mask channel voxels: min=%d, median=%d, max=%d"
            % (
                int(channel_counts.min()),
                int(np.median(channel_counts)),
                int(channel_counts.max()),
            )
        )

        mode = str(MASK_4D_MODE).strip().lower()
        if mode == "first":
            print("[Info] Using mask[..., 0] as the 3D spatial mask.")
            mask_array = mask_channels[..., 0].astype(np.float32)
        elif mode == "majority":
            required = int(
                np.ceil(mask_channels.shape[-1] * MASK_MAJORITY_FRACTION)
            )
            print(
                "[Info] Majority mask: at least %d/%d channels."
                % (required, mask_channels.shape[-1])
            )
            mask_array = (
                np.count_nonzero(mask_channels, axis=-1) >= required
            ).astype(np.float32)
        elif mode == "union":
            print("[Info] Using the union of all mask channels.")
            mask_array = np.any(mask_channels, axis=-1).astype(np.float32)
        else:
            raise ValueError(
                "Unsupported MASK_4D_MODE=%r. Use first, majority, or union."
                % MASK_4D_MODE
            )

    mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 3:
        raise ValueError(
            "Mask must be 3D after preparation, but got %s."
            % (mask_array.shape,)
        )

    shape_matches = tuple(mask_array.shape) == tuple(target_spatial_shape)
    affine_matches = np.allclose(
        np.asarray(mask_affine, dtype=np.float64),
        np.asarray(target_affine, dtype=np.float64),
        rtol=0.0,
        atol=1e-4,
    )

    if not shape_matches or not affine_matches:
        if not AUTO_RESAMPLE_MASK:
            raise ValueError("Mask and noisy image are on different grids.")

        print("[Info] Resampling mask with nearest-neighbor interpolation.")
        source_mask = nib.Nifti1Image(
            mask_array,
            np.asarray(mask_affine, dtype=np.float64),
        )
        resampled = resample_from_to(
            source_mask,
            (
                tuple(int(v) for v in target_spatial_shape),
                np.asarray(target_affine, dtype=np.float64),
            ),
            order=0,
        )
        mask_array = np.asarray(
            resampled.get_fdata(dtype=np.float32), dtype=np.float32
        )

    mask_bool = mask_array > float(MASK_THRESHOLD)
    if tuple(mask_bool.shape) != tuple(target_spatial_shape):
        raise ValueError(
            "Prepared mask shape %s does not match image shape %s."
            % (mask_bool.shape, tuple(target_spatial_shape))
        )
    if not np.any(mask_bool):
        raise ValueError("Prepared mask is empty.")

    print("Mask voxel count    : %d" % np.count_nonzero(mask_bool))
    return mask_bool


def build_repeated_noisy_mean_reference(noisy, bvalue_groups):
    reference = np.empty_like(noisy, dtype=np.float32)

    for label, _, indices in bvalue_groups:
        if len(indices) == 0:
            raise ValueError("Empty b-value group: %s" % label)
        group_mean = np.mean(noisy[indices], axis=0, dtype=np.float32)
        reference[indices] = group_mean[None, ...]

    return reference


def prepare_valid_ssim_mask(mask_bool, spatial_shape, win_size):
    if tuple(mask_bool.shape) != tuple(spatial_shape):
        raise ValueError(
            "Mask shape %s does not match image shape %s."
            % (mask_bool.shape, spatial_shape)
        )

    structure = np.ones((win_size, win_size, win_size), dtype=bool)
    valid_mask = binary_erosion(
        mask_bool,
        structure=structure,
        border_value=0,
    )

    if not np.any(valid_mask):
        pad = win_size // 2
        valid_mask = mask_bool.copy()
        valid_mask[:pad, :, :] = False
        valid_mask[-pad:, :, :] = False
        valid_mask[:, :pad, :] = False
        valid_mask[:, -pad:, :] = False
        valid_mask[:, :, :pad] = False
        valid_mask[:, :, -pad:] = False

    if not np.any(valid_mask):
        raise ValueError(
            "No valid brain voxels remain for win_size=%d." % win_size
        )

    return valid_mask


def noisy_vs_shell_mean_channel_ssim(
    repeated_reference,
    noisy,
    mask_bool,
    data_range,
):
    """
    Return one masked 3D SSIM score for every original noisy volume.

    Each noisy volume is compared with the noisy mean of its own b-value
    group. The same valid brain-window mask is used as in SSIM_new.py.
    """
    if repeated_reference.shape != noisy.shape:
        raise ValueError(
            "Input shapes differ: %s vs %s"
            % (repeated_reference.shape, noisy.shape)
        )
    if repeated_reference.ndim != 4:
        raise ValueError("Expected [C,X,Y,Z] data.")

    min_spatial_dim = min(noisy.shape[1:])
    win_size = min(7, min_spatial_dim)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError("Spatial dimensions are too small for SSIM.")

    valid_mask = prepare_valid_ssim_mask(
        mask_bool,
        noisy.shape[1:],
        win_size,
    )
    print("SSIM win_size     : %d" % win_size)
    print("Valid SSIM voxels : %d per channel" % np.count_nonzero(valid_mask))

    channel_scores = np.full(noisy.shape[0], np.nan, dtype=np.float64)

    for channel in range(noisy.shape[0]):
        _, ssim_map = structural_similarity(
            repeated_reference[channel],
            noisy[channel],
            data_range=float(data_range),
            win_size=win_size,
            full=True,
        )

        values = np.asarray(ssim_map[valid_mask], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size > 0:
            channel_scores[channel] = float(finite_values.mean())

        print(
            "  volume %3d/%3d: SSIM=%0.6f"
            % (channel + 1, noisy.shape[0], channel_scores[channel])
        )

    return channel_scores


def save_baseline_results(channel_scores, groups, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    group_scores = []
    group_min = []
    group_max = []
    group_std = []
    group_counts = []

    for label, _, indices in groups:
        values = channel_scores[indices]
        finite = values[np.isfinite(values)]
        labels.append(label)
        group_counts.append(len(indices))
        group_scores.append(float(np.mean(finite)) if finite.size else np.nan)
        group_min.append(float(np.min(finite)) if finite.size else np.nan)
        group_max.append(float(np.max(finite)) if finite.size else np.nan)
        group_std.append(float(np.std(finite)) if finite.size else np.nan)

    overall = float(np.nanmean(channel_scores))

    np.save(save_dir / "baseline_channel_ssim.npy", channel_scores)
    np.savez(
        save_dir / "baseline_noisy_vs_shell_mean.npz",
        overall_ssim=np.asarray(overall, dtype=np.float64),
        group_labels=np.asarray(labels),
        group_counts=np.asarray(group_counts, dtype=np.int64),
        group_ssim=np.asarray(group_scores, dtype=np.float64),
        group_min=np.asarray(group_min, dtype=np.float64),
        group_max=np.asarray(group_max, dtype=np.float64),
        group_std=np.asarray(group_std, dtype=np.float64),
        channel_ssim=channel_scores,
    )

    lines = []
    lines.append("SSIM-DA baseline for in-vivo data")
    lines.append(
        "SSIM-DA: structural similarity between direction-averaged images "
        "within each b-value shell."
    )
    lines.append("Overall SSIM-DA: %.9f" % overall)
    lines.append("")
    lines.append("Group\tVolumes\tMean SSIM-DA\tMin\tMax\tStd")
    for label, count, mean_v, min_v, max_v, std_v in zip(
        labels,
        group_counts,
        group_scores,
        group_min,
        group_max,
        group_std,
    ):
        lines.append(
            "%s\t%d\t%.9f\t%.9f\t%.9f\t%.9f"
            % (label, count, mean_v, min_v, max_v, std_v)
        )

    text_path = save_dir / "baseline_noisy_vs_shell_mean.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return overall, labels, group_counts, group_scores, group_min, group_max, group_std


def main():
    if not NOISY_PATH.exists():
        raise FileNotFoundError("Missing noisy data: %s" % NOISY_PATH)
    if not MASK_PATH.exists():
        raise FileNotFoundError("Missing brain mask: %s" % MASK_PATH)

    noisy_hwdc, affine = load_nifti(str(NOISY_PATH))
    mask_data, mask_affine = load_nifti(str(MASK_PATH))

    noisy_hwdc = np.asarray(noisy_hwdc, dtype=np.float32)
    if noisy_hwdc.ndim != 4:
        raise ValueError("Noisy data must be 4D: %s" % (noisy_hwdc.shape,))
    if noisy_hwdc.shape[-1] != INPUT_DEPTH:
        raise ValueError(
            "Data contain %d volumes, but INPUT_DEPTH=%d."
            % (noisy_hwdc.shape[-1], INPUT_DEPTH)
        )

    mask_bool = prepare_mask_for_image(
        mask_data=mask_data,
        mask_affine=mask_affine,
        target_spatial_shape=noisy_hwdc.shape[:3],
        target_affine=affine,
    )

    # [X,Y,Z,C] -> [C,X,Y,Z]
    noisy = noisy_hwdc.transpose(3, 0, 1, 2)
    groups = resolve_volume_groups(INPUT_DEPTH)
    repeated_reference = build_repeated_noisy_mean_reference(noisy, groups)

    masked_values = noisy[:, mask_bool]
    noisy_min = float(np.nanmin(masked_values))
    noisy_max = float(np.nanmax(masked_values))
    print("Masked noisy range : [%0.6f, %0.6f]" % (noisy_min, noisy_max))
    if noisy_min < -0.05 or noisy_max > float(DATA_RANGE) + 0.05:
        print(
            "[Warning] DATA_RANGE=%g may not match the actual intensity "
            "range." % DATA_RANGE
        )

    print("\nCalculating noisy-volume versus noisy-shell-mean SSIM...")
    channel_scores = noisy_vs_shell_mean_channel_ssim(
        repeated_reference,
        noisy,
        mask_bool,
        DATA_RANGE,
    )

    (
        overall,
        labels,
        counts,
        means,
        mins,
        maxs,
        stds,
    ) = save_baseline_results(channel_scores, groups, SAVE_DIR)

    print("\n================ SSIM-DA baseline validation ================")
    print("Overall noisy SSIM-DA: %0.9f" % overall)
    
    for label, count, mean_v, min_v, max_v, std_v in zip(
        labels, counts, means, mins, maxs, stds
    ):
        print(
            "%8s  n=%3d  SSIM-DA mean=%0.9f  min=%0.9f  "
            "max=%0.9f  std=%0.9f"
            % (label, count, mean_v, min_v, max_v, std_v)
        )
    
    print("\nSaved SSIM-DA scores : %s" % (SAVE_DIR / "baseline_channel_ssim.npy"))
    print("Saved SSIM-DA summary: %s" % (SAVE_DIR / "baseline_noisy_vs_shell_mean.npz"))
    print("Saved SSIM-DA report : %s" % (SAVE_DIR / "baseline_noisy_vs_shell_mean.txt"))


if __name__ == "__main__":
    main()
