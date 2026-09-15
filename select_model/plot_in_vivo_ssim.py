from __future__ import print_function

"""Plot in-vivo SSIM curves and identify candidate checkpoints.

User configuration:
    Key parameters to edit: metric/checkpoint paths, save directory, checkpoint interval, iteration range, b-value metadata, mask options, plot limits, and selection window.
    Relative paths assume execution from the repository root.
"""


from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import median_filter


# =====================================================================
# 0. Plot style
# =====================================================================
available_fonts = {font.name for font in font_manager.fontManager.ttflist}

if "Times New Roman" in available_fonts:
    MAIN_FONT = "Times New Roman"
elif "Liberation Serif" in available_fonts:
    MAIN_FONT = "Liberation Serif"
elif "Nimbus Roman" in available_fonts:
    MAIN_FONT = "Nimbus Roman"
else:
    MAIN_FONT = "DejaVu Serif"

print("Using font:", MAIN_FONT)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [MAIN_FONT]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 12


# =====================================================================
# 1. Configuration
# =====================================================================
METRIC_DIR = Path(
    "outputs/DIP_M1_W/"
    "three_different_resolution2/p9/"
    "offline_metrics_in_vivo_ssim_only2/"
)

SAVE_FIG_DIR = METRIC_DIR / "curve_figures_ssim_median_max_square"

START_ITERATION = None
END_ITERATION = None

# Display only; it does not affect selection.
PLOT_XLIM = (0, 50000)

# Set to None for automatic y-axis limits.
SSIM_YLIM = (0.50, 0.83)

# X-axis is displayed as 0, 1, 2, ... with x10^4 shown at the lower right.
ITERATION_TICK_STEP = 10000
ITERATION_AXIS_SCALE = 10000
SHOW_ITERATION_POWER_LABEL = True

FIG_DPI = 600
FIG_CREATE_DPI = 600
FIG_FORMAT = "png"

# Slightly square, matching the previous SNR-SSIM figure style.
FIG_SIZE = (6.6, 6.0)

LINE_WIDTH = 2.0
MAX_MAJOR_TICKS = 5

SSIM_RAW_COLOR = "tab:orange"
SSIM_FILTERED_COLOR = "tab:orange"
LIMITED_SSIM_COLOR = "blue"
GLOBAL_SSIM_COLOR = "gold"
POINT_EDGE_COLOR = "black"

TITLE_TEXT = "Noisy-SSIM"
Y_AXIS_LABEL = "SSIM-noisy"
LEGEND_LOCATION = "center right"

SSIM_KEY = "ssim_noisy_all"
SSIM_FILTERED_KEY = "ssim_noisy_all_median"

# Must be a positive odd integer.
# With checkpoints every 40 iterations, window 11 spans about 400 iterations.
SSIM_MEDIAN_WINDOW = 11
PLOT_MEDIAN_FILTERED_SSIM = True

# Blue point.
LIMITED_SSIM_SELECTION_START_ITERATION = 0
LIMITED_SSIM_SELECTION_END_ITERATION = 20000

# Yellow point.
GLOBAL_SSIM_SELECTION_START_ITERATION = 0
GLOBAL_SSIM_SELECTION_END_ITERATION = 50000

LIMITED_SSIM_SELECTION_SAVE_DIR = (
    SAVE_FIG_DIR / "ssim_max_before_end_selection"
)
GLOBAL_SSIM_SELECTION_SAVE_DIR = (
    SAVE_FIG_DIR / "ssim_global_max_selection"
)
# =====================================================================


def load_array(path: Path, dtype=np.float64):
    if not path.exists():
        raise FileNotFoundError(f"Missing required metric file: {path}")

    values = np.load(path, allow_pickle=False)
    return np.asarray(values, dtype=dtype).reshape(-1)


def median_filter_1d(values, window_size):
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            "SSIM_MEDIAN_WINDOW must be a positive odd integer."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Noisy-SSIM contains NaN or infinite values."
        )

    return median_filter(
        values,
        size=window_size,
        mode="nearest",
    )


def select_median_filtered_max(
    iterations,
    raw_values,
    start_iteration,
    end_iteration,
    metric_name,
):
    """
    Restrict to the requested interval, apply one median filter,
    and select the maximum of the filtered curve.

    Filtering is performed inside each interval independently.
    Therefore, the 0-20000 selection does not use data after 20000.
    """
    iterations = np.asarray(iterations, dtype=np.int64).reshape(-1)
    raw_values = np.asarray(raw_values, dtype=np.float64).reshape(-1)

    if len(iterations) != len(raw_values):
        raise ValueError(
            f"{metric_name}: iterations and values have different lengths."
        )

    if end_iteration is not None and end_iteration < start_iteration:
        raise ValueError(
            f"{metric_name}: end iteration must be >= start iteration."
        )

    valid = (
        (iterations >= start_iteration)
        & np.isfinite(raw_values)
    )

    if end_iteration is not None:
        valid &= iterations <= end_iteration

    valid_indices = np.flatnonzero(valid)

    if valid_indices.size == 0:
        raise ValueError(
            f"No valid {metric_name} values in interval "
            f"[{start_iteration}, {end_iteration}]."
        )

    interval_raw = raw_values[valid_indices]
    interval_filtered = median_filter_1d(
        interval_raw,
        SSIM_MEDIAN_WINDOW,
    )

    selected_local_index = int(np.argmax(interval_filtered))
    selected_index = int(valid_indices[selected_local_index])

    return {
        "selected_index": selected_index,
        "selected_iteration": int(iterations[selected_index]),
        "selected_raw_ssim": float(raw_values[selected_index]),
        "selected_filtered_ssim": float(
            interval_filtered[selected_local_index]
        ),
        "selection_start_iteration": int(start_iteration),
        "selection_end_iteration": (
            None if end_iteration is None else int(end_iteration)
        ),
    }


def restrict_iteration_range(iterations, arrays):
    keep = np.ones(len(iterations), dtype=bool)

    if START_ITERATION is not None:
        keep &= iterations >= START_ITERATION

    if END_ITERATION is not None:
        keep &= iterations <= END_ITERATION

    if not np.any(keep):
        raise ValueError(
            "No data remain after applying the iteration range."
        )

    return (
        iterations[keep],
        {
            name: np.asarray(values)[keep]
            for name, values in arrays.items()
        },
    )


def format_interval(start_iteration, end_iteration):
    end_text = "end" if end_iteration is None else str(end_iteration)
    return f"{start_iteration}-{end_text}"


def build_point(selection, label, color, size, zorder):
    return {
        "label": label,
        "iteration": selection["selected_iteration"],
        "raw_ssim": selection["selected_raw_ssim"],
        "filtered_ssim": selection["selected_filtered_ssim"],
        "start_iteration": selection["selection_start_iteration"],
        "end_iteration": selection["selection_end_iteration"],
        "color": color,
        "size": size,
        "zorder": zorder,
    }


def mark_point(ax, point):
    ax.scatter(
        [point["iteration"]],
        [point["filtered_ssim"]],
        s=point["size"],
        marker="o",
        facecolor=point["color"],
        edgecolor=POINT_EDGE_COLOR,
        linewidth=0.8,
        zorder=point["zorder"],
        label=point["label"],
    )


def save_selection(save_dir, point):
    save_dir.mkdir(parents=True, exist_ok=True)

    selected = np.asarray(
        [point["iteration"]],
        dtype=np.int64,
    )

    np.save(save_dir / "selected_iteration.npy", selected)
    np.save(save_dir / "selected_epoch.npy", selected)

    lines = [
        f"Selection name = {point['label']}",
        f"Selection interval = "
        f"[{point['start_iteration']}, {point['end_iteration']}]",
        f"SSIM median window = {SSIM_MEDIAN_WINDOW}",
        "",
        f"Selected Iteration = {point['iteration']}",
        f"Original Noisy-SSIM = {point['raw_ssim']:.12g}",
        f"Median-filtered Noisy-SSIM = "
        f"{point['filtered_ssim']:.12g}",
        "",
        "Selection rule:",
        "Apply one median filter to Noisy-SSIM inside the configured "
        "interval and select the maximum of the filtered curve.",
    ]

    summary_path = save_dir / "selection_summary.txt"
    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"[Saved] {summary_path}")


def apply_iteration_axis(ax):
    """Format iterations as 0, 1, 2, ... with a x10^4 multiplier."""
    if PLOT_XLIM is not None:
        xmin, xmax = PLOT_XLIM
        ax.set_xlim(xmin, xmax)
    else:
        xmin, xmax = ax.get_xlim()

    first_tick = int(
        np.ceil(xmin / ITERATION_TICK_STEP)
        * ITERATION_TICK_STEP
    )
    last_tick = int(
        np.floor(xmax / ITERATION_TICK_STEP)
        * ITERATION_TICK_STEP
    )

    ticks = np.arange(
        first_tick,
        last_tick + ITERATION_TICK_STEP,
        ITERATION_TICK_STEP,
        dtype=np.int64,
    )

    if ticks.size > 0:
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [
                f"{tick / ITERATION_AXIS_SCALE:g}"
                for tick in ticks
            ]
        )
    else:
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=MAX_MAJOR_TICKS)
        )

    if SHOW_ITERATION_POWER_LABEL:
        ax.text(
            1.0,
            -0.115,
            r"$\times10^{4}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=16,
        )


def finish_axis(ax, title, ylabel, filename):
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        ylabel,
        color=SSIM_FILTERED_COLOR,
    )
    ax.grid(False)

    apply_iteration_axis(ax)

    if SSIM_YLIM is not None:
        ax.set_ylim(
            SSIM_YLIM[0],
            SSIM_YLIM[1],
        )

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=MAX_MAJOR_TICKS)
    )

    ax.tick_params(
        axis="x",
        direction="out",
        length=4,
        width=1,
        colors="black",
    )
    ax.tick_params(
        axis="y",
        direction="out",
        length=4,
        width=1,
        colors=SSIM_FILTERED_COLOR,
    )

    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_color("black")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            frameon=False,
            fontsize=10,
            loc=LEGEND_LOCATION,
        )


def save_figure(fig, filename):
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    save_path = SAVE_FIG_DIR / f"{filename}.{FIG_FORMAT}"

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(
        save_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[Saved] {save_path}")


def main():
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    iterations_full = load_array(
        METRIC_DIR / "epochs.npy",
        dtype=np.int64,
    )
    raw_ssim_full = load_array(
        METRIC_DIR / "ssim_noisy_all.npy",
    )

    if len(iterations_full) == 0:
        raise ValueError("epochs.npy is empty.")

    if len(raw_ssim_full) != len(iterations_full):
        raise ValueError(
            "ssim_noisy_all.npy and epochs.npy have different lengths."
        )

    if np.any(np.diff(iterations_full) <= 0):
        raise ValueError(
            "epochs.npy must be strictly increasing."
        )

    filtered_ssim_full = median_filter_1d(
        raw_ssim_full,
        SSIM_MEDIAN_WINDOW,
    )

    np.save(
        SAVE_FIG_DIR / f"{SSIM_FILTERED_KEY}.npy",
        filtered_ssim_full,
    )

    limited_selection = select_median_filtered_max(
        iterations_full,
        raw_ssim_full,
        LIMITED_SSIM_SELECTION_START_ITERATION,
        LIMITED_SSIM_SELECTION_END_ITERATION,
        "limited-interval Noisy-SSIM",
    )

    global_selection = select_median_filtered_max(
        iterations_full,
        raw_ssim_full,
        GLOBAL_SSIM_SELECTION_START_ITERATION,
        GLOBAL_SSIM_SELECTION_END_ITERATION,
        "global Noisy-SSIM",
    )

    limited_interval_text = format_interval(
        LIMITED_SSIM_SELECTION_START_ITERATION,
        LIMITED_SSIM_SELECTION_END_ITERATION,
    )
    global_interval_text = format_interval(
        GLOBAL_SSIM_SELECTION_START_ITERATION,
        GLOBAL_SSIM_SELECTION_END_ITERATION,
    )

    limited_point = build_point(
        limited_selection,
        label=f"SSIM selection ({limited_interval_text})",
        color=LIMITED_SSIM_COLOR,
        size=85,
        zorder=9,
    )

    global_point = build_point(
        global_selection,
        label=f"SSIM selection ({global_interval_text})",
        color=GLOBAL_SSIM_COLOR,
        size=70,
        zorder=10,
    )

    save_selection(
        LIMITED_SSIM_SELECTION_SAVE_DIR,
        limited_point,
    )
    save_selection(
        GLOBAL_SSIM_SELECTION_SAVE_DIR,
        global_point,
    )

    iterations, arrays = restrict_iteration_range(
        iterations_full,
        {
            SSIM_KEY: raw_ssim_full,
            SSIM_FILTERED_KEY: filtered_ssim_full,
        },
    )

    fig, ax = plt.subplots(
        figsize=FIG_SIZE,
        dpi=FIG_CREATE_DPI,
    )

    ax.plot(
        iterations,
        arrays[SSIM_KEY],
        linewidth=1.0,
        color=SSIM_RAW_COLOR,
        alpha=0.45,
        label="Original Noisy-SSIM",
    )

    if PLOT_MEDIAN_FILTERED_SSIM:
        ax.plot(
            iterations,
            arrays[SSIM_FILTERED_KEY],
            linewidth=LINE_WIDTH,
            color=SSIM_FILTERED_COLOR,
            label=(
                "Median-filtered Noisy-SSIM "
            ),
        )

    mark_point(ax, limited_point)
    mark_point(ax, global_point)

    finish_axis(
        ax,
        TITLE_TEXT,
        Y_AXIS_LABEL,
        filename="01_Noisy_SSIM",
    )
    save_figure(fig, "01_Noisy_SSIM")

    print(
        "\n================ In-vivo SSIM selection ================"
    )

    for point in (limited_point, global_point):
        print(f"\n{point['label']}")
        print(f"  Iteration: {point['iteration']}")
        print(
            f"  Original Noisy-SSIM: "
            f"{point['raw_ssim']:.10g}"
        )
        print(
            f"  Median-filtered Noisy-SSIM: "
            f"{point['filtered_ssim']:.10g}"
        )

    if (
        limited_point["iteration"]
        == global_point["iteration"]
    ):
        print(
            "\n[Notice] The blue and yellow points occur at the same "
            "iteration. Different colors and sizes are used."
        )

    print("\nFigure saved to:")
    print(
        SAVE_FIG_DIR / f"01_Noisy_SSIM.{FIG_FORMAT}"
    )


if __name__ == "__main__":
    main()
