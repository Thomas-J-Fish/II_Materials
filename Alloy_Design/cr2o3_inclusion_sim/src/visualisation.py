"""
visualisation.py
----------------
Plotting and output functions for the Cr2O3 inclusion phase-field simulation.

Produces:
  1. Side-by-side comparison of initial vs final inclusion shape
  2. Evolution panel showing snapshots at regular intervals
  3. Shape metrics over time (area, perimeter, circularity)
  4. Optional overlay on a micrograph image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os


# ---------------------------------------------------------------------------
# Colourmap — a clean dark-oxide-on-light-melt palette
# ---------------------------------------------------------------------------

_OXIDE_CMAP = LinearSegmentedColormap.from_list(
    "oxide",
    ["#f5f0e8", "#8b6914", "#3d2b00"],   # cream → amber → dark brown
)


# ---------------------------------------------------------------------------
# 1. Initial vs final comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    phi_initial: np.ndarray,
    phi_final:   np.ndarray,
    dx_um:       float,
    output_path: str = None,
    circularity_initial: float = None,
    circularity_final:   float = None,
    title_suffix: str = "",
):
    """
    Side-by-side plot of the initial (spalled) and final (relaxed) inclusion shape.

    Parameters
    ----------
    phi_initial, phi_final : (Ny, Nx) arrays
    dx_um         : grid spacing in micrometres (for scale bar)
    output_path   : if given, save figure to this path
    circularity_*  : optional scalar metrics to annotate
    title_suffix  : extra text appended to the figure title
    """
    Ny, Nx = phi_initial.shape
    extent = [0, Nx * dx_um, 0, Ny * dx_um]   # µm

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.patch.set_facecolor("#1a1a1a")

    for ax, phi, label, circ in zip(
        axes,
        [phi_initial, phi_final],
        ["Initial shape\n(spalled oxide fragment)", "Relaxed shape\n(after interfacial energy minimisation)"],
        [circularity_initial, circularity_final],
    ):
        im = ax.imshow(
            phi, origin="lower", cmap=_OXIDE_CMAP,
            vmin=0, vmax=1, extent=extent, interpolation="bilinear"
        )
        ax.set_xlabel("µm", color="#cccccc", fontsize=10)
        ax.set_ylabel("µm", color="#cccccc", fontsize=10)
        ax.tick_params(colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.set_facecolor("#1a1a1a")
        ax.set_title(label, color="#dddddd", fontsize=11, pad=8)

        if circ is not None:
            ax.text(
                0.03, 0.04, f"Circularity: {circ:.3f}",
                transform=ax.transAxes,
                color="#ffcc66", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#333333", ec="none", alpha=0.8),
            )

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("φ  (0 = melt,  1 = oxide)", color="#cccccc", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

    fig.suptitle(
        f"Cr₂O₃ inclusion shape relaxation — Allen-Cahn phase-field{title_suffix}",
        color="#eeeeee", fontsize=13, y=1.01
    )
    plt.tight_layout()

    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Evolution panel
# ---------------------------------------------------------------------------

def plot_evolution(
    snapshots: list,
    dx_um:     float,
    n_cols:    int   = 4,
    output_path: str = None,
):
    """
    Grid of snapshots showing how the inclusion shape evolves over time.

    Parameters
    ----------
    snapshots   : list of (step_index, phi) tuples from run_simulation()
    dx_um       : grid spacing in micrometres
    n_cols      : number of columns in the panel
    output_path : if given, save to this path
    """
    n = len(snapshots)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows))
    fig.patch.set_facecolor("#1a1a1a")
    axes = np.array(axes).flatten()

    Ny, Nx = snapshots[0][1].shape
    extent = [0, Nx * dx_um, 0, Ny * dx_um]

    for idx, (step, phi) in enumerate(snapshots):
        ax = axes[idx]
        ax.imshow(
            phi, origin="lower", cmap=_OXIDE_CMAP,
            vmin=0, vmax=1, extent=extent, interpolation="bilinear"
        )
        ax.set_title(f"step {step}", color="#cccccc", fontsize=9)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Inclusion shape evolution", color="#eeeeee", fontsize=12, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Shape metrics over time
# ---------------------------------------------------------------------------

def plot_metrics(
    metrics:     dict,
    output_path: str = None,
):
    """
    Plot area, perimeter, and circularity as functions of simulation step.

    Parameters
    ----------
    metrics     : dict returned by run_simulation()
    output_path : if given, save to this path
    """
    steps        = metrics["steps"]
    area         = metrics["area"]
    perimeter    = metrics["perimeter"]
    circularity  = metrics["circularity"]

    # Normalise area and perimeter to their initial values
    area_norm  = np.array(area)       / area[0]
    peri_norm  = np.array(perimeter)  / perimeter[0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.patch.set_facecolor("#1a1a1a")

    _metric_ax(axes[0], steps, area_norm,     "Area (normalised)",       "#4fc3f7")
    _metric_ax(axes[1], steps, peri_norm,     "Perimeter (normalised)",  "#f48fb1")
    _metric_ax(axes[2], steps, circularity,   "Circularity",             "#a5d6a7")

    axes[2].axhline(1.0, color="#ffffff", lw=0.6, ls="--", alpha=0.4, label="circle")
    axes[2].legend(fontsize=8, facecolor="#333333", labelcolor="#cccccc")

    fig.suptitle("Shape metrics vs simulation step", color="#eeeeee", fontsize=12, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, output_path)


def _metric_ax(ax, x, y, ylabel, color):
    ax.plot(x, y, color=color, lw=2)
    ax.set_xlabel("Step", color="#aaaaaa", fontsize=9)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa")
    ax.set_facecolor("#242424")
    ax.spines["bottom"].set_edgecolor("#555555")
    ax.spines["left"].set_edgecolor("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 4. Overlay on micrograph
# ---------------------------------------------------------------------------

def plot_micrograph_overlay(
    phi_final:      np.ndarray,
    micrograph_path: str,
    output_path:    str = None,
):
    """
    Overlay the simulated final phi contour on top of the SEM micrograph for
    direct visual comparison.

    The micrograph and simulation grid do not need to match in size — the
    contour is plotted in normalised axes coordinates.

    Parameters
    ----------
    phi_final        : (Ny, Nx) final phase-field array
    micrograph_path  : path to the SEM image
    output_path      : if given, save to this path
    """
    from skimage.io import imread
    from skimage.color import rgb2gray

    img  = imread(micrograph_path)
    gray = rgb2gray(img) if img.ndim == 3 else img

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#1a1a1a")

    # Left: micrograph
    axes[0].imshow(gray, cmap="gray", origin="upper")
    axes[0].set_title("SEM micrograph", color="#dddddd", fontsize=11)
    axes[0].axis("off")

    # Right: simulated final shape with contour
    axes[1].imshow(phi_final, cmap=_OXIDE_CMAP, origin="lower", vmin=0, vmax=1)
    axes[1].contour(phi_final, levels=[0.5], colors=["#ff4444"], linewidths=[1.5])
    axes[1].set_title("Simulated relaxed shape (φ = 0.5 contour in red)", color="#dddddd", fontsize=11)
    axes[1].axis("off")
    axes[1].set_facecolor("#1a1a1a")

    fig.suptitle("Simulated vs observed inclusion morphology", color="#eeeeee", fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_or_show(fig, output_path):
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {output_path}")
        plt.close(fig)
    else:
        plt.show()