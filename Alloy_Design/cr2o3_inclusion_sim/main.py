import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from src.initialisation import irregular_polygon, ellipse
from src.phase_field import run_simulation, _circularity, _area
from src.cooling import steps_from_cooling, print_time_scales
from src.visualisation import plot_comparison, plot_evolution, plot_metrics

CONFIG = {
    "run_mode":   "sweep",
    "grid_size":  256,
    "dx_m":       1.34e-7,
    "geometry":    "irregular",
    "n_vertices":  12,
    "roughness":   0.38,
    "seed":        7,
    "interface_width": 3,
    "mean_radius_px": 28,
    "bc":                  "neumann",
    "anisotropy_strength": 0.0,
    "n_fold":              6,
    "delta_T":    820.0,
    "T_melt":     2100.0,
    "T_solidus":  1280.0,
    "mobility":   1e-13,
    "sigma":      1.5,
    "sweep_radii_um":      [2.0, 3.5, 5.0, 7.0, 10.0],
    "sweep_cooling_rates": [5.0, 20.0, 100.0, 400.0],
    "max_steps":  8000,
    "min_steps":  500,
    "n_snapshots": 40,
    "output_dir": "outputs",
    "micrograph_path": None,
}

CR_COLOURS = {5.0: "#4fc3f7", 20.0: "#f48fb1", 100.0: "#a5d6a7"}
RADIUS_COLOURS = ["#ffcc66", "#ce93d8", "#80cbc4", "#ef9a9a", "#bcaaa4"]
DARK_BG = "#1a1a1a"; PANEL_BG = "#242424"; SPINE_COL = "#555555"
TICK_COL = "#aaaaaa"; TITLE_COL = "#dddddd"; TEXT_COL = "#cccccc"

def _style_ax(ax):
    ax.set_facecolor(PANEL_BG); ax.tick_params(colors=TICK_COL)
    for side in ["bottom", "left"]: ax.spines[side].set_edgecolor(SPINE_COL)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def _savefig(fig, path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    full = os.path.join(out_dir, path)
    fig.savefig(full, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  Saved -> {full}")

def build_inclusion(cfg, radius_px, grid_size=None, seed=None):
    N = grid_size or cfg["grid_size"]; s = seed if seed is not None else cfg["seed"]
    return irregular_polygon(grid_shape=(N,N), mean_radius=radius_px,
        n_vertices=cfg["n_vertices"], roughness=cfg["roughness"],
        seed=s, interface_width=cfg["interface_width"])

def choose_grid(radius_px):
    needed = max(64, int(radius_px * 7))
    return min(512, int(2 ** np.ceil(np.log2(needed))))

def get_steps(cfg, cooling_rate):
    n, t_dim, t_phys = steps_from_cooling(
        dx_m=cfg["dx_m"], dt_dimless=0.1, cooling_rate=cooling_rate,
        delta_T=cfg["delta_T"], mobility=cfg["mobility"], sigma=cfg["sigma"],
        min_steps=cfg["min_steps"], max_steps=cfg["max_steps"])
    return n, t_dim, t_phys

def run_sweep(cfg):
    out = cfg["output_dir"]
    dx_m = cfg["dx_m"]; dx_um = dx_m * 1e6
    radii_um = cfg["sweep_radii_um"]; cooling_rates = cfg["sweep_cooling_rates"]

    print("=" * 60)
    print("  Sweep: circularity vs radius and cooling rate")
    print(f"  Radii:   {radii_um} µm")
    print(f"  CR:      {cooling_rates} °C/s")
    print(f"  delta_T: {cfg['delta_T']} °C  |  M: {cfg['mobility']:.0e}  |  σ: {cfg['sigma']} J/m²")
    print("=" * 60)

    sweep_data = {cr: {} for cr in cooling_rates}
    fixed_cr = cooling_rates[len(cooling_rates) // 2]

    for cr in cooling_rates:
        _, _, t_phys = get_steps(cfg, cr)   # used for reporting only
        n_steps    = cfg["max_steps"]        # loop exits via solidification break
        save_every = max(1, n_steps // cfg["n_snapshots"])
        print(f"\n--- CR = {cr} °C/s  |  t_solid = {t_phys:.1f}s  |  max_steps = {n_steps} ---")
        for r_um in radii_um:
            r_px = r_um / dx_um
            if r_px < 8:
                sweep_data[cr][r_um] = None; print(f"  R={r_um}µm: too small, skip"); continue
            grid = choose_grid(r_px)
            print(f"  R={r_um}µm ({r_px:.1f}px, grid={grid})...", end=" ", flush=True)
            phi_init = build_inclusion(cfg, r_px, grid_size=grid)
            phi_f, snaps, metrics = run_simulation(
                phi_init=phi_init, n_steps=n_steps, dx=dx_m,
                interface_width=cfg["interface_width"], bc=cfg["bc"],
                anisotropy_strength=cfg["anisotropy_strength"], n_fold=cfg["n_fold"],
                save_every=save_every, verbose=False,
                T_start_celsius=cfg["T_melt"], cooling_rate=cr,
                M0=cfg["mobility"], sigma=cfg["sigma"],
                T_solidus_celsius=cfg["T_solidus"])
            ci = _circularity(phi_init); cf = _circularity(phi_f)
            sweep_data[cr][r_um] = {
                "circ_init": ci, "circ_final": cf,
                "circ_series": metrics["circularity"], "step_series": metrics["steps"],
                "time_series": metrics["physical_time_s"],
                "t_phys": t_phys, "n_steps": n_steps,
                "phi_init": phi_init, "phi_final": phi_f, "grid": grid, "r_px": r_px}
            print(f"circ {ci:.3f} -> {cf:.3f}")

    plot_circularity_vs_radius(sweep_data, cfg, out)
    plot_circ_vs_step_by_cr(sweep_data, cfg, fixed_cr, out)
    plot_circ_vs_step_by_radius(sweep_data, cfg, out)
    plot_shape_comparisons(sweep_data, cfg, dx_um, out)
    print(f"\nAll outputs written to ./{out}/\n")

def plot_circularity_vs_radius(sweep_data, cfg, out):
    cooling_rates = cfg["sweep_cooling_rates"]; radii_um = cfg["sweep_radii_um"]
    fig, ax = plt.subplots(figsize=(7, 5)); fig.patch.set_facecolor(DARK_BG); _style_ax(ax)
    init_r, init_c = [], []
    for r_um in radii_um:
        d = sweep_data[cooling_rates[0]].get(r_um)
        if d: init_r.append(r_um); init_c.append(d["circ_init"])
    ax.plot(init_r, init_c, color="#777777", lw=1.5, ls="--", label="Initial shape", zorder=2)
    for cr in cooling_rates:
        col = CR_COLOURS.get(cr, "#ffffff"); rs, cfs = [], []
        for r_um in radii_um:
            d = sweep_data[cr].get(r_um)
            if d: rs.append(r_um); cfs.append(d["circ_final"])
        ax.plot(rs, cfs, color=col, lw=2, marker="o", ms=6, label=f"{cr} °C/s", zorder=3)
    ax.axhline(1.0, color="#ffffff", lw=0.6, ls=":", alpha=0.35)
    ax.set_xlabel("Inclusion radius  (µm)", color=TICK_COL, fontsize=10)
    ax.set_ylabel("Circularity  (4πA/P²,  circle = 1)", color=TICK_COL, fontsize=10)
    ax.set_title("Final circularity vs inclusion size", color=TITLE_COL, fontsize=11, pad=6)
    ax.legend(fontsize=8, facecolor="#333", labelcolor=TEXT_COL, framealpha=0.85)
    fig.text(0.5, -0.04, f"M = {cfg['mobility']:.0e} m³J⁻¹s⁻¹  |  σ = {cfg['sigma']} Jm⁻²  |  δT = {cfg['delta_T']:.0f} °C",
        ha="center", color="#777", fontsize=8)
    fig.suptitle("Cr₂O₃ inclusion shape relaxation — size and cooling rate dependence",
        color="#eee", fontsize=12, y=1.02)
    plt.tight_layout(); _savefig(fig, "circularity_vs_radius.png", out)

def plot_circ_vs_step_by_cr(sweep_data, cfg, fixed_cr, out):
    cooling_rates = cfg["sweep_cooling_rates"]; radii_um = cfg["sweep_radii_um"]
    mid_r = radii_um[len(radii_um) // 2]
    dx_m = cfg["dx_m"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); fig.patch.set_facecolor(DARK_BG)
    for ax in axes: _style_ax(ax)
    for cr in cooling_rates:
        d = sweep_data[cr].get(mid_r)
        if not d: continue
        col = CR_COLOURS.get(cr, "#ffffff")
        steps = np.array(d["step_series"]); circs = np.array(d["circ_series"])
        times = np.array(d["time_series"])
        axes[0].plot(steps, circs, color=col, lw=2, label=f"{cr} °C/s")
        axes[1].plot(times, circs, color=col, lw=2, label=f"{cr} °C/s")
    for ax, xl in zip(axes, ["Simulation step", "Physical time  (s)"]):
        ax.axhline(1.0, color="#ffffff", lw=0.6, ls=":", alpha=0.35)
        ax.set_ylabel("Circularity", color=TICK_COL, fontsize=10)
        ax.set_xlabel(xl, color=TICK_COL, fontsize=10)
        ax.legend(fontsize=8, facecolor="#333", labelcolor=TEXT_COL, framealpha=0.85)
    axes[0].set_title(f"Circularity evolution — R = {mid_r} µm  (by cooling rate)", color=TITLE_COL, fontsize=11, pad=6)
    axes[1].set_title(f"Circularity vs physical time — R = {mid_r} µm", color=TITLE_COL, fontsize=11, pad=6)
    fig.suptitle("Shape relaxation dynamics — cooling rate dependence", color="#eee", fontsize=12, y=1.02)
    plt.tight_layout(); _savefig(fig, "circ_vs_step_by_cooling_rate.png", out)

def plot_circ_vs_step_by_radius(sweep_data, cfg, out):
    cooling_rates = cfg["sweep_cooling_rates"]; radii_um = cfg["sweep_radii_um"]
    fixed_cr = cooling_rates[1]
    dx_m = cfg["dx_m"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); fig.patch.set_facecolor(DARK_BG)
    for ax in axes: _style_ax(ax)
    for r_um, col in zip(radii_um, RADIUS_COLOURS):
        d = sweep_data[fixed_cr].get(r_um)
        if not d: continue
        steps = np.array(d["step_series"]); circs = np.array(d["circ_series"])
        times = np.array(d["time_series"])
        axes[0].plot(steps, circs, color=col, lw=2, label=f"{r_um} µm")
        axes[1].plot(times, circs, color=col, lw=2, label=f"{r_um} µm")
    for ax, xl in zip(axes, ["Simulation step", "Physical time  (s)"]):
        ax.axhline(1.0, color="#ffffff", lw=0.6, ls=":", alpha=0.35)
        ax.set_ylabel("Circularity", color=TICK_COL, fontsize=10)
        ax.set_xlabel(xl, color=TICK_COL, fontsize=10)
        ax.legend(fontsize=8, facecolor="#333", labelcolor=TEXT_COL, framealpha=0.85, title="Radius", title_fontsize=8)
    axes[0].set_title(f"Circularity evolution — CR = {fixed_cr} °C/s  (by radius)", color=TITLE_COL, fontsize=11, pad=6)
    axes[1].set_title(f"Circularity vs physical time — CR = {fixed_cr} °C/s", color=TITLE_COL, fontsize=11, pad=6)
    fig.suptitle("Shape relaxation dynamics — inclusion size dependence", color="#eee", fontsize=12, y=1.02)
    plt.tight_layout(); _savefig(fig, "circ_vs_step_by_radius.png", out)

def plot_shape_comparisons(sweep_data, cfg, dx_um, out):
    from matplotlib.colors import LinearSegmentedColormap
    oxide_cmap = LinearSegmentedColormap.from_list("oxide", ["#f5f0e8","#8b6914","#3d2b00"])
    radii_um = cfg["sweep_radii_um"]; cooling_rates = cfg["sweep_cooling_rates"]
    for cr in cooling_rates:
        valid = [(r, sweep_data[cr][r]) for r in radii_um if sweep_data[cr].get(r)]
        if not valid: continue
        n = len(valid)
        fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n)); fig.patch.set_facecolor(DARK_BG)
        if n == 1: axes = [axes]
        for row, (r_um, d) in enumerate(valid):
            for col_idx, (phi, label) in enumerate([
                    (d["phi_init"], "Initial (spalled)"),
                    (d["phi_final"], f"Relaxed  (circ={d['circ_final']:.3f})")]):
                ax = axes[row][col_idx]; grid = d["grid"]; extent = [0, grid*dx_um, 0, grid*dx_um]
                ax.imshow(phi, origin="lower", cmap=oxide_cmap, vmin=0, vmax=1,
                          extent=extent, interpolation="bilinear")
                ax.set_facecolor(DARK_BG); ax.tick_params(colors=TICK_COL, labelsize=7)
                for sp in ax.spines.values(): sp.set_edgecolor(SPINE_COL)
                ax.set_title(f"R = {r_um} µm — {label}", color=TITLE_COL, fontsize=9, pad=4)
                ax.set_xlabel("µm", color=TICK_COL, fontsize=8); ax.set_ylabel("µm", color=TICK_COL, fontsize=8)
        fig.suptitle(f"Inclusion shape relaxation — cooling rate {cr} °C/s\n"
            f"δT = {cfg['delta_T']:.0f} °C  |  t_solid = {cfg['delta_T']/cr:.1f} s  |  M = {cfg['mobility']:.0e} m³J⁻¹s⁻¹",
            color="#eee", fontsize=11, y=1.01)
        plt.tight_layout(); _savefig(fig, f"shape_comparison_CR{int(cr):03d}.png", out)

def main():
    cfg = CONFIG; os.makedirs(cfg["output_dir"], exist_ok=True)
    if cfg["run_mode"] == "sweep": run_sweep(cfg)

if __name__ == "__main__": main()