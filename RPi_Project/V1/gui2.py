# gui.py
# Full file replacement: adds a live Temperature vs Time plot fed by AD8495+MCP3008 over SPI.
# Requires:
#   - hardware/thermocouple.py (ThermocoupleAD8495, ThermocoupleConfig)
#   - hardware/__init__.py exporting ThermocoupleAD8495, ThermocoupleConfig
#
# Notes:
#   - Set VREF below to match your MCP3008 VREF wiring (3.3 or 5.0).
#   - This temperature stream runs continuously and plots live, independent of the loading cycle.

from __future__ import annotations

import os
os.environ["MPLBACKEND"] = "TkAgg"

import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from hardware import MotorController, Sensors, ThermocoupleAD8495, ThermocoupleConfig
from acquisition import run_loading_cycle


DataPoint = Tuple[float, float, float]      # (t, disp, force)
TempPoint = Tuple[float, float]             # (t, temp_C)


class SpringLoaderApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Automatic Spring Loader")
        self.geometry("1300x780")

        # ---- Hardware (simulation by default) ----
        self.motor = MotorController(simulation=True)
        self.sensors = Sensors(self.motor, simulation=True)

        # ---- Thermocouple hardware (AD8495 + MCP3008 over SPI) ----
        # IMPORTANT: set vref to match MCP3008 VREF pin wiring.
        # If MCP3008 VREF/VDD are on 3.3V -> vref=3.3
        # If MCP3008 VREF/VDD are on 5V   -> vref=5.0
        self.tc_cfg = ThermocoupleConfig(
            spi_bus=0,
            spi_dev=0,          # CE0
            channel=0,          # MCP3008 CH0
            vref=3.3,
            avg_samples=10,
            max_speed_hz=1350000
        )
        self.tc = ThermocoupleAD8495(self.tc_cfg)

        # ---- Experiment control ----
        self.stop_event = threading.Event()
        self.experiment_running = False
        self.data_queue: "queue.Queue[DataPoint]" = queue.Queue()

        # ---- Temperature streaming control ----
        self.temp_queue: "queue.Queue[TempPoint]" = queue.Queue()
        self.temp_running = True
        self.temp_t0 = time.monotonic()

        # ---- Data storage (current run only) ----
        self.time_data: List[float] = []
        self.disp_data: List[float] = []
        self.force_data: List[float] = []

        # ---- Temperature storage (continuous) ----
        self.temp_time: List[float] = []
        self.temp_data: List[float] = []

        # ---- Zero offsets for plotting ----
        self.t0: Optional[float] = None
        self.disp0: float = 0.0
        self.force0: float = 0.0

        # ---- Stress/strain internal SI storage ----
        self.area_m2_var = tk.DoubleVar(value=1.0e-6)   # m^2
        self.gauge_m_var = tk.DoubleVar(value=1.0e-2)   # m

        # ---- User-friendly unit inputs ----
        self.area_mm2_var = tk.DoubleVar(value=1.0)     # mm^2
        self.gauge_mm_var = tk.DoubleVar(value=10.0)    # mm

        # ---- Run parameters ----
        self.step_var = tk.DoubleVar(value=1.0)         # "step" size (deg in sim/position-mode)
        self.n_steps_var = tk.IntVar(value=50)
        self.dwell_var = tk.DoubleVar(value=0.05)
        self.rate_var = tk.IntVar(value=500)            # rotation rate proxy (future mapping)

        # ---- Overlay storage (stress–strain only) ----
        self.previous_ss_lines = []  # list of Line2D objects

        # ---- Layout ----
        self._build_controls()
        self._build_plots()
        self.apply_units_to_si()  # sync SI label at startup

        # start background temperature reader thread
        threading.Thread(target=self._temperature_worker, daemon=True).start()

        self.after(50, self._update_plot_from_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        control = ttk.Frame(self)
        control.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(control, text="Controls", font=("TkDefaultFont", 12, "bold")).pack(pady=(0, 10))

        # Start/Stop/Clear/New run
        self.start_button = ttk.Button(control, text="Start", command=self.start_experiment)
        self.start_button.pack(fill=tk.X, pady=4)

        self.stop_button = ttk.Button(control, text="Stop", command=self.stop_experiment)
        self.stop_button.pack(fill=tk.X, pady=4)

        self.new_run_button = ttk.Button(control, text="New run (overlay)", command=self.new_run_overlay)
        self.new_run_button.pack(fill=tk.X, pady=4)

        self.clear_button = ttk.Button(control, text="Clear all", command=self.clear_all)
        self.clear_button.pack(fill=tk.X, pady=4)

        # Zero
        self.zero_button = ttk.Button(control, text="Set zero", command=self.set_zero)
        self.zero_button.pack(fill=tk.X, pady=(10, 4))

        # Saving / summary
        self.save_csv_button = ttk.Button(control, text="Save CSV", command=self.save_csv)
        self.save_csv_button.pack(fill=tk.X, pady=4)

        self.save_png_button = ttk.Button(control, text="Save plots (PNG)", command=self.save_png)
        self.save_png_button.pack(fill=tk.X, pady=4)

        self.summary_button = ttk.Button(control, text="Export summary", command=self.export_summary)
        self.summary_button.pack(fill=tk.X, pady=4)

        ttk.Separator(control).pack(fill=tk.X, pady=12)

        # Run parameters
        ttk.Label(control, text="Run parameters", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        ttk.Label(control, text="Step [deg]").pack(anchor="w", pady=(8, 0))
        ttk.Entry(control, textvariable=self.step_var).pack(fill=tk.X)

        ttk.Label(control, text="Steps up").pack(anchor="w", pady=(8, 0))
        ttk.Entry(control, textvariable=self.n_steps_var).pack(fill=tk.X)

        ttk.Label(control, text="Dwell [s]").pack(anchor="w", pady=(8, 0))
        ttk.Entry(control, textvariable=self.dwell_var).pack(fill=tk.X)

        ttk.Label(control, text="Rotation rate (proxy)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(control, textvariable=self.rate_var).pack(fill=tk.X)

        ttk.Separator(control).pack(fill=tk.X, pady=12)

        # Stress–strain params (mm units + convert)
        ttk.Label(control, text="Stress–strain params", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        ttk.Label(control, text="Area A [mm²]").pack(anchor="w", pady=(8, 0))
        ttk.Entry(control, textvariable=self.area_mm2_var).pack(fill=tk.X)

        ttk.Label(control, text="Gauge length L0 [mm]").pack(anchor="w", pady=(8, 0))
        ttk.Entry(control, textvariable=self.gauge_mm_var).pack(fill=tk.X)

        ttk.Button(control, text="Apply A/L0 (→ SI)", command=self.apply_units_to_si).pack(fill=tk.X, pady=6)

        self.si_label_var = tk.StringVar(value="")
        ttk.Label(control, textvariable=self.si_label_var, foreground="gray").pack(fill=tk.X)

        ttk.Separator(control).pack(fill=tk.X, pady=12)

        # Status
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(control, textvariable=self.status_var, foreground="blue").pack(fill=tk.X)

        ttk.Label(
            control,
            text="Note: Stress–strain requires A and L0.\nOverlays apply only to stress–strain.\n"
                 "Temperature plot runs continuously.",
            foreground="gray"
        ).pack(fill=tk.X, pady=(8, 0))

    def _build_plots(self) -> None:
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)

        # Grid: 3x2, right col spans all rows
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.0, 1.8], height_ratios=[1.0, 1.0, 1.0])
        self.ax_xt = self.fig.add_subplot(gs[0, 0])
        self.ax_Ft = self.fig.add_subplot(gs[1, 0])
        self.ax_Tt = self.fig.add_subplot(gs[2, 0])
        self.ax_ss = self.fig.add_subplot(gs[:, 1])

        (self.line_xt,) = self.ax_xt.plot([], [], "b-", linewidth=1.5)
        (self.line_Ft,) = self.ax_Ft.plot([], [], "r-", linewidth=1.5)
        (self.line_Tt,) = self.ax_Tt.plot([], [], "g-", linewidth=1.5)
        (self.line_ss,) = self.ax_ss.plot([], [], "k-", linewidth=1.5)

        self._format_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _format_axes(self) -> None:
        self.ax_xt.set_title("Displacement vs time")
        self.ax_xt.set_xlabel("t [s]")
        self.ax_xt.set_ylabel("x [m]")

        self.ax_Ft.set_title("Force / torque proxy vs time")
        self.ax_Ft.set_xlabel("t [s]")
        self.ax_Ft.set_ylabel("F [N]")

        self.ax_Tt.set_title("Temperature vs time")
        self.ax_Tt.set_xlabel("t [s]")
        self.ax_Tt.set_ylabel("T [°C]")

        self.ax_ss.set_title("Stress–strain")
        self.ax_ss.set_xlabel("strain ε [-]")
        self.ax_ss.set_ylabel("stress σ [Pa]")

    # ------------------------------------------------------------------
    # Helpers: units + derived curves
    # ------------------------------------------------------------------
    def apply_units_to_si(self) -> None:
        """Convert mm² → m² and mm → m; stores in internal SI vars."""
        try:
            A_mm2 = float(self.area_mm2_var.get())
            L_mm = float(self.gauge_mm_var.get())
            if A_mm2 <= 0 or L_mm <= 0:
                raise ValueError("A and L0 must be > 0")

            A_m2 = A_mm2 * 1e-6
            L_m = L_mm * 1e-3

            self.area_m2_var.set(A_m2)
            self.gauge_m_var.set(L_m)
            self.si_label_var.set(f"SI: A={A_m2:.3e} m², L0={L_m:.3e} m")
            self.status_var.set("Applied A/L0 units")
        except Exception as e:
            messagebox.showerror("Invalid A/L0", str(e))

    def _compute_stress_strain(self) -> Tuple[List[float], List[float]]:
        """ε = x/L0, σ = F/A. Returns empty lists if A/L0 invalid."""
        try:
            A = float(self.area_m2_var.get())
            L0 = float(self.gauge_m_var.get())
            if A <= 0 or L0 <= 0:
                return ([], [])
        except Exception:
            return ([], [])

        strain = [x / L0 for x in self.disp_data]
        stress = [F / A for F in self.force_data]
        return (strain, stress)

    def _compute_summary(self) -> Optional[dict]:
        """
        Summary:
          - max_stress_Pa
          - max_strain
          - stiffness_Pa (slope dσ/dε in linear region)
          - linear_region (eps_lo, eps_hi)
        """
        strain, stress = self._compute_stress_strain()
        if not strain or not stress or len(strain) < 5:
            return None

        max_stress = max(stress)
        max_strain = max(strain)
        if max_strain <= 0:
            return None

        # default linear region: 5%–20% of max strain
        eps_lo = 0.05 * max_strain
        eps_hi = 0.20 * max_strain

        xs = [e for e in strain if eps_lo <= e <= eps_hi]
        ys = [s for e, s in zip(strain, stress) if eps_lo <= e <= eps_hi]

        if len(xs) < 3:
            # fallback to first N points
            n = min(20, len(strain))
            xs = strain[:n]
            ys = stress[:n]
            eps_lo, eps_hi = xs[0], xs[-1]

        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        stiffness = (num / den) if den > 0 else float("nan")

        return {
            "max_stress_Pa": max_stress,
            "max_strain": max_strain,
            "stiffness_Pa": stiffness,
            "linear_region": (eps_lo, eps_hi),
        }

    # ------------------------------------------------------------------
    # Control actions
    # ------------------------------------------------------------------
    def set_zero(self) -> None:
        """
        Zero-references plotted data from the latest available point.
        Also calls motor.go_home() (safe in sim; can be redefined for real hardware).
        """
        if self.time_data:
            self.t0 = self.time_data[-1]
            self.disp0 = self.disp_data[-1]
            self.force0 = self.force_data[-1]
        else:
            self.t0 = None
            self.disp0 = 0.0
            self.force0 = 0.0

        try:
            self.motor.go_home()
        except Exception:
            pass

        self.status_var.set("Zero set")

    def new_run_overlay(self) -> None:
        """Keep previous stress–strain curve as faint grey, then clear current run."""
        if self.disp_data and self.force_data:
            strain, stress = self._compute_stress_strain()
            if strain and stress:
                line, = self.ax_ss.plot(strain, stress, color="0.7", linewidth=1.0)
                self.previous_ss_lines.append(line)

        self._clear_current_run_only()
        self.status_var.set("Ready for new run (overlay kept)")

    def clear_all(self) -> None:
        """Clear current run and remove overlays."""
        # remove overlay lines
        for ln in self.previous_ss_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self.previous_ss_lines.clear()

        self._clear_current_run_only()
        self.status_var.set("Cleared all")

    def _clear_current_run_only(self) -> None:
        self.time_data.clear()
        self.disp_data.clear()
        self.force_data.clear()
        self.data_queue = queue.Queue()

        self.t0 = None
        self.disp0 = 0.0
        self.force0 = 0.0

        self.line_xt.set_data([], [])
        self.line_Ft.set_data([], [])
        self.line_ss.set_data([], [])

        for ax in (self.ax_xt, self.ax_Ft, self.ax_ss):
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw_idle()

    def start_experiment(self) -> None:
        if self.experiment_running:
            return

        self.experiment_running = True
        self.stop_event.clear()
        self.status_var.set("Running...")

        worker = threading.Thread(target=self._acquisition_worker, daemon=True)
        worker.start()

    def stop_experiment(self) -> None:
        self.stop_event.set()
        self.experiment_running = False
        self.status_var.set("Stopping...")

    # ------------------------------------------------------------------
    # Acquisition thread
    # ------------------------------------------------------------------
    def _acquisition_worker(self) -> None:
        try:
            step_deg = float(self.step_var.get())
            n_steps_up = int(self.n_steps_var.get())
            dwell_s = float(self.dwell_var.get())

            # Placeholder: rotation rate proxy is not used yet by acquisition/motor.
            _rate = int(self.rate_var.get())
            _ = _rate  # keep lint quiet

            for t, disp, force in run_loading_cycle(
                motor=self.motor,
                sensors=self.sensors,
                stop_flag=self.stop_event,
                step_deg=step_deg,
                n_steps_up=n_steps_up,
                dwell_s=dwell_s,
            ):
                if self.stop_event.is_set():
                    break

                # Apply zero offsets
                t_rel = t if self.t0 is None else (t - self.t0)
                disp_rel = disp - self.disp0
                force_rel = force - self.force0

                self.data_queue.put((t_rel, disp_rel, force_rel))

        except Exception as e:
            self.data_queue.put((-1.0, float("nan"), float("nan")))
            self._error_message = str(e)
        finally:
            self.experiment_running = False
            self.status_var.set("Stopped" if self.stop_event.is_set() else "Idle")

    # ------------------------------------------------------------------
    # Temperature stream thread
    # ------------------------------------------------------------------
    def _temperature_worker(self) -> None:
        """Continuously read thermocouple temperature and push to queue."""
        while self.temp_running:
            try:
                T = float(self.tc.read_temp_c())
                t = time.monotonic() - self.temp_t0
                self.temp_queue.put((t, T))
            except Exception:
                # Avoid killing the GUI if a read fails briefly
                pass
            time.sleep(0.2)

    # ------------------------------------------------------------------
    # GUI thread plot updates
    # ------------------------------------------------------------------
    def _update_plot_from_queue(self) -> None:
        updated = False
        temp_updated = False

        # Drain experiment datapoints
        try:
            while True:
                t, disp, force = self.data_queue.get_nowait()

                if t < 0 and (disp != disp or force != force):  # NaN check
                    messagebox.showerror("Acquisition error", getattr(self, "_error_message", "Unknown error"))
                    break

                self.time_data.append(float(t))
                self.disp_data.append(float(disp))
                self.force_data.append(float(force))
                updated = True

        except queue.Empty:
            pass

        # Drain temperature datapoints
        try:
            while True:
                tt, T = self.temp_queue.get_nowait()
                self.temp_time.append(float(tt))
                self.temp_data.append(float(T))
                temp_updated = True
        except queue.Empty:
            pass

        if updated:
            # Time series
            self.line_xt.set_data(self.time_data, self.disp_data)
            self.line_Ft.set_data(self.time_data, self.force_data)

            # Stress–strain
            strain, stress = self._compute_stress_strain()
            self.line_ss.set_data(strain, stress)

        if temp_updated:
            self.line_Tt.set_data(self.temp_time, self.temp_data)

        if updated or temp_updated:
            for ax in (self.ax_xt, self.ax_Ft, self.ax_Tt, self.ax_ss):
                ax.relim()
                ax.autoscale_view()
            self.canvas.draw_idle()

        self.after(50, self._update_plot_from_queue)

    # ------------------------------------------------------------------
    # Saving / exporting
    # ------------------------------------------------------------------
    def save_csv(self) -> None:
        if not self.time_data and not self.temp_time:
            self.status_var.set("No data to save")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save data as CSV",
        )
        if not filename:
            return

        strain, stress = self._compute_stress_strain()

        try:
            with open(filename, "w", newline="") as f:
                w = csv.writer(f)
                # Save mechanical run data (if any)
                w.writerow(["t [s]", "displacement [m]", "force [N]", "strain [-]", "stress [Pa]"])
                for i in range(len(self.time_data)):
                    eps = strain[i] if i < len(strain) else ""
                    sig = stress[i] if i < len(stress) else ""
                    w.writerow([self.time_data[i], self.disp_data[i], self.force_data[i], eps, sig])

                # Blank line + temperature data section
                w.writerow([])
                w.writerow(["Temperature stream"])
                w.writerow(["t [s]", "T [°C]"])
                for i in range(len(self.temp_time)):
                    w.writerow([self.temp_time[i], self.temp_data[i]])

            self.status_var.set(f"Saved CSV: {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def save_png(self) -> None:
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save plots as PNG",
        )
        if not filename:
            return
        try:
            self.fig.savefig(filename, dpi=200)
            self.status_var.set(f"Saved PNG: {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def export_summary(self) -> None:
        summary = self._compute_summary()
        if summary is None:
            messagebox.showinfo(
                "Summary",
                "Not enough valid stress–strain data to compute summary.\n"
                "Check A/L0 and collect more points."
            )
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save summary",
        )
        if not filename:
            return

        eps_lo, eps_hi = summary["linear_region"]

        try:
            with open(filename, "w") as f:
                f.write("Automatic Spring Loader – Summary\n")
                f.write("---------------------------------\n")
                f.write(f"Max strain ε_max [-]: {summary['max_strain']:.6g}\n")
                f.write(f"Max stress σ_max [Pa]: {summary['max_stress_Pa']:.6g}\n")
                f.write(f"Stiffness (slope dσ/dε) [Pa]: {summary['stiffness_Pa']:.6g}\n")
                f.write(f"Linear region used [ε]: {eps_lo:.6g} to {eps_hi:.6g}\n")
                f.write("\nInputs\n")
                f.write(f"A [m²]: {float(self.area_m2_var.get()):.6g}\n")
                f.write(f"L0 [m]: {float(self.gauge_m_var.get()):.6g}\n")

            self.status_var.set(f"Saved summary: {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self.stop_event.set()
        self.temp_running = False

        try:
            self.tc.close()
        except Exception:
            pass

        try:
            self.sensors.close()
        except Exception:
            pass

        self.destroy()


if __name__ == "__main__":
    app = SpringLoaderApp()
    app.mainloop()