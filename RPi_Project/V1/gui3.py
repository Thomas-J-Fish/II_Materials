# gui2.py
# Modernised Tkinter UI using ttkbootstrap + styled matplotlib + live thermocouple plot.
# - Fullscreen instrument mode by default (press Esc to exit fullscreen)
# - Touch-friendly spacing
# - Clean dark theme, modern button styles
#
# Requires:
#   pip install ttkbootstrap matplotlib spidev RPi.GPIO
#   sudo apt install python3-tk
#
# Thermocouple requires:
#   hardware/thermocouple.py and exports in hardware/__init__.py

from __future__ import annotations

import os
os.environ["MPLBACKEND"] = "TkAgg"

import time
import threading
import queue
import csv
from typing import List, Optional, Tuple

import ttkbootstrap as tb
from ttkbootstrap.constants import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from hardware import MotorController, Sensors, ThermocoupleAD8495, ThermocoupleConfig
from acquisition import run_loading_cycle


DataPoint = Tuple[float, float, float]  # (t, disp, force)
TempPoint = Tuple[float, float]         # (t, temp_C)


class SpringLoaderApp(tb.Window):
    def __init__(self) -> None:
        super().__init__(themename="darkly")

        self.title("Automatic Spring Loader")
        self.geometry("1300x780")

        # Fullscreen "instrument mode" (Esc to exit)
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))

        # --- Matplotlib style (match dark theme) ---
        plt.style.use("dark_background")

        # ---- Hardware (simulation by default) ----
        self.motor = MotorController(simulation=True)
        self.sensors = Sensors(self.motor, simulation=True)

        # ---- Thermocouple (AD8495 + MCP3008) ----
        # IMPORTANT: set vref to match MCP3008 VREF pin wiring (3.3 or 5.0).
        self.tc_cfg = ThermocoupleConfig(
            spi_bus=0,
            spi_dev=0,      # CE0
            channel=0,      # CH0
            vref=3.3,       # <-- CHANGE TO 5.0 IF MCP3008 VREF IS 5V
            avg_samples=10,
            max_speed_hz=1350000,
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

        # ---- Data storage (current mechanical run only) ----
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
        self.area_m2_var = tb.DoubleVar(value=1.0e-6)   # m^2
        self.gauge_m_var = tb.DoubleVar(value=1.0e-2)   # m

        # ---- User-friendly unit inputs ----
        self.area_mm2_var = tb.DoubleVar(value=1.0)     # mm^2
        self.gauge_mm_var = tb.DoubleVar(value=10.0)    # mm

        # ---- Run parameters ----
        self.step_var = tb.DoubleVar(value=1.0)         # deg
        self.n_steps_var = tb.IntVar(value=50)
        self.dwell_var = tb.DoubleVar(value=0.05)
        self.rate_var = tb.IntVar(value=500)            # proxy

        # ---- Overlay storage (stress–strain only) ----
        self.previous_ss_lines = []

        # ---- Layout ----
        self._build_controls()
        self._build_plots()
        self.apply_units_to_si()

        # Start background temperature reader thread
        threading.Thread(target=self._temperature_worker, daemon=True).start()

        self.after(50, self._update_plot_from_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        control = tb.Frame(self, padding=16)
        control.pack(side=LEFT, fill=Y)

        tb.Label(control, text="Controls", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(0, 12))

        # Big live temperature readout
        self.temp_readout_var = tb.StringVar(value="--.-- °C")
        tb.Label(control, text="Live Temperature", font=("Helvetica", 11)).pack(anchor="w")
        tb.Label(control, textvariable=self.temp_readout_var, font=("Helvetica", 28, "bold")).pack(
            anchor="w", pady=(0, 12)
        )

        # Start/Stop row
        btn_row = tb.Frame(control)
        btn_row.pack(fill=X, pady=(0, 10))

        self.start_button = tb.Button(btn_row, text="Start", bootstyle=SUCCESS, command=self.start_experiment)
        self.start_button.pack(side=LEFT, fill=X, expand=True, padx=(0, 6))

        self.stop_button = tb.Button(btn_row, text="Stop", bootstyle=DANGER, command=self.stop_experiment)
        self.stop_button.pack(side=LEFT, fill=X, expand=True, padx=(6, 0))

        # Other actions
        tb.Button(control, text="New run (overlay)", bootstyle=PRIMARY, command=self.new_run_overlay).pack(
            fill=X, pady=6
        )
        tb.Button(control, text="Clear all", bootstyle=SECONDARY, command=self.clear_all).pack(fill=X, pady=6)

        tb.Separator(control).pack(fill=X, pady=14)

        tb.Button(control, text="Set zero", bootstyle=INFO, command=self.set_zero).pack(fill=X, pady=6)

        tb.Button(control, text="Save CSV", bootstyle=OUTLINE, command=self.save_csv).pack(fill=X, pady=6)
        tb.Button(control, text="Save plots (PNG)", bootstyle=OUTLINE, command=self.save_png).pack(fill=X, pady=6)
        tb.Button(control, text="Export summary", bootstyle=OUTLINE, command=self.export_summary).pack(
            fill=X, pady=6
        )

        tb.Separator(control).pack(fill=X, pady=14)

        # Run parameters
        tb.Label(control, text="Run parameters", font=("Helvetica", 12, "bold")).pack(anchor="w")

        self._labeled_entry(control, "Step [deg]", self.step_var)
        self._labeled_entry(control, "Steps up", self.n_steps_var)
        self._labeled_entry(control, "Dwell [s]", self.dwell_var)
        self._labeled_entry(control, "Rotation rate (proxy)", self.rate_var)

        tb.Separator(control).pack(fill=X, pady=14)

        # Stress–strain params
        tb.Label(control, text="Stress–strain params", font=("Helvetica", 12, "bold")).pack(anchor="w")

        self._labeled_entry(control, "Area A [mm²]", self.area_mm2_var)
        self._labeled_entry(control, "Gauge length L0 [mm]", self.gauge_mm_var)

        tb.Button(control, text="Apply A/L0 (→ SI)", bootstyle=PRIMARY, command=self.apply_units_to_si).pack(
            fill=X, pady=(8, 6)
        )

        self.si_label_var = tb.StringVar(value="")
        tb.Label(control, textvariable=self.si_label_var, foreground="#9aa0a6").pack(fill=X)

        tb.Separator(control).pack(fill=X, pady=14)

        # Status
        self.status_var = tb.StringVar(value="Idle")
        tb.Label(control, textvariable=self.status_var, foreground="#4ea1ff", font=("Helvetica", 11, "bold")).pack(
            fill=X
        )

        tb.Label(
            control,
            text="Tip: Esc exits fullscreen.\nTemperature plot runs continuously.",
            foreground="#9aa0a6",
        ).pack(fill=X, pady=(10, 0))

    def _labeled_entry(self, parent: tb.Frame, label: str, var) -> None:
        tb.Label(parent, text=label, font=("Helvetica", 10)).pack(anchor="w", pady=(10, 2))
        tb.Entry(parent, textvariable=var, bootstyle="dark").pack(fill=X)

    def _build_plots(self) -> None:
        plot_frame = tb.Frame(self, padding=12)
        plot_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        self.fig = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)

        # Layout tuned for landscape touchscreen:
        # Left column: x-t, F-t, T-t stacked
        # Right column: stress–strain spanning all rows
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.1, 1.9], height_ratios=[1.0, 1.0, 1.0])
        self.ax_xt = self.fig.add_subplot(gs[0, 0])
        self.ax_Ft = self.fig.add_subplot(gs[1, 0])
        self.ax_Tt = self.fig.add_subplot(gs[2, 0])
        self.ax_ss = self.fig.add_subplot(gs[:, 1])

        # Modern colours
        (self.line_xt,) = self.ax_xt.plot([], [], color="#4ea1ff", linewidth=2.0)
        (self.line_Ft,) = self.ax_Ft.plot([], [], color="#ff6b6b", linewidth=2.0)
        (self.line_Tt,) = self.ax_Tt.plot([], [], color="#51cf66", linewidth=2.0)
        (self.line_ss,) = self.ax_ss.plot([], [], color="#f8f9fa", linewidth=2.3)

        self._format_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

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

        # Consistent modern styling
        for ax in (self.ax_xt, self.ax_Ft, self.ax_Tt, self.ax_ss):
            ax.grid(alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

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
            tb.dialogs.Messagebox.show_error(message=str(e), title="Invalid A/L0")

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
        strain, stress = self._compute_stress_strain()
        if not strain or not stress or len(strain) < 5:
            return None

        max_stress = max(stress)
        max_strain = max(strain)
        if max_strain <= 0:
            return None

        eps_lo = 0.05 * max_strain
        eps_hi = 0.20 * max_strain

        xs = [e for e in strain if eps_lo <= e <= eps_hi]
        ys = [s for e, s in zip(strain, stress) if eps_lo <= e <= eps_hi]

        if len(xs) < 3:
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
        """Zero-references plotted data from the latest available point."""
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
                line, = self.ax_ss.plot(strain, stress, color="0.6", linewidth=1.2)
                self.previous_ss_lines.append(line)

        self._clear_current_run_only()
        self.status_var.set("Ready for new run (overlay kept)")

    def clear_all(self) -> None:
        """Clear current run and remove overlays."""
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
            _rate = int(self.rate_var.get())
            _ = _rate

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
        while self.temp_running:
            try:
                T = float(self.tc.read_temp_c())
                t = time.monotonic() - self.temp_t0
                self.temp_queue.put((t, T))
            except Exception:
                pass
            time.sleep(0.2)

    # ------------------------------------------------------------------
    # GUI thread plot updates
    # ------------------------------------------------------------------
    def _update_plot_from_queue(self) -> None:
        updated = False
        temp_updated = False
        latest_T: Optional[float] = None

        # Drain mechanical run datapoints
        try:
            while True:
                t, disp, force = self.data_queue.get_nowait()

                if t < 0 and (disp != disp or force != force):  # NaN check
                    tb.dialogs.Messagebox.show_error(
                        message=getattr(self, "_error_message", "Unknown error"),
                        title="Acquisition error",
                    )
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
                latest_T = float(T)
                temp_updated = True
        except queue.Empty:
            pass

        if latest_T is not None:
            self.temp_readout_var.set(f"{latest_T:0.2f} °C")

        if updated:
            self.line_xt.set_data(self.time_data, self.disp_data)
            self.line_Ft.set_data(self.time_data, self.force_data)

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

        filename = tb.filedialog.asksaveasfilename(
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
                w.writerow(["t [s]", "displacement [m]", "force [N]", "strain [-]", "stress [Pa]"])
                for i in range(len(self.time_data)):
                    eps = strain[i] if i < len(strain) else ""
                    sig = stress[i] if i < len(stress) else ""
                    w.writerow([self.time_data[i], self.disp_data[i], self.force_data[i], eps, sig])

                w.writerow([])
                w.writerow(["Temperature stream"])
                w.writerow(["t [s]", "T [°C]"])
                for i in range(len(self.temp_time)):
                    w.writerow([self.temp_time[i], self.temp_data[i]])

            self.status_var.set(f"Saved CSV: {os.path.basename(filename)}")
        except Exception as e:
            tb.dialogs.Messagebox.show_error(message=str(e), title="Save failed")

    def save_png(self) -> None:
        filename = tb.filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save plots as PNG",
        )
        if not filename:
            return
        try:
            self.fig.savefig(filename, dpi=220)
            self.status_var.set(f"Saved PNG: {os.path.basename(filename)}")
        except Exception as e:
            tb.dialogs.Messagebox.show_error(message=str(e), title="Save failed")

    def export_summary(self) -> None:
        summary = self._compute_summary()
        if summary is None:
            tb.dialogs.Messagebox.show_info(
                message="Not enough valid stress–strain data to compute summary.\nCheck A/L0 and collect more points.",
                title="Summary",
            )
            return

        filename = tb.filedialog.asksaveasfilename(
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
            tb.dialogs.Messagebox.show_error(message=str(e), title="Save failed")

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