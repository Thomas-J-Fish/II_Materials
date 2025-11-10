"""
Minimal, single-file Tkinter UI for a Raspberry Pi materials test rig.
- Start/Stop a displacement-controlled test loop
- Set ramp rate (mm/s), max extension (mm), sample rate (Hz)
- Live plot of Force vs Time and Extension vs Time
- CSV logging to ./runs/

Swap the SIMULATED sensor/actuator with real drivers:
- TODO: integrate HX711 (force) and DS18B20/thermocouple (temperature)
- TODO: integrate stepper driver control (e.g., pigpio or RPi.GPIO for STEP/DIR)

Run:  python3 app.py
"""

import os
import csv
import time
import queue
import threading
from dataclasses import dataclass, asdict

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =============================
# Hardware abstraction (mocked)
# =============================
class Actuator:
    def __init__(self, lead_mm_per_rev=2.0, steps_per_rev=200, microstep=16):
        self.lead = lead_mm_per_rev
        self.steps_per_mm = (steps_per_rev * microstep) / lead_mm_per_rev
        self.pos_mm = 0.0
        self.enabled = False
        self.lock = threading.Lock()

    def enable(self, on: bool):
        with self.lock:
            self.enabled = on

    def move_at_rate(self, rate_mm_s: float):
        """For real hardware: schedule STEP pulses via pigpio at given rate.
        Here we just update a simulated position variable.
        """
        with self.lock:
            if not self.enabled:
                return
            self.pos_mm += rate_mm_s * 0.02  # called every 20 ms

    def zero(self):
        with self.lock:
            self.pos_mm = 0.0

    def position_mm(self) -> float:
        with self.lock:
            return self.pos_mm

class ForceSensor:
    def __init__(self):
        self.bias = 0.0

    def read_n(self):
        return 0.0

    def tare(self):
        self.bias = self.read_n()

    def read_n_calibrated(self) -> float:
        # Simulated: spring-like behavior with hysteresis-ish noise
        base = max(0.0, AppGlobals.actuator.position_mm() * 0.12)
        return base + 0.3 * (time.time() % 1.0)  # crude ripple

class TempSensor:
    def read_c(self) -> float:
        return 23.0

# ================
# Shared globals
# ================
class AppGlobals:
    actuator = Actuator()
    force = ForceSensor()
    temp = TempSensor()

# ==================
# Data model & types
# ==================
@dataclass
class Sample:
    t: float
    ext_mm: float
    force_n: float
    temp_c: float

# ==============
# Worker thread
# ==============
class TestWorker(threading.Thread):
    def __init__(self, params, data_q: queue.Queue, event_stop: threading.Event):
        super().__init__(daemon=True)
        self.params = params
        self.q = data_q
        self.stop = event_stop

    def run(self):
        hz = self.params['sample_hz']
        dt = 1.0 / hz
        ramp = float(self.params['ramp_rate_mm_s'])
        maxext = float(self.params['max_extension_mm'])

        AppGlobals.actuator.enable(True)
        AppGlobals.actuator.zero()

        t0 = time.time()
        direction = +1
        next_tick = t0

        try:
            while not self.stop.is_set():
                now = time.time()
                if now < next_tick:
                    time.sleep(max(0, next_tick - now))
                    continue
                next_tick += dt

                # Update actuator position (simulate)
                AppGlobals.actuator.move_at_rate(direction * ramp)
                pos = AppGlobals.actuator.position_mm()
                if pos >= maxext:
                    direction = -1
                elif pos <= 0:
                    direction = +1

                # Read sensors
                f = AppGlobals.force.read_n_calibrated()
                tc = AppGlobals.temp.read_c()
                sample = Sample(t=now - t0, ext_mm=pos, force_n=f, temp_c=tc)
                self.q.put(sample)
        finally:
            AppGlobals.actuator.enable(False)

# =========
# The UI
# =========
class App:
    def __init__(self, root):
        self.root = root
        root.title("Pi Materials Test Rig — Minimal UI")
        root.geometry("980x640")

        self.data_q = queue.Queue()
        self.stop_evt = threading.Event()
        self.worker = None
        self.running = False
        self.rows = []
        self.csv_path = None

        self._build_controls()
        self._build_plot()
        self._schedule_ui_update()

    def _build_controls(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(side=tk.TOP, fill=tk.X)

        self.ramp_var = tk.DoubleVar(value=1.0)
        self.maxext_var = tk.DoubleVar(value=20.0)
        self.rate_var = tk.DoubleVar(value=20.0)

        ttk.Label(frm, text="Ramp rate (mm/s)").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.ramp_var, width=8).grid(row=0, column=1, padx=6)

        ttk.Label(frm, text="Max extension (mm)").grid(row=0, column=2, sticky="w")
        ttk.Entry(frm, textvariable=self.maxext_var, width=8).grid(row=0, column=3, padx=6)

        ttk.Label(frm, text="Sample rate (Hz)").grid(row=0, column=4, sticky="w")
        ttk.Entry(frm, textvariable=self.rate_var, width=8).grid(row=0, column=5, padx=6)

        ttk.Button(frm, text="Start", command=self.start_test).grid(row=0, column=6, padx=6)
        ttk.Button(frm, text="Stop", command=self.stop_test).grid(row=0, column=7, padx=6)
        ttk.Button(frm, text="E‑Stop", command=self.estop, style="Danger.TButton").grid(row=0, column=8, padx=6)
        ttk.Button(frm, text="Save CSV…", command=self.save_csv).grid(row=0, column=9, padx=6)

        self.status = tk.StringVar(value="Idle. Configure parameters and press Start.")
        ttk.Label(self.root, textvariable=self.status, padding=(10,4)).pack(side=tk.TOP, anchor="w")

        style = ttk.Style(self.root)
        style.configure("Danger.TButton", foreground="white", background="#b00020")

    def _build_plot(self):
        fig = Figure(figsize=(8,4), dpi=100)
        self.ax1 = fig.add_subplot(121)
        self.ax2 = fig.add_subplot(122)

        self.ax1.set_title("Force vs Time")
        self.ax1.set_xlabel("t (s)")
        self.ax1.set_ylabel("F (N)")

        self.ax2.set_title("Extension vs Time")
        self.ax2.set_xlabel("t (s)")
        self.ax2.set_ylabel("x (mm)")

        self.f_line1, = self.ax1.plot([], [], "tab:red")
        self.f_line2, = self.ax2.plot([], [], "tab:blue")

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.t_vals = []
        self.f_vals = []
        self.x_vals = []

    def _schedule_ui_update(self):
        self.root.after(50, self._on_timer)

    def _on_timer(self):
        # Drain queue
        drained = 0
        while not self.data_q.empty():
            s: Sample = self.data_q.get_nowait()
            self.rows.append(asdict(s))
            self.t_vals.append(s.t)
            self.f_vals.append(s.force_n)
            self.x_vals.append(s.ext_mm)
            drained += 1
        if drained:
            self._refresh_plot()
        self._schedule_ui_update()

    def _refresh_plot(self):
        self.f_line1.set_data(self.t_vals, self.f_vals)
        self.f_line2.set_data(self.t_vals, self.x_vals)
        # Autoscale nicely
        for ax in (self.ax1, self.ax2):
            ax.relim(); ax.autoscale_view()
        self.canvas.draw_idle()

    def start_test(self):
        if self.running:
            return
        try:
            params = {
                'ramp_rate_mm_s': float(self.ramp_var.get()),
                'max_extension_mm': float(self.maxext_var.get()),
                'sample_hz': float(self.rate_var.get()),
            }
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric values.")
            return

        self.status.set("Running… Press Stop to end; E‑Stop to cut motion.")
        self.stop_evt.clear()
        self.worker = TestWorker(params, self.data_q, self.stop_evt)
        self.worker.start()
        self.running = True

    def stop_test(self):
        if not self.running:
            return
        self.stop_evt.set()
        self.worker.join(timeout=2.0)
        self.running = False
        self.status.set("Stopped. You can save CSV or start another run.")

    def estop(self):
        # In real hardware: also drop driver EN pin or power relay.
        self.stop_evt.set()
        AppGlobals.actuator.enable(False)
        self.running = False
        self.status.set("EMERGENCY STOP — motion disabled.")

    def save_csv(self):
        if not self.rows:
            messagebox.showinfo("No data", "Run a test first.")
            return
        os.makedirs("runs", exist_ok=True)
        default = time.strftime("runs/run_%Y%m%d_%H%M%S.csv")
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(default), filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t","ext_mm","force_n","temp_c"])
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        self.status.set(f"Saved: {path}")


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()