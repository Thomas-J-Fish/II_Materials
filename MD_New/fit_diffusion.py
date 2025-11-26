#!/usr/bin/env python3
# fit_diffusion.py
# Fit Einstein relation to msd.out produced by in.caf2_md

import numpy as np

# Sampling setup must match in.caf2_md
DT_ps = 0.001      # timestep in ps (1 fs)
NEVERY = 100       # fix ave/time sampling interval -> 0.1 ps between lines

msdF = []
with open('msd.out') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        # Our fix wrote: c_msdF[4] c_msd_all[4]  (two columns)
        msdF.append(float(parts[0]))

msdF = np.array(msdF)
t = np.arange(len(msdF)) * (NEVERY * DT_ps)  # ps

# Use the last half of the data for linear fit (skip early transients)
n = len(t)
tfit = t[n//2:]
yfit = msdF[n//2:]

# Linear regression: MSD = m * t + c
m, c = np.polyfit(tfit, yfit, 1)
D_A2_per_ps = m / 6.0
D_cm2_per_s = D_A2_per_ps * 1e-4   # 1 Å^2/ps = 1e-4 cm^2/s

print(f"Points used: {len(tfit)}")
print(f"Slope m = {m:.6f} Å^2/ps")
print(f"D (F-) = {D_A2_per_ps:.6f} Å^2/ps   = {D_cm2_per_s:.3e} cm^2/s")
