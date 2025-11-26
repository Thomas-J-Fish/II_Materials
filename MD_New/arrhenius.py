#!/usr/bin/env python3
# arrhenius.py
# Fit ln(D) vs 1/T to extract activation energy Ea

import numpy as np

# TODO: Fill with your measured values (example placeholders)
T  = np.array([900, 1000, 1100, 1200], dtype=float)    # K
D  = np.array([1.1e-6, 2.3e-6, 4.7e-6, 9.0e-6])        # cm^2/s

x = 1.0 / T
y = np.log(D)

m, c = np.polyfit(x, y, 1)  # ln D = ln A - Ea/(R) * (1/T)
R = 8.314462618  # J/mol/K
Ea_J_per_mol = -m * R
Ea_kJ_per_mol = Ea_J_per_mol / 1000.0

print(f"Slope = {m:.6e}  ->  Ea = {Ea_kJ_per_mol:.2f} kJ/mol")
print(f"Pre-exponential ln A = {c:.3f}  ->  A = {np.exp(c):.3e} cm^2/s")
