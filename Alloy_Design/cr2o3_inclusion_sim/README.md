# Cr₂O₃ Inclusion Shape Relaxation — Allen-Cahn Phase-Field Simulation

A self-contained Python simulation for predicting the equilibrium morphology of
Cr₂O₃ (chromium oxide) inclusions in a Fe-28wt%Cr-14wt%Ni-8wt%Mo cast alloy,
produced as part of a Materials Science project at Cambridge.

## Physical background

During casting, spalled Cr₂O₃ fragments from the chromium feedstock enter the
melt as irregularly shaped solid inclusions.  Because the casting temperature is
insufficient to dissolve the oxide, these fragments persist in the final
microstructure.  Their shape evolves over time toward a morphology that minimises
the total interfacial energy between the Cr₂O₃ inclusion and the surrounding
alloy melt.

This simulation models that process using the **Allen-Cahn phase-field equation**:

```
∂φ/∂t = M [ ε² ∇²φ  −  W · df/dφ ]
```

where φ is the order parameter (1 = oxide, 0 = melt), ε is the gradient energy
coefficient, W is the double-well barrier height, and M is the kinetic mobility.
The model parameters are derived from the Cr₂O₃/melt interfacial energy (≈ 1.5
J/m²), following Cramb & Jimbo (1992).

## Repository structure

```
cr2o3_inclusion_sim/
│
├── README.md
├── requirements.txt
│
├── inputs/
│   └── micrograph.png          # (optional) SEM image for comparison
│
├── src/
│   ├── __init__.py
│   ├── thermodynamics.py       # Interfacial energy & Allen-Cahn parameters
│   ├── initialisation.py       # Initial inclusion geometry builders
│   ├── phase_field.py          # Allen-Cahn solver & shape metrics
│   └── visualisation.py        # Plotting and output
│
├── outputs/                    # Generated automatically on first run
│   ├── comparison.png          # Initial vs final shape side-by-side
│   ├── evolution.png           # Snapshot panel showing shape evolution
│   ├── metrics.png             # Area, perimeter, circularity vs step
│   └── micrograph_overlay.png  # (optional) overlay on SEM image
│
└── main.py                     # Entry point — configure and run here
```

## Installation

Requires Python 3.9+.  Install dependencies with:

```bash
pip install -r requirements.txt
```

No commercial software or compiled code is required.

## Running the simulation

```bash
python main.py
```

All parameters are set in the `CONFIG` dictionary at the top of `main.py`.  
No command-line arguments are needed.

## Key parameters to tune

| Parameter | Location | Effect |
|---|---|---|
| `dx_m` | `main.py` CONFIG | Grid spacing in metres — set so your inclusion spans 30–80 pixels |
| `mean_radius_px` | `main.py` CONFIG | Mean inclusion radius in pixels |
| `roughness` | `main.py` CONFIG | Shape irregularity: 0 = circle, 0.5 = very jagged |
| `n_steps` | `main.py` CONFIG | More steps = more relaxation toward equilibrium |
| `SIGMA` | `src/thermodynamics.py` | Cr₂O₃/melt interfacial energy in J/m² |
| `anisotropy_strength` | `main.py` CONFIG | 0 = isotropic; 0.05–0.15 = weak crystal anisotropy |

## Comparing to your micrographs

1. Place your SEM image in `inputs/micrograph.png`
2. Set `"micrograph_path": "inputs/micrograph.png"` in the CONFIG
3. Run — a side-by-side overlay plot will be generated automatically

The key quantitative comparators are:
- **Circularity** (4πA/P²): how rounded the inclusion is (1.0 = perfect circle)
- **Aspect ratio**: compare elongation of simulated vs observed inclusions
- **Size**: ensure `mean_radius_px × dx_m` matches the observed inclusion size

## Outputs

| File | Description |
|---|---|
| `outputs/comparison.png` | Initial (angular) vs final (relaxed) inclusion shape |
| `outputs/evolution.png` | Grid of snapshots showing shape evolution over time |
| `outputs/metrics.png` | Area, perimeter, and circularity as functions of step |
| `outputs/micrograph_overlay.png` | Simulated contour overlaid on SEM image (if provided) |

## Limitations and caveats

- The simulation is **2D** — real inclusions are 3D, and a 2D cross-section may
  overestimate irregularity.
- The interfacial energy value (1.5 J/m²) is an estimate; accurate values for
  this specific alloy composition are not available in the literature.
- The model assumes the oxide is **chemically inert** (no dissolution).  If
  partial dissolution occurs, the shape evolution will differ.
- The simulation captures **shape relaxation**, not solidification or grain growth.
  It should be compared to the as-cast microstructure, not a heat-treated one.

## References

- Cramb, A.W. & Jimbo, I. (1992). *Calculation of the interfacial properties of
  liquid steel–slag systems.* ISIJ International, 32(4), 476–485.
- Provatas, N. & Elder, K. (2010). *Phase-Field Methods in Materials Science and
  Engineering.* Wiley-VCH.
- Allen, S.M. & Cahn, J.W. (1979). *A microscopic theory for antiphase boundary
  motion and its application to antiphase domain coarsening.* Acta Metallurgica,
  27(6), 1085–1095.