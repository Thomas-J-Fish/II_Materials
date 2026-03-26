# Cr₂O₃ Inclusion Shape Relaxation — Allen-Cahn Phase-Field Simulation

A Python simulation modelling how chromium oxide (Cr₂O₃) inclusions evolve their shape while suspended in a liquid Fe-28wt%Cr-14wt%Ni-8wt%Mo alloy melt during casting.

---

## Physical background

During alloy preparation, the chromium feedstock carries a native Cr₂O₃ oxide layer. When the feedstock is added to the melt at ~2100 °C, fragments of this oxide spall off and become suspended in the liquid alloy as solid inclusions with irregular, angular shapes. Because the casting temperature is not high enough to dissolve the oxide, these inclusions persist into the final microstructure.

While the alloy remains liquid, the oxide–melt interface has a positive interfacial energy (σ ≈ 1.5 J m⁻²). The system minimises its total energy by reducing the interfacial area — meaning the inclusion tends to round its sharp corners and evolve toward a more circular shape. When the alloy solidifies, the inclusion shape is frozen permanently. The extent of rounding depends on:

- **Inclusion size**: larger inclusions take longer to round (relaxation time τ ∝ R²)
- **Cooling rate**: faster cooling reaches solidification sooner, freezing the shape earlier
- **Temperature**: the interface mobility M follows an Arrhenius relationship, so rounding is fastest near the melt temperature and slows dramatically as the alloy cools

This simulation predicts the degree of shape relaxation as a function of these parameters and compares results across a range of inclusion sizes and cooling rates.

---

## Installation

Requires Python 3.9 or higher.

```bash
git clone <repo>
cd cr2o3_inclusion_sim
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations, numerical methods |
| `matplotlib` | All plotting and figure output |
| `scipy` | Signed distance transform for interface initialisation |
| `scikit-image` | Polygon rasterisation, image I/O for micrograph comparison |

---

## Repository structure

```
cr2o3_inclusion_sim/
├── main.py                   Entry point. All configuration lives here.
├── requirements.txt
├── inputs/
│   └── micrograph.png        Place your SEM image here (optional)
├── outputs/                  All generated PNGs written here automatically
└── src/
    ├── __init__.py           Package exports
    ├── thermodynamics.py     Allen-Cahn solver parameters (ε, W, mobility)
    ├── initialisation.py     Builds initial φ arrays (polygon, ellipse, image)
    ├── phase_field.py        Core solver: step(), run_simulation(), metrics
    ├── cooling.py            Physical time scale mapping and Arrhenius mobility
    └── visualisation.py      Matplotlib plotting functions
```

---

## Running the simulation

```bash
python main.py
```

All parameters are set in the `CONFIG` dictionary at the top of `main.py`. No command-line arguments are needed. Outputs are written to `outputs/`.

---

## Configuration reference

All tunable parameters are in the `CONFIG` block in `main.py`.

### Grid and scale

| Key | Default | Description |
|---|---|---|
| `grid_size` | `256` | Simulation grid pixels per side. Larger = higher resolution but slower. |
| `dx_m` | `1.34e-7` | Physical size of one pixel in metres. Derived from your SEM scale bar: 0.05 mm / 374 px. |

### Inclusion geometry

| Key | Default | Description |
|---|---|---|
| `geometry` | `"irregular"` | Shape type: `"irregular"` (spalled fragment) or `"ellipse"` (reference case) |
| `n_vertices` | `12` | Number of polygon vertices. More gives a more complex initial shape. |
| `roughness` | `0.38` | Fractional spread of vertex radii. 0 = circle, 0.5 = very jagged. |
| `seed` | `7` | Random seed. Change to generate a different polygon shape. |
| `interface_width` | `3` | Diffuse interface half-width in pixels. Controls the tanh transition zone. |

### Physical parameters

| Key | Default | Description |
|---|---|---|
| `delta_T` | `820.0` | Temperature window for shape relaxation [°C]. T_melt − T_solidus = 2100 − 1280. |
| `T_melt` | `2100.0` | Initial melt temperature [°C]. |
| `T_solidus` | `1280.0` | Solidification temperature [°C]. Shape is frozen below this. |
| `mobility` | `1e-13` | Interface mobility M₀ at T_melt [m³ J⁻¹ s⁻¹]. Most uncertain parameter. |
| `sigma` | `1.5` | Cr₂O₃/melt interfacial energy σ [J m⁻²]. From Cramb & Jimbo (1992). |

### Sweep parameters

| Key | Default | Description |
|---|---|---|
| `sweep_radii_um` | `[2.0, 3.5, 5.0, 7.0, 10.0]` | Inclusion radii to simulate [µm] |
| `sweep_cooling_rates` | `[5.0, 20.0, 100.0, 400.0]` | Cooling rates to simulate [°C/s] |
| `max_steps` | `8000` | Safety ceiling on simulation steps. The loop exits automatically at solidification. |
| `min_steps` | `500` | Minimum steps (prevents trivially short runs for very fast cooling). |
| `n_snapshots` | `40` | Number of saved snapshots per run for the evolution plots. |

### Choosing `mobility`

Mobility is the dominant source of uncertainty. Use the relaxation time τ = R²/(M₀σ) to guide your choice. For physically meaningful cooling rate differentiation you want τ to be comparable to t_solid = δT/CR for your inclusion size range:

| Mobility M₀ | τ at R=5µm | Crossover cooling rate |
|---|---|---|
| 1×10⁻¹² | 17 s | ~50 °C/s |
| 1×10⁻¹³ | 167 s | ~5 °C/s |
| 1×10⁻¹⁴ | 1667 s | ~0.5 °C/s |

---

## Outputs

| File | Description |
|---|---|
| `circularity_vs_radius.png` | Final circularity vs inclusion radius, one curve per cooling rate. Key result plot. |
| `circ_vs_step_by_cooling_rate.png` | Circularity evolution over simulation steps and physical time for different cooling rates at a fixed radius. |
| `circ_vs_step_by_radius.png` | Circularity evolution for different inclusion sizes at a fixed cooling rate. |
| `shape_comparison_CR***.png` | Side-by-side initial vs final inclusion shapes for each cooling rate, across all radii. |

---

## Comparing to your micrographs

1. Place your SEM image in `inputs/micrograph.png`
2. Set `"micrograph_path": "inputs/micrograph.png"` in CONFIG
3. Run — a micrograph overlay plot is generated automatically

The circularity metric used (4πA/P²) is computed identically to standard image analysis software (ImageJ etc.), using pixel-count area and pixel-face-transition perimeter. A perfect discrete circle on a square grid gives circularity ~0.63–0.65 rather than 1.0 due to the staircase boundary — this is the correct ceiling for both simulation and experimental measurements, making them directly comparable.

---

## Physical model summary

### Governing equation — Allen-Cahn

```
∂φ/∂t = M_dimless [ ε² ∇²φ − W f'(φ) + λ ]
```

- **φ** (phi): order parameter. φ=1 inside oxide, φ=0 in melt, smooth tanh transition at the interface.
- **ε²∇²φ**: gradient energy term. Drives interface smoothing and corner rounding.
- **W f'(φ)**: double-well term. Keeps φ close to 0 or 1 in bulk regions; f(φ) = φ²(1−φ)².
- **λ**: volume-conserving Lagrange multiplier. Enforces constant inclusion area (no growth or shrinkage, only shape change).

### Temperature-dependent mobility — Arrhenius

```
M(T) = M₀ × exp(−Q/R × (1/T − 1/T_melt))
```

- **M₀**: mobility at T_melt (set in CONFIG as `mobility`)
- **Q**: activation energy, 250 kJ/mol (typical for solid oxide / liquid metal)
- At solidus (1280°C), M has dropped to ~0.1% of its value at 2100°C

### Physical time conversion

```
Δt_physical = dt_dimless × dx² / (M(T) × σ)
```

Each dimensionless solver step corresponds to a physical duration that decreases as the alloy cools. The simulation accumulates these increments and stops when the total reaches t_solid = δT/cooling_rate.

### Relaxation time scale

```
τ = R² / (M₀ × σ)
```

This is the characteristic time for an inclusion of radius R to round significantly. When τ << t_solid, the inclusion fully rounds before solidification. When τ >> t_solid, the inclusion barely changes shape. The crossover condition τ ~ t_solid determines which combinations of size and cooling rate produce partial rounding — the physically interesting regime for this study.

---

## Key assumptions and limitations

| Assumption | Justification | Impact if wrong |
|---|---|---|
| Isothermal relaxation kinetics | The solver runs in dimensionless time; Arrhenius corrects the time scale | Low: Arrhenius is well-established for this type of process |
| Chemically inert inclusion | No dissolution of Cr₂O₃ into the melt | Medium: partial dissolution would reduce inclusion size over time |
| No melt flow | Fluid convection neglected | Medium: flow would advect and deform inclusions |
| 2D simulation | Real inclusions are 3D | Low for shape trends; absolute circularity values would differ |
| Isotropic interfacial energy | `anisotropy_strength = 0` | Low: Cr₂O₃ has weak anisotropy; enabling it produces faceted shapes |
| M₀ uncertain by ~2 orders of magnitude | No direct measurements for this system | High for absolute time scales; trends are robust |

---

## References

- Allen, S.M. & Cahn, J.W. (1979). *A microscopic theory for antiphase boundary motion.* Acta Metallurgica, 27(6), 1085–1095.
- Cramb, A.W. & Jimbo, I. (1992). *Interfacial properties of liquid steel–slag systems.* ISIJ International, 32(4), 476–485.
- Provatas, N. & Elder, K. (2010). *Phase-Field Methods in Materials Science and Engineering.* Wiley-VCH.