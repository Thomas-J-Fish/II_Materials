# Cr₂O₃ Inclusion Shape Relaxation — Allen-Cahn Phase-Field Simulation

A Python simulation modelling how chromium oxide (Cr₂O₃) inclusions evolve their shape while suspended in a liquid Fe-28wt%Cr-14wt%Ni-8wt%Mo alloy melt during casting. 

---

## Background

During alloy preparation, the chromium feedstock carries a native Cr₂O₃ oxide layer. When added to the melt at ~2100 °C, fragments spall off and become suspended as solid inclusions resulting in dispersion hardening but also reduced toughness as the sharp edges act stress concentrators. The casting temperature is insufficient to fully dissolve them, so they persist into the final microstructure as the dark spots visible under SEM.

While the alloy is liquid, interfacial energy (σ ≈ 1.5 J m⁻²) drives the inclusion toward a more circular shape. The extent of rounding before solidification freezes the shape depends on three competing factors: inclusion size (τ ∝ R²), cooling rate (faster cooling = less time), and temperature (mobility M follows Arrhenius, dropping ~800× between 2100 °C and 1280 °C).

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

All parameters are in the `CONFIG` block at the top of `main.py`. Outputs are written to `outputs/`.

---

## Physical model

### Allen-Cahn equation

The order parameter φ (1 = oxide, 0 = melt) evolves as:

```
∂φ/∂t = M_dimless [ ε² ∇²φ − W f'(φ) + λ ]
```

The gradient term ε²∇²φ drives interface smoothing. The double-well term W f'(φ), where f(φ) = φ²(1−φ)², keeps the two phases distinct. λ is a volume-conserving Lagrange multiplier that enforces constant inclusion area — without it, Allen-Cahn minimises total perimeter by shrinking the inclusion to zero rather than just rounding it.

The solver runs in dimensionless pixel units (ε = W = M_dimless = 1). Physical time is recovered via:

```
Δt_physical = dt_dimless × dx² / (M(T) × σ)
```

### Arrhenius mobility

```
M(T) = M₀ × exp(−Q/R × (1/T − 1/T_melt))
```

M₀ is the mobility at T_melt (set in CONFIG). With Q = 250 kJ/mol, M at the solidus (1280 °C) is ~0.1% of its value at 2100 °C. This means most shape relaxation occurs in the first few seconds while the alloy is near the melt temperature. Each simulation step represents a different physical duration depending on the current temperature, and the loop exits automatically when the accumulated physical time reaches t_solid = δT / cooling_rate.

### Circularity metric

```
C = 4πA / P²
```

Area A is the pixel count inside the φ = 0.5 contour. Perimeter P counts pixel-face transitions between inside and outside pixels. This is identical to what ImageJ computes on a thresholded SEM image. A perfect discrete circle on a square grid gives C ≈ 0.63–0.65 rather than 1.0 due to the staircase boundary — this ceiling applies to both simulation and experimental measurements, making them directly comparable.

### Relaxation time scale

```
τ = R² / (M₀ × σ)
```

The crossover condition τ ~ t_solid identifies the physically interesting regime where partial rounding occurs. At M₀ = 1×10⁻¹³ m³ J⁻¹ s⁻¹:

| R | τ | Crossover cooling rate |
|---|---|---|
| 2 µm | 27 s | ~30 °C/s |
| 5 µm | 167 s | ~5 °C/s |
| 10 µm | 667 s | ~1.2 °C/s |

---

## Key parameters

| Parameter | Value | Notes |
|---|---|---|
| `dx_m` | 1.34×10⁻⁷ m/px | From SEM scale bar: 0.05 mm / 374 px |
| `T_melt` | 2100 °C | Melt hold temperature |
| `T_solidus` | 1280 °C | Shape frozen below this |
| `delta_T` | 820 °C | Full relaxation window |
| `mobility` | 1×10⁻¹³ m³ J⁻¹ s⁻¹ | Most uncertain parameter — vary to explore sensitivity |
| `sigma` | 1.5 J m⁻² | Cramb & Jimbo (1992) |
| `sweep_radii_um` | [2, 3.5, 5, 7, 10] µm | Range covering observed inclusion sizes |
| `sweep_cooling_rates` | [5, 20, 50, 100, 200] °C/s | Copper block contact: 5–100 °C/s expected |

---

## Results

### Shape evolution — cooling rate 100 °C/s

Initial (spalled, angular) and final (relaxed) inclusion shapes across the full size range, at a representative fast cooling rate. Smaller inclusions round substantially; larger inclusions show only partial rounding within the available solidification window.

![Shape comparison at 100 °C/s](shape_comparison_CR100.png)

---

### Circularity vs physical time — size dependence

At a fixed cooling rate of 50 °C/s, inclusions of different radii follow the expected R² scaling: 2 µm inclusions (purple) round rapidly within ~2 s, while 10 µm inclusions (brown) are still evolving slowly at the end of the solidification window (~16 s). The left panel shows that all runs use the same number of solver steps but the physical time axis (right) correctly reflects the R²-dependent kinetics.

![Circularity vs time by radius](circ_vs_step_by_radius.png)

---

### Circularity vs physical time — cooling rate dependence

At a fixed inclusion radius of 5 µm, slower cooling rates provide a longer window above the solidus and reach higher final circularity. The curves overlap at early times (when all samples are near 2100 °C and M is similar) and diverge as faster-cooled samples solidify first. The 5 °C/s curve (blue) continues rounding for ~60 s; the 200 °C/s curve (purple) is frozen by ~4 s.

![Circularity vs time by cooling rate](circ_vs_step_by_cooling_rate.png)

---

### Final circularity vs inclusion size

The key summary plot. Each curve represents a different cooling rate; the dashed line is the initial circularity of the unrelaxed spalled fragment. Three trends are clear:

1. **Smaller inclusions are more circular** at every cooling rate, consistent with τ ∝ R² kinetics.
2. **Slower cooling rates produce higher final circularity** at every size, because more time is spent in the high-mobility temperature regime near the melt.
3. **The curves fan out with increasing radius**, meaning cooling rate becomes progressively more discriminating for larger inclusions. For the smallest inclusions (2 µm), even the fastest cooling rate (200 °C/s) allows near-complete rounding because τ ≈ 3 s is short relative to t_solid ≈ 4 s. For 10 µm inclusions, final circularity ranges from 0.55 (5 °C/s) down to 0.27 (200 °C/s) — a clear, measurable difference that could in principle be read from experimental micrographs.

![Final circularity vs radius](circularity_vs_radius.png)

![Micrograph](micrograph.png)

---

## Interpretation

The simulation supports the hypothesis that the dark spots observed in the microstructure are spalled Cr₂O₃ fragments rather than precipitates formed during solidification. Precipitates would form with shapes controlled by crystal growth kinetics (typically dendritic or faceted) and would not show the size-dependent rounding trend. The observed morphology — smaller inclusions more circular, larger ones more angular — is a kinetic signature of inclusions that were present in the liquid melt and had partial but incomplete time to relax under interfacial energy minimisation.

The Arrhenius correction reveals that the majority of rounding occurs in the first few seconds at near-melt temperatures. By 1600 °C the mobility has already dropped tenfold, and by 1400 °C it is nearly negligible. This means the 10-second hold at 2100 °C before cooling has a disproportionate influence on final inclusion shape compared to the subsequent slow cool.

Quantitative comparison with measured circularity values from SEM image analysis would allow M₀ to be estimated for this system — a measurement that, to our knowledge, has not been directly reported in the literature for Cr₂O₃ in Fe-Cr-Ni-Mo melts.

---

## Assumptions and limitations

| Assumption | Impact if wrong |
|---|---|
| Chemically inert inclusion (no dissolution) | Partial dissolution would reduce size and change the size distribution over time |
| No melt flow | Convection would advect and deform inclusions |
| 2D simulation | Absolute circularity values differ from 3D; trends are robust |
| Isotropic interfacial energy | Cr₂O₃ has weak anisotropy; enabling it (`anisotropy_strength > 0`) produces faceted shapes |
| M₀ uncertain by ~2 orders of magnitude | Trends are robust; absolute time scales are not |
| Liquidus/solidus temperatures estimated | Thermo-Calc CALPHAD calculation recommended for quantitative accuracy |

---

## References

- Allen, S.M. & Cahn, J.W. (1979). Acta Metallurgica, 27(6), 1085–1095.
- Cramb, A.W. & Jimbo, I. (1992). ISIJ International, 32(4), 476–485.
- Provatas, N. & Elder, K. (2010). *Phase-Field Methods in Materials Science and Engineering.* Wiley-VCH.