# 3D-CWT

Three-dimensional coupled-wave calculations for square-lattice photonic-crystal
surface-emitting lasers (PCSELs). The Python package calculates the vertical
waveguide mode, the four-wave coupling matrix, and finite-cavity modes on a
staggered grid.

The current optical solver uses NumPy and SciPy.

Current release: **1.3.1**.

## Citing 3D-CWT

If you use this code in your research, please cite:

1. H. Huang, Z. Xu, Q. Xin, and Z. Zhang, “Towards efficient PCSEL design: a fully AI-driven approach,” *Chinese Optics Letters*, **24**(4), 043602 (2026). [doi:10.3788/COL202624.043602](https://doi.org/10.3788/COL202624.043602).
2. H. Huang, C. Yang, H. Li, and Z. Zhang, “Unveiling the potential of photonic crystal surface emitting lasers: a concise review,” *Semiconductor Science and Technology*, **40**(4), 43001 (2025). [doi:10.1088/1361-6641/adb7fd](https://doi.org/10.1088/1361-6641/adb7fd).

<details>
<summary>BibTeX</summary>

```bibtex
@article{huang_Efficient_2026,
  title = {Towards Efficient {{PCSEL}} Design: A Fully {{AI-driven}} Approach},
  shorttitle = {Towards Efficient {{PCSEL}} Design},
  author = {Huang, Hai and Xu, Ziteng and Xin, Qi and Zhang, Zhaoyu},
  journal = {Chinese Optics Letters},
  volume = {24},
  number = {4},
  pages = {043602},
  year = 2026,
  doi = {10.3788/COL202624.043602},
  langid = {english}
}

@article{huang_Unveiling_2025,
  title = {Unveiling the Potential of Photonic Crystal Surface Emitting Lasers: A Concise Review},
  shorttitle = {Unveiling the Potential of Photonic Crystal Surface Emitting Lasers},
  author = {Huang, Hai and Yang, Chuanyi and Li, Hui and Zhang, Zhaoyu},
  year = 2025,
  month = apr,
  journal = {Semiconductor Science and Technology},
  volume = {40},
  number = {4},
  pages = {43001},
  publisher = {IOP Publishing},
  issn = {0268-1242, 1361-6641},
  doi = {10.1088/1361-6641/adb7fd},
  langid = {english}
}
```

</details>

## Example: a 70 µm square cavity

This example uses the circular-hole (CC) structure from Chapter 4 of
[Yong Liang's doctoral thesis](https://doi.org/10.14989/doctor.k18283),
Table 4.1: a 295 nm square lattice, 16% air filling fraction, and a
70 × 70 µm² cavity. The figure shows a calculated single-lobed mode and its
center-line intensity profiles.

![Calculated envelope intensity in the 70 µm square cavity](docs/images/liang-70um.png)

The plotted quantity is the slowly varying envelope
`|Rx|² + |Sx|² + |Ry|² + |Sy|²`, divided by its peak. The numerical results,
layer stack, and refinement checks are in the
[70 µm example notes](docs/examples/liang-70um.md).

## Validation against Liang's thesis

We tested the published `cwt3d 1.3.0` against Chapters 3 and 4 of the thesis:
Fourier-cutoff convergence, three hole shapes over filling fractions 0.08–0.24,
finite-cavity spectra and fields, envelope-grid refinement, and cavity sizes
40–400 µm. CC denotes circular holes, ET equilateral triangles, and RIT
right-isosceles triangles.

![Three hole shapes: filling-fraction scans against thesis CWT and FDTD markers](docs/validation/liang-reproduction/analysis/02_three_shapes_FF.png)

For the 70 µm structure, the following amplitude-loss thresholds use the
paper's Fourier cutoff `D=10` and a refined envelope grid `N=96`:

| Mode | CC: thesis αL | CC: calculated αL | ET: thesis αL | ET: calculated αL |
|---|---:|---:|---:|---:|
| A | 0.23 | 0.227058 | 0.43 | 0.427323 |
| B | 0.52 | 0.525020 | 0.57 | 0.568033 |
| C | 2.18 | 2.172118 | 2.08 | 1.934732 |
| D | 2.18 | 2.172118 | 2.01 | 2.008507 |

Seven of the eight thresholds differ by at most 1.3%; **ET-C remains 6.98%
lower than the thesis**. Its frequency position also differs. Increasing `D`
does not remove this discrepancy. The `N=64→96` threshold change is at most
0.0368%, but this is not a bound on the continuum error or proof of Fourier
convergence: the RIT B-mode radiation loss still changes by 2.93% at `D=24→32`.

The three-shape frequency curves closely follow the thesis **CWT** results.
Default ET/RIT radiation losses have mean relative differences of about
14.5%/17.4% for reference losses above 3 cm⁻¹. A separate calculation using the
thesis's approximate Green function gives closer CWT agreement; it is not the
default layered model, and neither comparison establishes agreement with all
FDTD points. The size-dependent A/B threshold exchange is reproduced, and the
edge-power fractions differ from Table 4.3 by at most 0.67 percentage points.

See the [full report and comparison data](docs/validation/liang-reproduction/README.md)
and the [seven-page PDF](docs/validation/liang-reproduction/liang-reproduction-report.pdf)
for the δL–αL spectra, mode fields, convergence curves, model differences,
reading tolerances, and independent checks. No solver parameters were fitted
to the paper.

## Install and run

Run these commands from the repository root.
Python 3.10 or later is required. The `plot` extra installs Matplotlib for the
figure; the calculation itself only needs NumPy and SciPy.

```bash
python -m pip install -e ".[plot]"
python examples/liang_70um.py --N 96 --D 32 --q 64 --output results/liang-70um
python examples/plot_liang_70um.py results/liang-70um
```

The first script writes `summary.json`, `modes.csv`, and `fields.npz`. The second
writes `selected_mode.png`. Choose a new output directory for each calculation;
the solver refuses to overwrite an existing result.

For a quicker first run, use the lower-order defaults:

```bash
python examples/liang_70um.py --output results/liang-70um-quick
```

`--N` is the number of envelope-grid cells per side, `--D` is the auxiliary
Fourier cutoff, and `--q` is the vertical integration order. They control
different numerical errors. The quick run and the plotted run use the same
physical structure. The plotted introductory example uses `D=32`; the validation
table above deliberately uses the thesis's `D=10` for comparison.

### Reciprocity acceleration

Reciprocal Green-value reuse is **enabled by default**. It does not require
geometric symmetry of the holes or layer stack within the supported scalar TE
model. To use the original two-evaluation formula, add `--no-reciprocity`:

```bash
python examples/liang_70um.py --no-reciprocity --output results/liang-70um-no-reciprocity
```

The Python setting is `reciprocity=False`, for example
`run_case(N=64, D=10, q=32, reciprocity=False)`. The selected setting is recorded
in the output and uses a separate cache key. Both paths keep the same precision
and integration controls.

Three matched-precision benchmarks (circle, asymmetric two-layer patterns, and
an irregular sampled hole) gave **1.59–1.85× coupling speedups**, or **1.54–1.78×
for coupling plus the finite solve and field reconstruction**. These are
workload-dependent measurements, not a universal twofold speedup. See the
[derivation, accuracy tests and raw timing records](docs/validation/reciprocal-green-overlap/README.md).

## Use the calculation from Python

The example exposes the calculation as a function. Run this from the repository
root after installation:

```python
from examples.liang_70um import run_case

case = run_case(N=96, D=32, q=64)
mode = case.modes[0]

print("delta L:", mode.delta * 70)
print("alpha L:", mode.alpha * 70)
print("eigenpair residual:", mode.residual)

intensity = mode.intensity       # shape: (Ny, Nx)
Rx = mode.centers.Rx             # complex envelope at cell centers
C = case.coupling.C              # 4 x 4 coupling matrix, in µm^-1
A, B = case.pencil.A, case.pencil.B
```

`run_case` returns the waveguide reference, coupling matrix, sparse pencil,
eigenpairs, reconstructed fields, and a summary dictionary. Pass `output=...`
to save them. To build a different structure or reuse a coupling matrix, follow
the [complete Python API example](docs/api.md).

The default search targets one mode near the lowest-frequency periodic band
edge. `--k` and `--sigma` select a larger set or a different part of the
spectrum; one search is not a survey of all competing modes. Eigenvalues are
`delta + 1j*alpha`, where `alpha` is an amplitude-loss coefficient. Power loss
is `2*alpha`.

## Documentation

- [Running examples and reading the output](docs/usage.md)
- [Python interfaces and a complete custom calculation](docs/api.md)
- [70 µm structure, results, and numerical checks](docs/examples/liang-70um.md)
- [Liang thesis reproduction report, figures, and comparison tables](docs/validation/liang-reproduction/README.md)
- [Model conventions and earlier literature checks](docs/literature_validation.md)
- [Numerical tests](docs/validation.md) and [performance measurements](docs/performance.md)
- [Publishing releases to PyPI](docs/publishing.md)

The current model covers lossless scalar TE waves on a square lattice. The
70 µm calculation uses a layered outgoing Green function and halfspace
claddings. The report above separates these modeling assumptions from the
thesis approximation. Carrier transport and above-threshold laser dynamics
are not included in the new package.

To run the tests:

```bash
python -m pip install -e ".[dev,plot]"
python -m pytest -q
```

