# Running the optical solver

Install from the repository root with Python 3.10 or later:

```bash
python -m pip install -e ".[plot]"
```

NumPy and SciPy run the calculation. Matplotlib is optional and is used only
by the plotting script.

## Start with the 70 µm example

```bash
python examples/liang_70um.py --output results/liang-70um-quick
python examples/plot_liang_70um.py results/liang-70um-quick
```

This calculates a 70 × 70 µm² square with circular holes using Liang's thesis
Table 4.1. Defaults are `N=64`, `D=16`, `q=48`, `F=64`, and one eigenpair.
The [example notes](examples/liang-70um.md) list the full stack, the settings
used for the README figure, and the measured changes under refinement.

To reproduce the README figure:

```bash
python examples/liang_70um.py --N 96 --D 32 --q 64 --output results/liang-70um
python examples/plot_liang_70um.py results/liang-70um
```

Use a fresh output directory. The script checks it before calculating and
refuses a nonempty directory. A failed write can leave partial files; retain
only runs with a completed `summary.json` and use a new directory when retrying.

## Change numerical settings

| Option | Meaning | Default |
| --- | --- | --- |
| `--L-um` | Square cavity side length in µm | 70 |
| `--N` | Envelope-grid cells per side | 64 |
| `--D` | Auxiliary Fourier cutoff, `m,n=-D,...,D` | 16 |
| `--q` | Vertical integration order | 48 |
| `--F` | Shape Fourier integration order; circles use analytic coefficients | 64 |
| `--k` | Requested eigenpairs near the shift | 1 |
| `--sigma` | Complex search shift `delta+1j*alpha`, in µm⁻¹ | Selected from the periodic band edge |
| `--backend` | `prepared` or `reference` Green evaluation | `prepared` |
| `--output` | New result directory | `results/liang-70um` |

`N` is not the number of crystal periods. The grid resolves slowly varying
wave envelopes. The lattice period stays at 0.295 µm when `N` changes.
`D` controls a different approximation: the number of auxiliary spatial
harmonics used to calculate the coupling matrix.

The default shift is `min(real(eig(C))) - 0.6/L + 0.25j/L`, chosen to target
the single-lobed mode below the lowest-frequency periodic band edge of this
example. To inspect another part of the spectrum, specify the shift explicitly:

```bash
python examples/liang_70um.py --k 4 --sigma=-0.095+0.004j --output results/liang-70um-neighbors
```

The solver returns a local spectral subset. Finding the lowest-loss mode among
all bands and spatial orders requires multiple searches and mode identification.

## Read the results

- `summary.json` records the structure, reference frequency, numerical controls,
  shift, versions, timings, eigenvalues, residuals, and power budgets.
- `modes.csv` contains one row per returned mode, ordered by increasing signed
  amplitude loss `alpha`.
- `fields.npz` contains the complex fields and matrices. Read it with
  `numpy.load(..., allow_pickle=False)`.
- `selected_mode.png` is generated separately by the plotting script.

```python
import json
import numpy as np

with open("results/liang-70um/summary.json", encoding="utf-8") as handle:
    summary = json.load(handle)
with np.load("results/liang-70um/fields.npz", allow_pickle=False) as data:
    eigenvalues = data["eigenvalues"]
    fields = data["centers"]       # (mode, component, y, x)
    intensity = np.sum(abs(fields[0])**2, axis=0)
print(summary["modes"][0]["alphaL"])
```

The component order is `(Rx, Sx, Ry, Sy)`. `faces_Rx` and `faces_Sx` have shape
`(mode, Ny, Nx+1)`; `faces_Ry` and `faces_Sy` have shape `(mode, Ny+1, Nx)`.
Coordinates are stored as `x_centers_um`, `y_centers_um`, `x_faces_um`, and
`y_faces_um`. `C`, `C1D`, `Crad`, and `C2D` are complex `(4, 4)` matrices.

The convention is `eigenvalue = delta + 1j*alpha`, in µm⁻¹. `alpha` is the
amplitude-loss/gain-compensation coefficient. The power-loss coefficient is
`2*alpha`; the dimensionless thesis-style quantities are `deltaL` and `alphaL`.
The fields are normalized by `hx*hy*sum(intensity)=1`, so intensity is in
µm⁻², not watts. The figure divides it by its peak.

For the lossless optical example, `2*alpha` balances the outward edge flux plus
the signed coupling loss. The summary reports both terms and the balance error.
A small eigenpair residual verifies the discrete solve, not Fourier or grid
convergence.

## Build your own calculation

The [Python API guide](api.md) gives a complete example without a command-line
wrapper. It explains layer indices, geometry inputs, the Bragg reference,
coupling matrices, finite-grid boundary conditions, and returned field objects.
The helper functions in `examples/liang_70um.py` can also be imported from a
repository checkout. They are example code, not modules installed in the wheel.

## Earlier examples

`examples/square_finite.py` remains available. Its `demo` and `liang` presets
use the **Chapter 3, Table 3.1 right-triangle structure**, on a default
14.75 µm square. They are different from the Chapter 4 circular-hole 70 µm
example above.

```bash
python examples/square_finite.py --preset liang --output results/table3-triangle
```

`scripts/validate_convergence.py` runs the earlier Table 3.1 scans. Its controls
and results are described in [literature_validation.md](literature_validation.md).
It is not the refinement driver for the new Chapter 4 example.

## Tests

```bash
python -m pip install -e ".[dev,plot]"
python -m pytest -q
```
