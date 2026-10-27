# Python API

The optical calculation has four steps: solve the vertical waveguide, build the
four-wave coupling matrix, assemble the finite cavity, and recover its fields.
All lengths below are in micrometres. Relative permittivity is `n**2`, not `n`.

## A complete calculation

This is the circular-hole structure in Liang's thesis, Table 4.1, on a
70 × 70 µm² square. The two outer claddings extend to infinity in this model.

```python
import numpy as np
from scipy.optimize import brentq
from cwt3d.geometry import CircleHole
from cwt3d.waveguide import TEWaveguide
from cwt3d.coupling import build_coupling
from cwt3d.finite.grid import StaggeredGrid
from cwt3d.finite.assembly import FinitePencil
from cwt3d.finite.eigensolver import solve_finite
from cwt3d.finite.fields import reconstruct_modes

period = 0.295
hole = CircleHole(
    radius_um=period * np.sqrt(0.16 / np.pi),
    period_um=(period, period),
    eps_background=3.524**2,
    eps_hole=1.0,
)
eps = [
    3.307**2, 3.524**2, 3.553**2, 3.524**2, 3.279**2,
    3.524**2, hole.mean_epsilon.real, 3.524**2, 3.307**2,
]
thickness = [0.030, 0.064, 0.040, 0.040, 0.022, 0.118, 0.020]

def make_guide(k0):
    return TEWaveguide(eps, thickness, k0=k0, period_um=period)

# Choose the reference frequency at the second-order Bragg condition.
beta0 = 2 * np.pi / period
bracket = (
    beta0 / np.sqrt(max(eps)) * (1 + 1e-7),
    beta0 / np.sqrt(max(eps[0], eps[-1])) * (1 - 1e-7),
)
k0 = brentq(
    lambda k: make_guide(k).solve_mode().beta - beta0,
    *bracket, xtol=1e-13,
)
vertical_mode = make_guide(k0).solve_mode()

coupling = build_coupling(
    vertical_mode, period, {6: hole},
    cutoff=16, quadrature_order=48, fourier_order=64,
    backend="prepared",
)
length = 70.0
grid = StaggeredGrid(Nx=64, Ny=64, Lx_um=length, Ly_um=length)
pencil = FinitePencil(grid, coupling)

# Target the single-lobed finite-cavity mode below the lowest band edge.
band_edge = np.min(np.linalg.eigvals(coupling.C).real)
sigma = complex(band_edge - 0.6 / length, 0.25 / length)
eigenpairs = solve_finite(
    pencil, sigma=sigma, k=1,
    tol=1e-10, residual_tol=1e-9, seed=33,
)
modes = reconstruct_modes(pencil, eigenpairs, coupling=coupling)
selected = modes[0]
print("delta L =", selected.delta * length)
print("alpha L =", selected.alpha * length)
print("residual =", selected.residual)
intensity = selected.intensity  # shape: (Ny, Nx)
```

The shift is a search location in the complex eigenvalue plane. It selects a
neighborhood of the spectrum; it does not impose the returned eigenvalue.
Change `sigma` and request more eigenpairs with `k` when studying other branches.
A single search does not identify every competing mode.

## Geometry and vertical waveguide

| Interface | Input | Result |
| --- | --- | --- |
| `CircleHole(radius_um, period_um, center_um=None, eps_background=..., eps_hole=...)` | One inclusion in a rectangular cell | Callable dielectric profile, `.fill_fraction`, `.mean_epsilon` |
| `RightTriangleHole(legs_um, period_um, center_um=None, eps_background=..., eps_hole=...)` | Right-angled triangular inclusion | The same geometry interface; an equilateral triangle is a different shape |
| `SampledPermittivity(values, period_um)` | Cell-centered array with axes `(y, x)` | A periodic dielectric map for FFT Fourier coefficients |
| `TEWaveguide(eps, thickness_um, *, k0=None, wavelength_um=None, period_um=None)` | Two halfspaces plus finite layers | A lossless scalar TE waveguide |
| `guide.solve_mode(index=0)` | Guided-mode index, fundamental first | A normalized vertical `TEMode` |

`eps[0]` and `eps[-1]` are halfspaces. `thickness_um` describes only
`eps[1:-1]`, so its length is `len(eps) - 2`. Supply exactly one of `k0` and
`wavelength_um`; `k0` is the vacuum wavenumber in µm⁻¹. Guide parameters are
read-only: construct a new guide when changing the stack or frequency.

A `TEMode` provides `.beta`, `.guide`, and `.boundary_residual`. The
normalization includes both cladding tails. It is separate from the later
normalization of a finite-cavity envelope.

## Coupling matrix

```python
coupling = build_coupling(
    vertical_mode, period, {6: hole},
    cutoff=16, quadrature_order=48, fourier_order=64,
    backend="prepared",
)
C = coupling.C
```

The dictionary keys are indices in `guide.eps`, **including** the leading
halfspace. Here the patterned layer is `eps[6]`. Its averaged permittivity must
match the geometry's `.mean_epsilon` (or `np.mean(samples.values)` for a
`SampledPermittivity`). A second patterned layer can be added as another
dictionary entry. For a right triangle, `center_um` is the centroid, not the
right-angle vertex.

`build_coupling` requires `vertical_mode.beta` to match `2π/period` to the stated
tolerance. Solve for the reference frequency as above instead of loosening that
tolerance to accept an unrelated wavelength.

The result contains complex `(4, 4)` matrices, in µm⁻¹:

- `.C`: total coupling, in the order `(Rx, Sx, Ry, Sy)`.
- `.C1D`: direct back-diffraction.
- `.Crad`: coupling through the zeroth-order radiating field.
- `.C2D`: coupling through the retained higher spatial harmonics.

`cutoff` sets `m,n = -D,...,D` for the auxiliary harmonics, excluding the four
basic waves and the radiation order. `quadrature_order` controls vertical
integration. `fourier_order` controls shape integration where used; circular
holes use analytic Fourier coefficients. `reference` and `prepared` evaluate
the same equations; `prepared` reuses the vertical Green-function systems.

### Calculation setting: reciprocity reuse

`reciprocity=True` is the default for both backends. It reuses identical-interval
Green values using scalar reciprocity, without requiring geometric symmetry.
Set `reciprocity=False` to restore the original two-evaluation formula:

```python
coupling = build_coupling(
    vertical_mode, period, {6: hole},
    cutoff=16, quadrature_order=48, fourier_order=64,
    backend="prepared", reciprocity=False,
)
```

Both `examples/liang_70um.py` and `examples/square_finite.py` expose
`--no-reciprocity` to disable it and `--reciprocity` to explicitly enable it.
The chapter-4 `run_case(...)` API accepts the same `reciprocity` boolean.
`coupling.reciprocity`, the provenance and exported `summary.json` input record
retain the selected setting. Cache keys distinguish enabled and disabled runs,
so disabling reuse cannot return an enabled-run cache entry. Unknown kernels
and overriding subclasses keep the original formula even when enabled.
Precision and quadrature controls are unchanged.

For an infinite periodic structure, use
`cwt3d.infinite.solve_infinite(coupling)`. Its four eigenvalues describe the
periodic band edges, without finite-cavity edge loss.

## Finite cavity and modes

`StaggeredGrid(Nx, Ny, Lx_um, Ly_um)` defines the envelope grid. `Nx` and `Ny`
are numerical cells, not numbers of photonic-crystal periods. The number of
free unknowns is `4*Nx*Ny`.

`FinitePencil(grid, coupling)` assembles sparse matrices for
`A @ v = eigenvalue * B @ v`. It also accepts a numeric complex `(4, 4)` matrix
in place of `coupling`. Only the incoming wave is zero at each cavity edge:

```text
Rx(left) = 0     Sx(right) = 0
Ry(bottom) = 0   Sy(top) = 0
```

`solve_finite(pencil, sigma=..., k=..., tol=..., residual_tol=..., seed=...)`
returns an `EigenResult`:

| Attribute | Meaning |
| --- | --- |
| `.eigenvalues` | Complex `delta + 1j*alpha`, in µm⁻¹ |
| `.eigenvectors` | Free-face vectors, one eigenvector per column |
| `.residuals` | Residuals of the original generalized eigenproblem |
| `.status`, `.returned` | Convergence status and valid pair count |
| `.lu_seconds`, `.iteration_seconds` | Measured factorization and iteration times |

The core solver orders pairs by distance to `sigma`. The chapter-4 example
reorders its displayed/exported modes by increasing `alpha`. Neither ordering
is a global threshold ranking. Unconverged solves raise `EigenSolveError` by
default; its `.partial` contains checked partial results if available.

`reconstruct_modes(pencil, eigenpairs, coupling=coupling)` returns normalized
`FiniteMode` objects. Each provides:

- `.centers.Rx`, `.centers.Sx`, `.centers.Ry`, `.centers.Sy`: complex `(Ny, Nx)` arrays.
- `.faces.Rx`, `.faces.Sx`: `(Ny, Nx+1)`; `.faces.Ry`, `.faces.Sy`: `(Ny+1, Nx)`.
- `.intensity`: `|Rx|² + |Sx|² + |Ry|² + |Sy|²` at cell centers.
- `.delta`, `.alpha`, `.residual`, `.boundary_error`.
- `.budget`: signed edge and coupling power-loss rates and their balance error.

The finite mode is normalized by `hx*hy*sum(intensity) = 1`.
`alpha = Im(eigenvalue)` is an amplitude-loss/gain-compensation coefficient;
the corresponding power-loss coefficient is `2*alpha`. The envelope intensity
is not an optical power in watts or the interference-resolved microscopic
field. Material threshold gain also requires the active-region confinement
factor and any additional losses.

For this lossless example, the independently evaluated budget satisfies
`2*alpha = budget['edge'] + budget['coupling']`. Passing the matching physical
`CouplingResult` also gives `C1D`, `Crad`, and `C2D` contributions. The radiative
scope is recorded in the budget; do not assume zeroth-order radiation is the
only loss channel for every possible structure.
