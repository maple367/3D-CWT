# Validation: legacy characterization versus v3 acceptance

## Scope and isolation

`tests/regressions` explicitly opts into the **unfixed legacy baseline**
`520f4d8-unfixed` via `LEGACY_BASELINE`. This label identifies historical
behavior, not a runtime HEAD check. These tests are not tests of a v3 solver.

`_legacy.py` parses checked-in `model/_Model.py` and `calculator/_calc.py`,
selects only an allowlisted class and its original method ASTs, and compiles
those methods inside an empty class with the original name. Original method
bodies and traceback filenames/line numbers are preserved. In particular,
`Model` compilation preserves `self.__eps_profile_z` name mangling; the fixture
binds `_Model__eps_profile_z` explicitly. No legacy module import, constructor,
base class, COMSOL process, Numba dependency or arbitrary-input `eval` is used.
This extraction is for trusted repository source, not an untrusted-code sandbox.

## Frozen defects and independent targets

- **chi**: two nonempty layers, one absent layer, cutoff 1 and dimensionless
  overlap/varsigma fixtures give a hand-computable sum (10 for x, -5j for y).
  The actual `_chi_func` returns the bare sum rather than equation (3.35),
  `chi = -k0^2/(2 beta0) * sum`. The separate oracle checks sign, magnitude
  and inverse-length scaling when both wavenumbers scale with fixed
  dimensionless material/overlap data. This is a controlled dimensional
  fixture, not a full physical geometry rescaling calculation.
- **Green**: the actual `beta_z_func_higher_order` gives gamma^2=35 for
  k0=1, beta0=3, epsilon=1+0j, order (2,0). With reflection-free
  kappa=(1,0,0,1), the actual legacy Green function has relative residual
  approximately 2 for `(d_z^2-gamma^2)G=0`, on both sides away from the
  source. Independent `exp(-gamma*abs(z-z'))/(2 gamma)` passes the same
  finite-difference operator check, decays, is continuous, and has derivative
  jump -1, corresponding to `(d_z^2-gamma^2)G=-delta`.
- **DFT**: actual `_cal` calls on fresh instances with valid zero, positive
  and negative indices first assign `array[index]` and then raise
  `AttributeError` at the original `return self._xi`. Traceback and stored
  array value distinguish this defect from import/index/build failures.
  Separately, direct uniform unit-cell quadrature checks area normalization,
  positive coefficient phase, negative reconstruction phase, translation,
  axis ordering and complex amplitude on two grid sizes. It neither calls
  the legacy DFT nor claims that a new Fourier implementation exists.

No skip/xfail masks these defects. Green characterization tests mean the
historical bugs were reproduced, **not fixed**. If the legacy files remain
unchanged, these tests should keep passing even after v3 gains correct APIs.
If legacy behavior is intentionally repaired, revise its explicit baseline
and characterization expectations rather than treating them as v3 acceptance.

## Execution evidence (Task 2)

1. Tests were written first. Initial
   `.venv/Scripts/python.exe -m pytest tests/regressions -q` returned
   **3 collection errors** because the package/helper did not yet exist.
   This is scaffolding RED, not evidence of a physics failure.
2. Adding only the isolation helper/package made the characterization suite
   **13 passed**. No old implementation was edited to obtain GREEN.
3. Additional physical-acceptance probes at
   `.validation-scratch/task2_legacy_physical_red.py` genuinely failed
   against the isolated old methods: chi 10 instead of -6.666666666666666,
   Green relative residual 1.9999999702493498, DFT AttributeError at
   `_calc.py:189` (**3 failed**). These temporary probes are deliberately
   outside the normal suite; legacy has not been made physically correct.

4. Final full suite: **34 passed**; `git diff --check` passed. A fresh
   interpreter probe compiling all three isolated classes found no loaded
   `model`, `calculator`, `mph` or `numba` modules. Legacy source, `src` and
   `pyproject.toml` have no diff.

Re-run the repository checks with:

```text
.venv/Scripts/python.exe -m pytest tests/regressions -q
.venv/Scripts/python.exe -m pytest -q
git diff --check
```

## Future new-API acceptance

Keep these historical tests separate. When the new Coupling, Green and
Fourier APIs exist, add dedicated tests calling those real APIs with the
same deterministic fixtures and independent targets: chi formula/scaling,
source-free Green ODE plus jump/decay, and direct Fourier coefficients.
Do not merely redirect legacy bug-expectation assertions to the new API or
claim coverage from an oracle testing only itself. No future numerical
modules or geometric Fourier implementation are supplied in Task 2.

## Task 3: independent geometry and Fourier APIs

Task 3 now adds real new-API acceptance tests without changing the historical
legacy tests or source. No waveguide, Green function, or coupling matrix is
implemented here. The package root exports are unchanged; import the two
small modules directly.

### Coordinates and public entry points

- `geometry.CircleHole(radius_um, period_um=(1,1), center_um=None,
  eps_background=1, eps_hole=0)` provides broadcast evaluation, exact
  `fill_fraction`, and `mean_epsilon`. The default center is the cell midpoint.
- `geometry.RightTriangleHole(legs_um, ...)` has independent positive x/y
  legs. `center_um` is its **centroid**; the right-angle vertex is
  `center - legs/3`. Equal legs describe the literature's isosceles case;
  unequal legs are not mislabeled as isosceles. Rotation is not supported.
- Coordinates and periods are in um; the cell is `[0,ax] x [0,ay]`.
  Hole extents must fit entirely inside the cell: clipping/wrapping is never
  silently used to change the area. Calling geometry outside the cell does
  not wrap it; periodic wrapping, when needed, is explicit caller work.
- `geometry.sample_permittivity(callable, period_um, shape=(Ny,Nx))`
  evaluates `epsilon(x_um,y_um)` with NumPy broadcasting at midpoint samples
  `x=(j+1/2)*ax/Nx`, `y=(i+1/2)*ay/Ny`.
  `SampledPermittivity(values, period_um=(1,1))` snapshots an existing finite
  complex grid with immutable storage. Rows are **y**, columns are **x**.
  Values are point samples, **not pixel averages**. No pixel-sinc correction
  or continuous-boundary interpretation is implied.
- All Fourier functions accept signed integer `orders` of shape `(...,2)`,
  with last axis `(m,n)`, and return complex arrays of shape `(...)`.
  The coefficient is the cell average of
  `epsilon*exp(+2*pi*i*(m*x/ax+n*y/ay))`; reconstruction has the negative sign.
- `fourier.shape_fourier(hole, orders, quadrature_order=64)` uses the exact
  circle Bessel form and a separate zero-order limit plus background.
  Triangles use the reference integral below, not an unverified closed form.
- `fourier.integrate_hole_fourier(hole, orders, quadrature_order=64)` uses
  polar quadrature for circles and the smooth Duffy map for triangles,
  including their Jacobians, integrating only the inclusion contrast.
- `fourier.integrate_fourier(epsilon, period_um, orders,
  quadrature_order=64)` is a tensor Gauss-Legendre reference for general
  smooth broadcast-compatible complex functions. It rejects incompatible
  callable output shapes. Fixed quadrature order is **not an adaptive error
  tolerance**: establish convergence by increasing it. Discontinuous arbitrary
  masks need domain splitting or independent resolution checks.
- `fourier.fft_fourier(sampled, orders)` computes one positive-sign `ifft2`
  batch, indexes `[n % Ny,m % Nx]`, and applies the midpoint phase
  `exp(i*pi*(m/Nx+n/Ny))`. There is no `_xi`, global cache, or first-call
  initialization dependency. Duplicate requests and out-of-band orders raise.
  Canonical indices are `-(N//2)` through `(N-1)//2`; even grids accept only
  the **negative** Nyquist representative. The grid is never silently enlarged.
  Complex materials do not obey an assumed conjugate symmetry.

### TDD and actual verification

1. Tests were written before either module. The first targeted run failed
   with **2 collection errors** (`cwt3d.geometry` absent): scaffolding RED,
   not a demonstrated numerical defect.
2. The initial implementation exposed two overly optimistic test assumptions:
   triangle harmonic `(13,-11)` needs more than 24 Gauss nodes per axis for
   near-machine accuracy, and a discontinuous triangle on a 901x1100 midpoint
   grid has about `1.14e-3` error, not `<8e-4`. Tests now compare 48/64 Gauss
   rules and multiple midpoint resolutions rather than claiming a single-grid
   high precision. No geometry or coefficient was altered to force these checks.
3. A further behavior-first test for incompatible callable output genuinely
   returned **1 failed** (`DID NOT RAISE ValueError`) against the new integrator.
   Explicit broadcasting to the intended quadrature grid fixed the silent
   extra-axis integration; that regression is now GREEN.
4. Full repository run: `.venv/Scripts/python.exe -m pytest -q` ->
   **81 passed** (34 previous + 47 new). `git diff --check` passed.
   A fresh-process probe loaded the new APIs without any `model`, `calculator`,
   `mph`, or `numba` modules. Numba/COMSOL were neither installed nor required.

The independently executed probe at
`.validation-scratch/task3_metrics.py` returned these absolute errors:

| Check | Maximum absolute error |
|---|---:|
| Circle Bessel vs 80-node polar quadrature, complex contrast | 1.78e-16 |
| Triangle 8 vs 64 Gauss nodes, including `(13,-11)` | 2.51e-3 |
| Triangle 24 vs 64 nodes | 2.47e-5 |
| Triangle 48 vs 64 nodes | 1.92e-15 |
| Triangle 80 vs 64 nodes | 5.12e-16 |
| Smooth asymmetric complex function FFT, `(Ny,Nx)=(35,48)` | 2.78e-16 |
| Same smooth function, direct 64-node integration | 4.44e-16 |

Discontinuous geometry midpoint-grid errors (including the zero order):

| Geometry | `(Ny,Nx)` | Maximum absolute error |
|---|---|---:|
| Circle | `(41,54)` | 3.99e-4 |
| Circle | `(181,246)` | 2.38e-4 |
| Circle | `(701,950)` | 1.36e-5 |
| Triangle | `(451,550)` | 1.19e-3 |
| Triangle | `(901,1100)` | 1.14e-3 |
| Triangle | `(1801,2200)` | 4.31e-4 |

Boundary convergence is not generally monotonic at each resolution. Tests
also cover all canonical orders of a random complex 7x10 grid against an
explicit independent discrete sum, sign/axis ordering, Nyquist center phase,
real-only conjugate symmetry, continuous translations, reconstruction,
first/repeated calls and changing grid data. Circle/triangle average epsilon
and exact fill fractions have dedicated tests, as do invalid geometry,
materials, grids, quadrature orders, harmonic indices and alias requests.

Remaining limits: fixed-order reference quadrature has no certified error
estimate; arbitrarily high harmonics require convergence studies. FFT aliases
unresolved content in sampled data even when requested orders are in-band.
No continuous high-order accuracy or speedup benchmark is claimed. Analytic
circle evaluation and a batched FFT are the only performance-oriented paths
in this task. Legacy DFT remains intentionally unfixed; its new independent
replacement is what these acceptance tests verify.

### Task 3 quality fix: sampled-array metadata isolation

The original `values` getter returned the internal ndarray. Immutable bytes
prevented element writes but did not protect ndarray `shape`, `dtype`, or
`strides`: changing public metadata silently changed subsequent FFT results.

- Behavior-first RED:
  `.venv/Scripts/python.exe -m pytest tests/test_fourier.py -q -k sampled_metadata`
  returned **3 failed, 23 deselected** before the implementation changed.
  All failures were exact FFT equality assertions after metadata mutation,
  not collection/setup failures. Reshaping `(3,4)` to `(2,6)` changed the
  `(1,0)` coefficient from `-0.7071067811865476j` to approximately `-1j`.
- The minimal fix stores immutable bytes and a private shape tuple, and
  reconstructs a fresh ndarray header on each `values` access with
  `np.ndarray(..., buffer=bytes)`. Its public `.base` is the immutable bytes,
  not an internal ndarray. No per-access data copy or mutable backing is used;
  element assignment and `setflags(write=True)` remain rejected.
- The three regressions exercise shape/dtype/strides mutations and traverse
  every ndarray in the public `.base` chain, so merely returning a view of
  the old internal array would not suffice. They verify exact FFT equality,
  unchanged input data/metadata, retained getter results, fresh getter results,
  and a separate sampled snapshot created from the exposed values.
- GREEN: the same targeted command returned **3 passed, 23 deselected**.
  `.venv/Scripts/python.exe -m pytest -q` returned **84 passed, 3 warnings**.
  The warnings are NumPy deprecations for the deliberately exercised metadata
  setters. `git diff --check` passed (only a Git LF/CRLF conversion notice).

This fix does not expand numerical-scale guarantees: extreme lengths near
`1e-160` or `1e160` um are not certified against underflow/overflow. No legacy
source or Fourier implementation was changed for this quality fix.

## Task 4: stable open-boundary multilayer TE average waveguide

`cwt3d.waveguide.TEWaveguide(eps, thickness_um, *, k0=None,
wavelength_um=None, period_um=None)` solves the actual piecewise-constant
scalar TE equation, `Theta''+(k0^2*epsilon-beta^2)*Theta=0`. Epsilon is
**relative permittivity**, not refractive index. The ordered epsilon list is
`[lower halfspace, internal layers..., upper halfspace]`; thicknesses contain
only the internal layers. The first interface is z=0, with increasing z toward
the upper halfspace. Lengths are um; beta and k0 are um^-1. Exactly one of k0
and vacuum wavelength is specified, related by `k0=2*pi/wavelength_um`.
Optional period_um is fixed design metadata, never replaced by `2*pi/beta`.
There are no constructor processes, writes, COMSOL imports or legacy imports.

### Small reusable API and conventions

- `solve_modes()` returns all numerically resolved strictly bound modes in
  decreasing positive beta order. `solve_mode(index=0)` selects the fundamental
  by default; no modes returns an empty list or raises `NoGuidedModeError`,
  respectively. Invalid or unavailable indices raise explicitly.
- A mode exposes `beta`, complex `theta(z_um, amplitude=1)` (alias `Theta0`),
  `derivative(z_um, amplitude=1)` and `eps_profile(z_um)`. Coordinates and
  complex amplitudes broadcast, with scalar input/output supported. Complex
  amplitudes are not silently converted to real. At an epsilon discontinuity
  the profile uses the layer to its right; field and derivative are continuous.
- Default fields satisfy `integral_{-inf}^{inf}|Theta|^2 dz=1`; multiplying by
  amplitude changes the integral to `|amplitude|^2`. Both halfspace tails are
  included analytically as `|Theta(interface)|^2/(2*gamma)`; no finite box or
  field-minimum cutoff approximates a halfspace. Finite-layer endpoint-basis
  mass integrals have analytic forms, with smooth 12-point Gauss evaluation
  for phase <0.1 to avoid subtractive cancellation.
- Only finite **positive real lossless epsilon** is supported; complex-typed
  inputs with exactly zero imaginary part are accepted. Actual complex epsilon,
  including small loss, is explicitly rejected. With `exp(+i*omega*t)` and
  `exp(-i*beta*x)`, a passive extension must use Im(epsilon)<0 and forward
  attenuation Im(beta)<0. This module does not claim complex-material device
  support or silently reuse a legacy `n+i*k` convention.
- Equal adjacent epsilon is supported and partition-invariant. Zero, negative
  or infinite internal thicknesses are rejected; halfspace thicknesses must
  never be inserted into the finite-layer list. The kz=0 internal-layer limit
  is a linear basis with the exact endpoint stiffness limit, not NaNs.

### Stability and root certification

The solver uses exact element Dirichlet-to-Neumann (dynamic stiffness) maps
in dimensionless x=k0*z and beta^2/k0^2. An oscillatory material is subdivided
so its largest possible phase is <=pi/3 throughout the bound-mode interval.
Consequently no element sine denominator can have a pole. Evanescent maps and
field interpolation contain only decaying exponentials (`expm1` near zero),
never `exp(+gamma*d)`. Very thick barriers need no spatial subdivision.

The assembled symmetric tridiagonal stiffness has strictly increasing ordered
eigenvalues as beta^2 increases: its derivative is the positive element mass
form plus positive halfspace Robin derivatives. Counting its negative
eigenvalues at the cladding threshold determines how many root brackets are
needed. Each ordered eigenvalue is independently bracketed and solved; this
is not a determinant sign-change scan. Every returned root is then checked
against the **original endpoint derivatives**, including both halfspace Robin
conditions and every element interface, with normalized residual <=1e-8.
Optimizer success alone is insufficient. There are no finite-box modes.

Numerical limits are explicit: modes within 1e-12 in effective-index squared
of the cladding threshold are not certified; an absolute 1e-12 stiffness
inertia tolerance excludes numerically marginal threshold states. A pair of
roots closer than 1e-11*k0 in beta raises an unresolved-degeneracy error rather
than claiming distinct accurately resolved modes. A propagating layer needing
more than 20,000 elements is rejected. Extreme floating-point length/material
ranges are not certified. Nonfinite derived k0 is rejected explicitly.
No absorption/gain, negative-epsilon/plasmonic, leaky/radiating, TM, anisotropic
or periodic-in-z eigenproblem is implemented in this task.

### Actual TDD and verification evidence

1. `tests/test_waveguide.py` was created before the implementation. Its first
   run returned **1 collection error**, `ModuleNotFoundError: cwt3d.waveguide`:
   scaffolding RED, not a numerical failure. Adding the solver gave **16 passed**.
2. Additional behavior-first coverage exposed a real input-validation defect:
   wavelength `1e-320` produced an `OverflowError` while building the mesh rather
   than the promised `ValueError`. The targeted run was **1 failed, 17 passed**.
   Rejecting unrepresentable derived k0 fixed that regression.
3. Final Task 4 coverage includes **20 passing tests**: six independent analytic
   symmetric-slab even/odd roots and parities, fundamental/index ordering,
   asymmetric multilayer roots checked by independent `solve_ivp` integration
   of the original ODE, finite-difference source-free ODE residual, interface
   field/derivative continuity, independent quadrature plus analytic tails,
   complex amplitude/coordinate broadcasting, scale covariance, equivalent
   material partitions, zero-kz limit, invalid inputs, and no-guided-mode cases.
4. Thick-layer tests enable `over/invalid/divide='raise'`; neither the solver
   nor tests suppress floating-point overflow. A 10,000-um evanescent layer
   appended to a slab recovers its unchanged root, and a 10,000-um **internal
   barrier between unequal cores** recovers the union of isolated-core spectra.
   Underflow of exponentially negligible coupling is allowed; unresolved exact
   or near degeneracy remains the explicit limitation above.

The independent scratch probe `.validation-scratch/task4_metrics.py`
returned:

| Check | Observed result |
|---|---:|
| Symmetric slab modes | 6 |
| Maximum beta error vs independent even/odd dispersion roots | 3.55e-15 um^-1 |
| Asymmetric multilayer modes | 3 |
| Independent normalization absolute error | 0.0 at float64 rounding |
| Maximum original endpoint derivative normalized residual | 7.03e-16 |
| Field difference across interfaces, z +/- 1e-9 um | 1.59e-8 |
| Derivative difference across interfaces, z +/- 1e-9 um | 1.55e-7 |
| Thick-layer gamma*d | 1.057e5 |
| Thick-layer beta error vs unextended slab | 1.78e-15 um^-1 |
| Thick-layer original endpoint residual | 5.01e-16 |
| Forbidden imported modules (mph, numba, model, calculator) | none |

The finite-offset interface differences include the physical variation over
2e-9 um, not just the exact endpoint matching residual. No COMSOL process was
started. Historical source and tests are unchanged. Final full suite:
`.venv/Scripts/python.exe -m pytest -q` returned **104 passed, 3 warnings**
(the existing deliberate NumPy metadata-setter warnings). Independent
specification and code-quality reviews remain a separate gate; this
implementation does not self-certify as reviewed.

### Task 4 quality R1: immutable public configuration and solved state

The review found that assigning public configuration after construction mixed
new configuration with the old derived mesh. Before this fix, an actual probe
using `TEWaveguide([1,9,1], [.6], k0=4)` followed by `guide.k0=5` returned
beta `14.13153780436745`, while a fresh k0=5 guide returned
`14.392754595046537`. Sampling the previously solved mode at
`[-.1,.1,.3,.5,.7]` changed its field by up to `0.08521694160012383`
although its stored beta did not change.

- **Strategy:** keep the existing read API, but store configuration privately
  and expose getter-only properties for `k0`, `eps`, `thickness_um`,
  `interfaces_um`, and `period_um`. Assignment and deletion raise
  `AttributeError`. Sequence getters return tuples of scalar values, not
  mutable internal arrays. To change configuration, construct a new guide.
  `TEMode.guide`, `beta`, and `boundary_residual` use the same read-only
  property approach, preventing replacement of the guide or inconsistent
  solved-state metadata. The mode may retain its guide reference because
  that guide's public configuration cannot change. Private-state tampering
  is outside the API contract. No numerical algorithm or supported physics
  was changed.
- **Behavior-first RED:** after adding tests, before changing the solver,
  `.venv/Scripts/python.exe -m pytest tests/test_waveguide.py -q -k
  'read_only or mutable_aliases' --tb=short` returned **16 failed, 1 passed,
  20 deselected**. All 16 failures were `DID NOT RAISE AttributeError` for
  public assignment/deletion, not setup or collection errors. The alias
  isolation check already passed and remains a regression guard.
- **GREEN:** the same selection returned **17 passed, 20 deselected**.
  It checks all five guide properties and all three mode properties for
  assignment and deletion rejection; unchanged old-mode beta, residual,
  sampled field and derivative; exact equality of repeated solves against a
  newly constructed guide; independent new configurations; caller input-array
  mutation; immutable getter sequences; and mutation of returned evaluation
  arrays without internal-state changes.
- **Full verification:** `.venv/Scripts/python.exe -m pytest -q` returned
  **121 passed, 3 warnings** (only the existing NumPy metadata-setter
  deprecations). `git diff --check` passed, with the existing LF/CRLF notice.
  Changes are limited to `src/cwt3d/waveguide.py`, `tests/test_waveguide.py`,
  and this document. Legacy source is untouched; no commit or push was made.
  Independent quality re-review is still required.
