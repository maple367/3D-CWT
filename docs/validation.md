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

## Task 5: homogeneous outgoing passive Green kernel

`cwt3d.greens.HomogeneousGreen(k0, epsilon, k_parallel=0)` is an independent
infinite-homogeneous scalar kernel. Lengths are um, wavenumbers are um^-1,
epsilon is relative permittivity, and G has units um. Public configuration
and the derived scalar properties `gamma` and `kz` are getter-only; no internal
array metadata is exposed. Zero-dimensional input arrays are converted to
independent Python scalars. The package root and historical source are unchanged.

### Convention, API, and explicit limits

For `exp(+i*omega*t)`, passive epsilon requires `Im(epsilon)<=0`; active
materials are explicitly rejected, never conjugated or silently converted.
The operator and outgoing branch are

```text
[d_z^2 + k0^2*epsilon - k_parallel^2] G = -delta(z-z')
gamma = sqrt(k_parallel^2 - k0^2*epsilon), Re(gamma)>=0, Im(gamma)>=0
kz = -i*gamma, Re(kz)>=0, Im(kz)<=0
G(z,z') = exp(-gamma*abs(z-z'))/(2*gamma)
```

Thus lossless zero-parallel radiation is exactly
`-i/(2*k0*sqrt(epsilon))*exp(-i*k0*sqrt(epsilon)*abs(z-z'))` for positive
epsilon. High-parallel orders are genuinely decaying, not oscillatory. Signed
zero on the radicand's real axis is normalized to the outgoing branch.

- Call `kernel(z,z_prime)` with finite real scalar/array coordinates. Standard
  broadcasting, including independent grids `z[:,None]` and `z_prime[None,:]`,
  produces complex output; there is no callback vectorization or shape cache.
- `kernel.derivative(z,z_prime,side=None)` is the receiver derivative. At an
  exact source coincidence the default raises; `side='left'` gives +1/2 and
  `side='right'` gives -1/2, so the jump is -1. Elsewhere `side` has no effect.
  G itself is continuous and defined at the source.
- Exact float64 zero radicand raises `CutoffError` (a `ValueError`). Its message
  explicitly covers unresolved near-cutoff cancellation as well as exact
  cutoff. A nonzero gamma is not tolerance-clamped: tests retain gamma=1e-14
  and the large amplitude from the immediate float64 neighbor of cutoff.
- Nonfinite, boolean, nonnumeric, wrongly dimensional configuration, active
  epsilon, nonpositive k0, negative k_parallel and complex coordinates raise
  `ValueError`. Extreme squared wavenumbers or propagation intermediates
  outside the resolved float64 range are rejected. Arbitrary extreme dynamic
  range and sub-float64 cutoff separation are not certified.
- Only decaying spatial exponentials occur. With `over/invalid/divide='raise'`,
  both fields and derivatives remain finite (zero after harmless underflow)
  at distances of 1e6 um in the tested evanescent passive materials.
- No interfaces, layered Green functions, C matrix, harmonic-order API,
  anisotropic/TM physics or measured performance acceleration are implemented.

### Actual RED/GREEN and physical-regression evidence

1. `tests/test_greens_homogeneous.py` was written before `greens.py`.
   `.venv/Scripts/python.exe -m pytest tests/test_greens_homogeneous.py -q`
   first returned **1 collection error**, `ModuleNotFoundError: cwt3d.greens`.
   This is scaffolding RED, not a numerical failure of a new implementation.
   Adding the implementation made the suite GREEN.
2. The permanent new-API test repeats the Task 2 fixture: k0=1, epsilon=1,
   beta0=3, order=(2,0), hence k_parallel=6 and gamma=sqrt(35). It actually
   calls `HomogeneousGreen`, checks the independent real exponential, both
   source-free sides, spatial decay and the derivative jump. It does not
   mistake the historical oracle-only tests for a new implementation fix.
3. A separate same-gate comparison was also actually executed:
   `.venv/Scripts/python.exe -m pytest .validation-scratch/task5_physics_probe.py -q --tb=short`
   returned **1 failed, 1 passed**: actual isolated legacy methods failed
   the physical residual threshold, and the real new API passed. This
   comparison was run after implementation, not presented as a pre-code
   behavioral TDD failure. The first probe attempted arrays on legacy's scalar
   method; it was corrected to a scalar loop before obtaining physical metrics.
   No legacy, mph, calculator or numba module was imported.
4. Running `.venv/Scripts/python.exe .validation-scratch/task5_physics_probe.py`
   produced the following independently evaluated metrics. ODE h=1e-4 um,
   source=.31 um, offsets +/- .2 and +/- .7 um; source slopes use h=1e-6 um.

| Check | Observed result |
|---|---:|
| Actual isolated legacy maximum relative ODE residual | 1.9999999709031708 |
| New API maximum relative ODE residual | 2.9342765235256057e-8 |
| Exact one-sided receiver derivative jump error versus -1 | 0.0 |
| Independent G finite-difference source-slope jump error | 2.9580217155711352e-6 |
| G source-side continuity difference at +/-h | 0.0 |
| Forbidden imported module roots | none |

Final commands:

```text
.venv/Scripts/python.exe -m pytest tests/test_greens_homogeneous.py -q
.venv/Scripts/python.exe -m pytest -q
git diff --check
```

Results: **54 passed** in the new module and **175 passed, 3 existing warnings**
in the full suite; diff whitespace validation passed. The new result-metadata
mutation test locally filters only the expected NumPy shape-setter deprecation,
not floating-point warnings. Tests additionally cover passive complex branch
oracles and outgoing flux sign, lossless radiation, negative epsilon,
independent finite-difference derivatives/ODE/source jumps, symmetry,
translation and length scaling, equivalent parallel wavevectors, broadcasting,
empty arrays, invalid inputs, read-only configuration and input/result mutation
isolation. Only the two new files and this validation appendix are changed.
No commit/push was made. Specification and quality double review are still
required; layered Task 6 is intentionally not started.

## Task 6: stable full layered scalar TE Green kernel

### Public integration API and conventions

```python
from cwt3d.greens import LayeredGreen, PoleError, CutoffError
kernel = LayeredGreen(k0, eps, thickness_um, k_parallel=0, origin_um=0)
# Or snapshot a TEWaveguide-compatible structure, without solving/importing a mode:
kernel = LayeredGreen.from_structure(guide, k_parallel)
values = kernel(z[:, None], z_prime[None, :])
receiver_derivatives = kernel.derivative(z[:, None], z_prime[None, :], side='right')
```

`eps[0]` and `eps[-1]` are semi-infinite halfspaces; only the finite internal
layers have positive `thickness_um`. Two eps entries and an empty thickness
sequence describe one interface. The first interface is `origin_um` (default
zero). `k0`/`k_parallel` are um^-1, coordinates/thicknesses/G are um, epsilon
is relative permittivity. The existing `exp(+i*omega*t)`, passive
`Im(epsilon)<=0`, and `[d_z^2+k0^2 epsilon-k_parallel^2]G=-delta` convention
is unchanged. Gain is explicitly rejected. Passive complex, real positive,
and negative real epsilon are accepted.

Sources and receivers can independently occupy any layer, either halfspace,
or an interface. `eps_profile` uses the **right** material at an interface;
G is continuous, and its receiver derivative is continuous there unless it
is also the source. At exact source coincidence, `derivative` requires
`side='left'` or `'right'`; their difference is -1. This convention also
applies to interface sources. Away from the source, side has no effect.
Lossy reciprocity is transpose symmetry `G(z,s)=G(s,z)`, not conjugation.

Public `eps`, `thickness_um`, `interfaces_um` are immutable scalar tuples;
`k0`, `k_parallel`, `origin_um` are getter-only scalars. Caller arrays are
snapshotted. Evaluation/profile outputs own independent mutable metadata and
data; no source, result or broadcast-shape cache is retained. Empty batches,
scalar calls and general NumPy broadcasting are supported. `from_structure`
snapshots only k0/eps/thickness and optional origin, with no mode dependency.

### Stable algorithm and source normalization

For each element of width d, gamma is on the same outgoing branch as Task 5.
The exact outward-derivative endpoint map has diagonal `gamma*coth(gamma*d)`
and off-diagonal `-gamma*csch(gamma*d)`. These are evaluated using only
`exp(-gamma*d)` and `expm1(-2*gamma*d)`, never growing T-matrix products.
Small arguments use analytic series. In particular **internal gamma=0 is
supported**, with the exact linear-element map `[[1,-1],[-1,1]]/d`.

Weakly attenuating oscillatory layers are subdivided to imaginary phase
<=pi/3 so an element Dirichlet resonance cannot be mistaken for a physical
pole. A layer with `Re(gamma*d)>=1` needs no subdivision, including very thick
lossy barriers. Halfspaces contribute exact Robin terms `+gamma` on the two
outer endpoints; they are never truncated to a finite absorbing box.

The assembled matrix K is complex symmetric. An interior source loads the
two endpoints by its exact homogeneous endpoint shape functions `N(s)`.
Receiver evaluation is `N(z)^T K^-1 N(s)` plus the element's zero-endpoint
Dirichlet Green kernel if both coordinates share that element. An interface
source is a unit nodal load. Exterior sources have exponentially weighted
boundary loads and the corresponding halfline Dirichlet Green contribution.
The local Green term supplies the derivative kink; integration of the
operator enforces `G'(s+)-G'(s-)=-1`, not a material-weighted jump. All
interpolation/local Green factors are also evaluated with decaying
exponentials and small-argument linear limits.

This fixed mesh is important: an initial implementation inserting source and
receiver nodes developed spurious ill-conditioning when coordinates were
near existing interfaces, and inaccurate finite-difference source slopes.
The final source/receiver shape-function construction avoids those tiny
artificial elements without snapping physical coordinates.

### Explicit limits and singularity handling

- This is a scalar TE reference kernel, not TM, anisotropy, C/SGM or a mode
  solver. It solves a dense system and computes singular values **per pair**;
  broadcasting is provided for later quadrature correctness, not acceleration.
  Mesh construction has a 512-node budget; over-budget oscillatory structures
  raise `ValueError` rather than running without element-pole protection.
- Exact **exterior** cutoff raises `CutoffError`: a threshold radiation
  prescription is not implemented. This is a stated boundary limitation, not
  a claim that an internal zero-kz layer makes the full problem singular.
- `PoleError` diagnoses `s_min(K)<=32*eps_machine*s_max(K)` or a failed singular
  solve. This includes true bound poles and numerically unresolved near poles;
  it is a numerical diagnostic, not a proof of an exact mathematical root.
  No imaginary regularizer, denominator clamp or finite pole value is used.
  Resolved nonzero near poles retain large physical enhancement. Arbitrary
  ill-conditioning/extreme scales and sub-roundoff pole separation are not
  certified; users requiring those ranges need higher precision.
- Harmless spatial underflow is allowed locally, even under a caller's strict
  underflow policy; over/invalid/divide are not swallowed and global `np.seterr`
  is never changed. Unsupported nonfinite derived phases/mesh resolution raise.
  HomogeneousGreen behavior is unchanged, including its own underflow policy.

### Actual TDD and independent physical validation

1. The test module was written first. Its initial run returned **1 collection
   error**, missing `LayeredGreen`: scaffolding RED, not a physics failure.
2. The first numerical implementation returned **4 failed, 19 passed**:
   coordinate-insertion conditioning triggered false pole diagnoses and the
   source-slope check failed. Replacing coordinate insertion by exact endpoint
   source loads plus the local Dirichlet kernel gave **23 passed**.
3. An additional behavior-first strict intermediate-underflow regression
   returned **1 failed, 22 deselected** (`FloatingPointError` in a tail product).
   Scalar propagation products now allow harmless underflow without modifying
   NumPy policy. Additional randomized/material/metadata/budget tests pass.
4. The independent oracle in `tests/test_greens_layered.py` solves a separate
   small interface-amplitude linear system in `exp(+/-gamma*z)`, enforcing all
   field/derivative interface equations and source jump explicitly. It uses
   no production DtN, interpolation, mesh or Green helpers. It is restricted
   to moderate widths, not used as a thick-layer oracle. Single-interface
   Fresnel formulas and homogeneous reduction are additional analytic oracles.
5. Permanent coverage includes both source directions at a single interface,
   lossless and lossy multi-interface systems, sources at interfaces/in both
   halfspaces, passive randomized negative/positive materials, source-free
   ODE on both sides, analytic receiver derivatives, independent G-based
   slope jump, transpose reciprocity, outgoing flux, halfspace decay,
   translation/length scaling, all broadcasting and configuration isolation.
   A whole-layer Dirichlet resonance with gamma*d=i*pi is finite after proper
   subdivision. Internal exact gamma=0 is checked against its linear solution
   and a nonzero-gamma limit.
6. The real Task 4 guide `[1,9,1]`, d=.6 um, k0=4 gives fundamental beta
   `11.30523024349396` um^-1, at which the Green kernel raises `PoleError`.
   Offsets 1e-4, 1e-5 and 1e-6 are finite, agree with the independent interface
   system, and show the expected approximately tenfold enhancement per decade.
7. A 10,000-um evanescent barrier has gamma*d=`26457.51311064591` and passes
   with over/invalid/divide/**under** all set to raise. Its left-source
   reflection reduces to the isolated interface Fresnel value, central
   coincident values reduce to the homogeneous kernel, and cross-barrier
   transmission underflows to zero. A 10,000-um passive **complex** homogeneous
   partition also reduces correctly and transmits zero at remote points.

Executed scratch probe `.validation-scratch/task6_metrics.py`:

| Check | Observed error/result |
|---|---:|
| Multilayer G vs independent interface system, max absolute error | 7.54e-16 um |
| Receiver derivative vs independent interface system, max absolute error | 1.30e-15 |
| Lossy transpose reciprocity, max absolute error | 2.37e-16 um |
| Homogeneous reduction across artificial interfaces, max absolute error | 2.02e-16 um |
| Source-free relative ODE residual, centered h=1e-4 um | 1.38e-8 |
| One-sided derivative jump error against -1 | 8.33e-17 |
| Independent G slope-jump error, h=1e-7 um | 1.56e-7 |
| Interface G difference at +/-1e-9 um | 9.53e-10 um |
| Interface derivative difference at +/-1e-9 um | 2.50e-9 |
| Near-pole relative oracle error, beta offset 1e-4 | 9.05e-13 |
| Near-pole relative oracle error, beta offset 1e-5 | 2.43e-11 |
| Near-pole relative oracle error, beta offset 1e-6 | 3.00e-10 |
| Near-pole magnitudes for those offsets | 1031.36, 10313.64, 103136.47 um |
| Forbidden imported module roots (mph/model/calculator/numba) | none |

Finite-offset interface differences include physical variation, not solely
algebraic matching error. The near-pole magnitudes are physically divergent
values, not accuracy guarantees arbitrarily close to a pole.

Verification commands: `pytest tests/test_greens_layered.py
 tests/test_greens_homogeneous.py -q` -> **82 passed** (28 new layered tests,
54 unchanged homogeneous tests); full `pytest -q` -> **203 passed, 3 existing
NumPy metadata-setter warnings**. `git diff --check` passed. Changes are limited
to `src/cwt3d/greens.py`, new `tests/test_greens_layered.py`, and this appendix.
No legacy source, waveguide, C/SGM, package exports, commit or push is changed.
Independent specification and quality double review remain a separate gate.

## Task 7: physical multilayer reference coupling and infinite four modes

New entry points (the package root and all historical source stay unchanged):

```python
from cwt3d.coupling import build_coupling
from cwt3d.infinite import solve_infinite
# mode is a normalized TEMode; a is an EXPLICIT scalar square period in um.
# pc_layers maps finite internal EPSILON indices (1 .. len(eps)-2) to geometry.
c = build_coupling(mode, a, pc_layers, cutoff=1, quadrature_order=12)
r = solve_infinite(c)
```

The order is V=(Rx,Sx,Ry,Sy), harmonics=(+x,-x,+y,-y), electric
components=(y,y,x,x). Every coupling matrix has units um^-1. `C1D`
contains only the four counterpropagating kappa entries, `Crad` the two
same-polarization zeta blocks, and `C2D` is chi with its full external
`-k0**2/(2*beta0)` factor. `C2D_green` and `C2D_local` expose the nonlocal
and negative-nu terms. The result also returns `mu`, `nu`, `varsigma`,
`aux_orders`, `xi`, `xi_orders`, `I`, `J`, `K`, layer IDs and radial IDs.
`mu/nu` axes are (aux, receiver layer, source basic wave); varsigma adds
an x/y component axis before the source wave. These dimensionless fields
are integrated over each RECEIVER layer before its Fourier coefficient
is applied; they are not averaged-PC coefficients. Sum the receiver-layer
axis to recover thesis (3.27)-(3.29). No term substitutes xi(z) for xi(z').

`I_l=integral_l |Theta|^2`, `J_l=I_l/epsilon_l`, and
`K_ll'=integral_l integral_l' Theta*(z) G(z,s) Theta(s)` factor each layer's
constant Fourier coefficients exactly. Integration is partitioned at all
material interfaces. Each same-layer double integral uses TWO Duffy
triangles with Jacobian L^2*u, so the Green derivative cusp is a boundary,
not an unresolved tensor-rule diagonal. Cross-layer rectangles use Gauss
products. Kernels/overlaps are reused by m^2+n^2, not per matrix entry.
The auxiliary square is |m|,|n|<=D excluding zero and all four basics.

Input scope: positive lossless epsilon, scalar nonmagnetic TE, square period,
CircleHole/RightTriangleHole/SampledPermittivity in finite internal layers.
Each geometry period and mean epsilon must match the guide; imaginary
material epsilon is rejected, never discarded. Other layers have xi=0;
an empty PC mapping and zero contrast give exactly zero matrices. Sampled
geometry obeys the existing canonical FFT band limits and sampling caveats.
`beta0=2*pi/a` must match mode.beta at default relative tolerance 1e-8.
A caller may explicitly relax `beta_rtol`; `relative_detuning` records the
mismatch, but no a/k0 is changed and no detuning diagonal is invented.
Propagating internal auxiliaries are allowed; `all_aux_evanescent` is the
conservative condition that EVERY auxiliary is evanescent in EVERY layer.
Only under that condition is C2D Hermiticity asserted in these tests.
Inherited Green pole/cutoff/budget limits still apply. Order and cutoff are
recorded, not claimed to be certified quadrature-error estimates.

Results use frozen scalar/tuple metadata and independent ndarray headers
backed directly by immutable bytes. Element and write-flag mutation fail;
public ndarray metadata edits cannot alter subsequent reads or eigenpairs.
`solve_infinite` returns actual complex `np.linalg.eig(C)` eigenvalues and column
vectors, delta=Re(lambda), amplitude threshold/loss alpha=Im(lambda), power
loss=2*alpha, and original normalized residuals. Imaginary parts are never
absolutized. Ordering is ordinary unsorted eig output, NOT mode tracking.
No approximate group-index frequency conversion, Q, finite-size solver,
COMSOL run, paper-figure reproduction or speedup claim is included.

### Actual numerical gates

The new tests were written first: initial execution failed to collect because
`cwt3d.coupling` did not exist (scaffolding RED). The completed physical tests
pass. A separate scratch run modified only IN-MEMORY copies of the actual
new build function: dropping chi's outer prefactor failed the real geometry
scale gate, and reusing receiver xi at the source failed the independent
double-layer adaptive radiation integral. Neither mutation changed disk files.

Independent oracles include: single-layer kappa by adaptive integration;
single-layer chi by source-cusp-split nested scipy.quad; a homogeneous
constant-field analytic double integral; two unlike PC layers with average
eps 8 and 10, different radii and lateral positions; adaptive complex source
fields and outgoing power at both exterior interfaces. This last power test
checks `2 Im(v† Crad v) = k0^4/beta0 * sum(kz*|u_boundary|^2)` for both
polarizations, independently of the matrix's anti-Hermitian decomposition.

Executed scratch verification `.validation-scratch/task7-verify.py`:

| Check | Observed result |
|---|---:|
| Single-layer chi vs independent adaptive oracle, absolute error | 8.13e-20 um^-1 |
| Double-layer radiation vs independent adaptive oracle, relative error | 1.50e-15 |
| Independent outgoing power identity, relative error | 3.23e-15 |
| Maximum four-mode normalized residual | 3.01e-16 |
| C1D Hermitian max absolute difference | 0 |
| All-evanescent C2D Hermitian max absolute difference | 4.85e-19 um^-1 |
| Crad anti-Hermitian-part minimum eigenvalue | 2.09e-5 um^-1 |
| All-term inverse-length scaling, length factor 3.7, max absolute error | 2.02e-17 um^-1 |
| Quadrature Q=4 vs Q=12, matrix difference norm | 1.47e-5 um^-1 |
| Quadrature Q=8 vs Q=12, matrix difference norm | 7.58e-12 um^-1 |
| Cutoff D1->D2 / D2->D3 / D3->D4 difference norms | 6.48e-3 / 3.13e-3 / 5.42e-4 um^-1 |

These are small physical-fixture convergence observations, not a D=4
convergence certificate for arbitrary structures. D=10 benchmarking remains
Task 12. Tests additionally cover diagonal phase-similarity under global
translation, eigenvalue invariance, zero contrast, explicit detuning,
propagating internal auxiliaries, invalid means/materials/indices, and
immutable matrix metadata. The complete C is intentionally not Hermitian.

Verification: `pytest tests/test_coupling.py -q` -> **10 passed**;
full `pytest -q` -> **213 passed, 3 existing NumPy deprecation warnings**.
`git diff --check` passes. A fresh import probe finds no mph, numba, model
or calculator modules. Only the three new src modules, new coupling test,
and this appendix are changed. No commit/push was made. Task 7 awaits
independent specification and quality review.
