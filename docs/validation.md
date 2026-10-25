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

## Task 8: face-staggered rectangular layout and exact boundary elimination

Public API: `from cwt3d.finite import StaggeredGrid, WaveFields`.
`StaggeredGrid(Nx, Ny, Lx_um, Ly_um)` is immutable. Nx/Ny count cells;
`hx_um=Lx_um/Nx`, `hy_um=Ly_um/Ny`; lengths/coordinates are in um.
Rows index y and columns index x on `[0,Lx_um] x [0,Ly_um]`.

| Component | Full locations / shape | Eliminated incoming face | Retained free block |
|---|---|---|---|
| Rx | x-face, y-center / (Ny,Nx+1) | `[:,0]` | `[:,1:]` |
| Sx | x-face, y-center / (Ny,Nx+1) | `[:,-1]` | `[:,:-1]` |
| Ry | x-center, y-face / (Ny+1,Nx) | `[0,:]` | `[1:,:]` |
| Sy | x-center, y-face / (Ny+1,Nx) | `[-1,:]` | `[:-1,:]` |

The four free blocks each have shape `(Ny,Nx)`. Their C-order flattenings
are concatenated in `BASIC_WAVE_ORDER=(Rx,Sx,Ry,Sy)`, giving exactly
`n_dof=4*Nx*Ny`. `component_shape(name)` describes full fields;
`free_index(name,row,col)` maps full indices, returning -1 only for its
own incoming face and rejecting negative/out-of-range/noninteger indices.
For Nx=2, Ny=3, block offsets are 0,6,12,18; independent hand-written full
index tables in test 01 assert every free id occurs exactly once.
There is no shared four-component corner node: each wave has its own face
locations. A one-cell domain has four free outgoing amplitudes, not four
fully Dirichlet waves. No dummy values or 1e8 boundary penalties are used.

- `coordinates(name)` returns read-only full-face `(X_um,Y_um)` meshes;
  `coordinates()` returns `(Ny,Nx)` cell-center meshes.
- `unpack(vector)` snapshots a finite `(n_dof,)` numeric vector as four full
  complex128 arrays in `WaveFields`, restoring exact zero incoming entries.
- `pack(fields)` checks the exact full shapes and rejects **any** nonzero
  incoming amplitude (including 1e-300j). It returns a read-only free-vector
  snapshot; outgoing faces are retained. Real numeric inputs are accepted.
- `WaveFields(Rx,Sx,Ry,Sy)` independently snapshots finite 2-D numeric arrays.
  It has no mutable public config or metadata. All public array getters use
  new ndarray headers with direct immutable bytes bases, including `.base`
  traversal protection; input mutation and public shape/dtype/strides edits
  cannot pollute saved fields, a grid, or another result.
- `interpolate(fields)` geometrically averages adjacent faces into four
  `(Ny,Nx)` arrays in another `WaveFields`, for later receiver coupling and
  mass terms. Rx/Sx average left/right; Ry/Sy average bottom/top. Arbitrary
  finite full fields are allowed here; interpolation does not enforce BCs.

### Actual TDD and verification evidence

1. Tests preceded the package: targeted pytest returned **1 collection
   error**, `ModuleNotFoundError: cwt3d.finite`. This is scaffolding RED,
   not a demonstrated numerical failure of an old discretization.
2. GREEN: `.venv/Scripts/python.exe -m pytest tests/test_grid.py -q` ->
   **65 passed**. Tests cover rectangular Nx=2,Ny=3,Lx=8,Ly=9 coordinates,
   manual free-id tables, exact complex roundtrips, four incoming/outgoing
   faces, one-cell intersections, constants and BC-compatible linear fields,
   read-only snapshots including public metadata edits and retained siblings,
   invalid counts/lengths/derived underflow/shapes/indices/nonfinite fields.
   Only intentional metadata-mutation deprecations in the new test are
   locally filtered; existing tests/warnings are unchanged.
3. Full `.venv/Scripts/python.exe -m pytest -q` -> **278 passed, 3 existing
   NumPy warnings**. `git diff --check` passed. A fresh-process test executes
   grid operations without loading model/calculator/utils/mph/numba.
4. Independent probe `.validation-scratch/task8_metrics.py` returned:
   `n_dof=24`, vector roundtrip max absolute error **0.0**, incoming max
   amplitude **0.0**, complex linear center interpolation error **0.0**,
   and unique/complete free-index coverage **True**.

Only the new finite package, new grid tests and this appendix are changed.
Legacy code remains untouched. No commit/push. Independent specification
and quality reviews are pending in the parent workflow. This is layout and
boundary elimination only: no PDE A/B assembly (Task 9), finite-domain
eigenvalue solver (Task 10), or completed finite-domain eigenproblem.

## Task 9: direct sparse staggered generalized pencil

Public API: `from cwt3d.finite import assemble_pencil, FinitePencil`.
`assemble_pencil(grid, coupling)` accepts a finite numeric `(4,4)` matrix
in um^-1 or the actual `CouplingResult` (`.C`, not `.C_total`; units checked).
It returns `A v = lambda B v`, `lambda = delta + i alpha`, with
`A = C_lift B + i D`, using the thesis (4.21) signs
`D=(+d_x Rx,-d_x Sx,+d_y Ry,-d_y Sy)`. Rows are component-major cell
centers; columns retain Task 8's component-major free faces, C order.
Both matrices are square complex128 CSR with dimension `4*Nx*Ny`.
B is a neighbor-face average and is generally non-Hermitian, not SPD.

Local one-dimensional COO triplets are converted directly to canonical CSR;
no dense domain matrix, large Kronecker product, explicit inverse, linear
solve, eigenvalue solve, ghost value or penalty is used. Incoming faces are
omitted exactly. D is also exposed before multiplication by i. For generic
nonzero uniform coupling, `nnz(B)=nnz(D)=8*Nx*Ny-2*Nx-2*Ny` and
`nnz(A)=4*nnz(B)`; exact cancellations/zero coupling reduce A's nnz.
Index/allocation representability and finite inverse spacings are checked
before allocation, without imposing an arbitrary int32 grid-size cap.

The grid is immutable; C is saved in immutable bytes and returns fresh
ndarray metadata. Each public A/B/D getter returns an independent full CSR
copy (O(nnz) time/storage), including data, indices, indptr and shape.
Caller input changes, sparse-buffer edits and public resizing cannot alter
later reads. A future solver should fetch A/B only once.

### Independent gates and actual results

- RED before implementation: `tests/test_assembly.py` failed collection with
  `ModuleNotFoundError: cwt3d.finite.assembly` (scaffolding RED, not a
  numerical defect claim). GREEN: **34 new tests passed**.
- Hand-written 1x1 oracle checks `B=.5 I`,
  `D=diag(1/hx,1/hx,1/hy,1/hy)` and full complex A, including S signs.
- A separate slow dense oracle enumerates full faces without production
  free-index/templates, on 2x3, 1x7, 8x1 and rectangular grids, with complex
  and zero C. A second oracle uses real `grid.unpack` fields and direct
  averaging/differencing for random complex vectors.
- 2x3, Lx=7,Ly=11: **24 free unknowns**, A/B shape **(24,24)**,
  nnz(A/B/D)=**152/38/38**. Maximum matrix-oracle errors are **0/0/0**;
  action errors are **1.78e-15/0/5.55e-17**.
- For arbitrary complex v, `U=Bv`, the exact telescoping flux gate checks
  `2*hx*hy*Im[U^H*(A*v-C_lift*U)]` against outgoing edge intensity
  `hy*sum(|Rx_right|^2+|Sx_left|^2)` plus
  `hx*sum(|Ry_top|^2+|Sy_bottom|^2)`. This covers general complex C and
  Hermitian C, without incorrectly requiring A to be Hermitian. The 2x3
  probe gives **64.77031440887399 vs 64.77031440887403**,
  relative difference **6.58e-16**.
- BC-compatible linear fields match analytic center values/derivatives;
  the only globally constant BC-compatible envelope is zero. Cubic fields
  are compared to **continuous analytic center** values, not a discrete
  self-oracle. For `(Nx,Ny)=(8,16),(16,32),(32,64)`, relative errors in
  B/D/A are respectively `(4.7423e-3,9.0970e-4,4.1818e-3)`,
  `(1.1831e-3,2.2712e-4,1.0435e-3)`,
  `(2.9561e-4,5.6760e-5,2.6074e-4)`; measured orders lie in
  **[2.00049,2.00308]**.
- Allocation guards reject dense 2-D domain allocations and monkeypatch
  dense/sparse inverses, solvers, eigensolvers, sparse densification and
  Kronecker construction to fail. Three-grid storage probe:

| Nx x Ny | Free DoF | nnz A / B / D | CSR bytes A / B / D | traced assembly peak bytes |
|---|---:|---:|---:|---:|
| 8 x 11 | 352 | 2664 / 666 / 666 | 54692 / 14732 / 14732 | 238132 |
| 16 x 22 | 1408 | 10960 / 2740 / 2740 | 224836 / 60436 / 60436 | 929604 |
| 32 x 44 | 5632 | 44448 / 11112 / 11112 | 911492 / 244772 / 244772 | 3719236 |

These show linear storage; tracemalloc peaks are a local Python/NumPy
allocation probe, not total process RSS or a production-device benchmark.
Real zero and nonzero geometry-derived CouplingResult integration, sparse
snapshot mutation, exact cancellation, invalid/overflow rejection and a
clean subprocess without model/calculator/utils/mph/numba imports also pass.
Metrics script/output: `.validation-scratch/task9_metrics.py` and `.json`.
Full `.venv/Scripts/python.exe -m pytest -q`: **312 passed, 3 existing
NumPy metadata-mutation deprecation warnings**. `git diff --check` passed
(with only a Git LF/CRLF conversion notice). No legacy sources changed,
no commit/push. Independent specification/quality reviews remain pending in
the parent workflow. **Task 10 eigensolve is not implemented or claimed.**

## Task 10: pure-Python targeted non-Hermitian sparse eigensolve

### API and supported scope

```python
from cwt3d.finite import solve_generalized, solve_finite, EigenSolveError

# General square sparse complex pencil; B need not be Hermitian or SPD.
r = solve_generalized(A, B, sigma=0.02+0.3j, k=4, tol=1e-12,
                      residual_tol=1e-8, seed=21, ncv=20, maxiter=240)
# Actual assembled optical pencil, same options and free-face ordering:
r = solve_finite(pencil, sigma=0.02+0.3j, k=4, seed=21)
fields = pencil.grid.unpack(r.eigenvectors[:, 0])
centers = pencil.grid.interpolate(fields)
```

- Production accepts sparse numeric A/B only, copies each into complex128 CSR,
  validates equal square shape, finite coefficients and n>=3. No domain-sized
  dense matrix, dense fallback, inverse, `eigsh`, or `eigs(A, M=B)` is used.
- Apply `T(x)=LU.solve(B@x)`, factoring `(A-sigma*B).tocsc()` exactly once per
  call, then `eigs(T, k=k, which='LM')` **without sigma or M**. Back-transform
  `lambda=sigma+1/mu`. No cache, hidden shift perturbation or regularization.
- `k=None` means `min(6,n-2)` (the smallest four-DOF grid requests two).
  Explicit k must be an integer with `1<=k<n-1`. Complex ARPACK requires
  `k+2<=ncv<=n`, not merely `ncv>k`; default is `min(n,max(2*k+1,20))`.
  `maxiter=None` uses `10*n`. `tol>=0` is the ARPACK tolerance, with zero
  meaning machine precision; `residual_tol>0` is an independent gate.
- sigma must be finite, seed is a nonnegative integer or None (default 0),
  and v0 is a copied finite nonzero numeric vector of exact shape `(n,)`.
  Its real/imaginary components are scaled before normalization, accepting
  even huge/subnormal finite vectors without a global `seterr` change.
  Starts use a local Generator; legacy NumPy RNG state is unchanged.
- Results contain **only a target subset**, sorted by distance from sigma;
  this is not mode tracking, a complete spectrum or a degeneracy certificate.
  For non-normal pencils a small residual is not an eigenvalue-error bound.
  Retain `pencil` with the returned free eigenvectors for later field work.

### Original-pencil residual, signed optical convention and failure policy

Every candidate is checked independently with sparse original A and B:

```text
residual = ||A v - lambda B v||_2 /
           ((||A||_F + |lambda| ||B||_F) ||v||_2)
```

Frobenius norms use scaled O(nnz) accumulation; there is no dense matrix norm.
Common scaling in the residual avoids unnecessarily forming `lambda*Bv` at
large lambda. Nonfinite residuals/vectors, zero vectors and Bv=0 are rejected.
The matrix-norm denominator remains meaningful at a real zero eigenvalue:
TDD exposed that an action-relative denominator incorrectly rejected this
case, and the genuine diagonal zero-mode regression now passes. Matrix norms
outside representable finite range cause a clear input-range error, not a
zero residual from an infinite denominator.

`eigenvalues`, column `eigenvectors`, `residuals`, `delta=Re(lambda)`,
`alpha=Im(lambda)` and `power_loss=2*alpha` are immutable bytes snapshots with
fresh ndarray headers on every read. No absolute value or clipping of alpha
is applied. For optical pencils lambda=delta+i*alpha is signed **threshold
amplitude-gain compensation**, not the sign convention for passive beta
propagation; generic pencils receive only this algebraic labeling. Optical
lambda/delta/alpha use um^-1, B is dimensionless. Public scalar metadata is
read-only and independently owned, also accessible as attributes:

- `requested`, `returned`, `converged`, `status`, `arpack_converged`;
- `sigma`, `k`, `tolerance`, `residual_tolerance`, `seed`, `ncv`, `maxiter`;
- `candidate_count`, `discarded_transform`, `discarded_residual`;
- `lu_seconds`, `iteration_seconds`, `lu_nnz=L.nnz+U.nnz`, `linear_solves`;
- `residual_normalization='frobenius'`, `A_frobenius`, `B_frobenius`.

`ArpackNoConvergence` preserves only partial pairs passing the same original
residual gate. Default behavior raises `EigenSolveError` carrying `.partial`.
`allow_partial=True` explicitly permits a nonempty partial result, always
marked `converged=False`, `status='partial'`. Zero usable pairs always raise
(`status='failed'`), even with opt-in. ARPACK success cannot override failed
residual/transform validation. Exact singular shift LU raises
`FactorizationError` with the shift and chained SuperLU error; a close but
nonzero shift is not rejected by a coarse pivot threshold.

Zero/nonfinite mu and nonfinite back-transformed lambda are counted as
`discarded_transform`: they are never manufactured into physical zero modes.
Singular B is supported only when the shift is factorizable and the requested
subset is finite, as verified by a diagonal singular-B fixture and dense QZ.
This is not full infinite-eigenvalue classification for arbitrary singular,
defective or ill-conditioned pencils; very small nonzero mu is not arbitrarily
thresholded into infinity.

### RED/GREEN and real-chain evidence

1. Tests first: importing the missing eigensolver produced **one collection
   error** (scaffolding RED). After implementation, the real zero-eigenvalue
   case produced a numerical RED: ARPACK succeeded but the action-relative
   residual rejected its 0/1 candidates. The normwise residual fixed this.
   An extreme-v0 regression separately exposed NumPy complex-division
   underflow; component-wise scaling fixed it without global error-state edits.
2. **38 eigensolver tests passed**. Independent dense `scipy.linalg.eig(A,B)`
   and assignment-based matching cover general complex non-Hermitian B,
   complex finite C, a four-DOF minimum grid, integer sparse inputs, signed
   alpha, repeatability and multiple shifts. Guards verify one CSC LU,
   operator-only LM ARPACK, no double shift/SPD mass/dense conversion/inverse.
   Tests mutate all public ndarray shape/dtype/strides, metadata and original
   sparse inputs without changing saved results or subsequent solves.
3. Controlled `ArpackNoConvergence` injections exercise partial/empty branches
   only; they are **not evidence of successful physical ARPACK convergence**.
   Separate real 90-DOF ARPACK run (`k=5,ncv=8,maxiter=1,tol=1e-14`) raises
   incomplete-convergence diagnostics. Real successful ARPACK with a deliberately
   impossible `residual_tol=1e-30` is rejected by the original residual gate.
4. Nonmock physical chain: TE `[1,9,1]`, core thickness 0.32 um, k0=6 um^-1;
   circle radius 0.13a, center (0.36a,0.47a), dielectric contrast 2 with mean
   core epsilon 9. Solve the guided mode, derive **a=0.38000347595163375 um**,
   build radiating/non-Hermitian coupling at **D=1, quadrature order 12**,
   assemble **Nx=3, Ny=2, Lx=20a, Ly=16a, n=24**, then solve with
   **sigma=0.02+0.3j, k=4, tol=1e-12, residual_tol=1e-8, seed=21,
   ncv=20, maxiter=240**. Fields unpack with exact incoming zeros and their
   interpolated centers equal Bv. Test-only dense QZ comparison gives maximum
   eigenvalue mismatch **8.689e-16 um^-1**; maximum original residual
   **1.621e-15**. Eigenvalues (um^-1):
   `(0.264624099796104+0.331611502897288j)`,
   `(0.264624199188648+0.331611179780025j)`,
   `(0.234659923221454+0.438865862133920j)`,
   `(0.234659824108128+0.438866015482556j)`.
5. One measured full-suite execution: LU **8.060e-5 s**, ARPACK iteration
   **9.810e-4 s**, **228 factor nnz**, **37 LU solves**. Physical-chain test
   wall time including dense oracle and field assertions **0.03560 s**.
   These are local single-run timings, not acceleration measurements or
   large-grid performance claims. No dense oracle exists in production.
6. Full `.venv/Scripts/python.exe -m pytest -q`: **350 passed**, three existing
   Fourier metadata deprecation warnings, **6.05 s**. `git diff --check`
   passed (Git's LF/CRLF notice only). Legacy sources are unchanged; no
   commit/push. Independent specification/quality reviews remain pending.

This connects the small pure-Python finite optical chain. It does **not**
complete grid/domain/cutoff convergence, full-device validation, literature
reproduction, mode tracking, transport porting, or non-normal sensitivity
certification; these remain separate acceptance tasks.

### Task 10 specification R1: exactly zero finite spectrum

**Root cause and policy.** For A=0, B=I, sigma=0.25, k=2, real ARPACK
back-maps to roundoff-size nonzero lambda. The ordinary original-pencil
Frobenius ratio is then 1/sqrt(6), regardless of the absolute error. Do not
increase residual_tol or add an absolute epsilon floor with undefined units.

Only when **every stored coefficient of A is exactly zero** (not merely a
small norm), successful existing LU of K=-sigma B establishes the regular
special case: sigma!=0, B nonsingular, T=-I/sigma, and det(-lambda B) has
only lambda=0. For each finite nonzero mu require
`abs(1+sigma*mu) <= 32*float64_eps` before representing its eigenvalue by
exact zero. This is a dimensionless conservative roundoff-consistency budget
for LU/Arnoldi, independent of both user tolerances. It is not an absolute
small-eigenvalue cutoff or an eigenvalue error bound. Poor numerical
consistency can reject an ill-conditioned special pencil; no condition-number
certification or regularization is claimed. The sigma*mu product has no units
(mu has inverse eigenvalue units); no large-shift scale is added to the usual
residual denominator. Nonzero A always follows the unchanged residual path.

For each returned exact (0,v), the original Av-lambda Bv is actually computed;
its absolute residual is zero. The degenerate 0/0 normalization is defined as
zero **only after** finite nonzero-vector and nonzero Bv checks. Thus arbitrary
nonzero vectors are valid for this regular all-zero pencil, but zero vectors
are not modes. Invalid/nonfinite transforms remain discarded before the
certificate. Failed certificates count as discarded_residual. Partial-result
and no-available-pair failure rules are unchanged. A=0 with singular/zero B,
or sigma=0, fails LU; it does not imply an enumerable regular finite spectrum.

**Behavior-first evidence.** New tests were run before the production edit:
`.venv/Scripts/python.exe -m pytest tests/test_eigensolver.py -q -k
'all_zero_finite or zero_pencil_undefined or true_tiny or large_shift or zero_spectrum_still' --tb=short`
returned **4 failed, 7 passed, 38 deselected**. The four numerical failures
were real ARPACK A=0 runs with B=I or invertible upper-triangular non-Hermitian
B, each with allow_partial=False/True; all lost 2/2 candidates at the residual
gate. After the fix the identical selection returned **11 passed, 38
deselected**. A further controlled test covers mixed certificate success and
failure, default exception versus explicit partial return, and sigma=0.

**Measured original-pencil residuals**, from the real-ARPACK recording probe
`.validation-scratch/task10_r1_metrics.py` (controlled injections only for
the final two negative controls):

| Case | Raw absolute residual (Euclidean norm) | Raw normwise ratio | Returned representation / gate |
|---|---:|---:|---|
| A=0, B=I, first pair | 6.71558e-18 | 0.4082482905 | lambda=0, absolute=0, normalized=0 |
| A=0, B=I, second pair | 2.42157e-20 | 0.4082482905 | lambda=0, absolute=0, normalized=0 |
| A=0, B=I+0.3i upper diagonal, first | 3.97412e-18 | 0.4329437305 | lambda=0, absolute=0, normalized=0 |
| Same, second | 4.18261e-18 | 0.4094706755 | lambda=0, absolute=0, normalized=0 |
| diag(1..6)*1e-24, B=I, first | 1.26437e-39 | 1.05462e-16 | lambda=1.000000000000001e-24+1.74e-42i, unchanged |
| Same, second | 4.86082e-39 | 3.36660e-16 | lambda=2.0000000000000046e-24-1.61e-41i, unchanged |
| Wrong lambda=1000, v=ones, A=diag(1..6), sigma=1e12 | 2440.9201 | 0.4052418287 | rejected; 0 returned even with allow_partial |
| Wrong lambda=1000, v=ones, A=0, sigma=1e12 | 2449.4897 | 0.4082482905 | consistency=9.999999e-10 fails certificate; 0 returned |

For the genuine zero pairs, the dimensionless consistency errors are
2.68623e-17, 9.68628e-20 (identity B) and 1.44574e-17, 1.60881e-17
(non-Hermitian B). The raw back-mapped lambdas/residuals above are probe
measurements, not the residuals of the exact returned zero pairs. The two bad
pairs would pass a naive shift-scaled denominator at residual_tol=1e-8;
they do not pass this implementation. Tiny nonzero eigenvalues are tested
relative to their own pencil scale with atol=0, not an absolute tolerance
that could accidentally accept zeros.

Final verification after the additional partial-certificate test:
`.venv/Scripts/python.exe -m pytest -q` returned **362 passed, 3 warnings in
6.24 s** (only existing Fourier metadata deprecations); `git diff --check`
passed (only Git LF/CRLF notices). Only Task 10 solver/tests and this validation
appendix were changed
for R1; the pre-existing finite/__init__.py Task 10 edit was left intact.
No commit/push; independent specification re-review is next, then quality
review if specification passes.

### Task 10 quality R2: scale-safe original-pencil residual

**Numerical RED.** The residual numerator used an unscaled Euclidean norm:
its sum of squared tiny components underflowed to zero even though the
Frobenius matrix norms were scaled. With A=s*diag(1..6), B=s*I, sigma=0,
controlled mu=1/1.5 and v=e1, the wrong lambda=1.5 passed at s=1e-170.
Its independent dimensionless residual is **0.03783972516125606**, well above
the unchanged default residual_tol=1e-8. The new common-pencil regressions,
run before the production repair, returned **11 failed, 9 passed, 50
deselected**. These were behavioral failures, including wrong-pair acceptance,
false zero correct-pair residuals, and real strict-gate acceptance.

**Policy and implementation.** Evaluate the same original-pencil ratio
`||Av-lambda Bv||2 / ((||A||F+|lambda| ||B||F)*||v||2)` on residual-only sparse
copies `A/c, B/c`, where c is their shared largest real/imaginary coefficient.
Use these same copies for both matvecs and sparse Frobenius norms; the common
factor c cancels algebraically. Remove vector amplitude before matvecs, another
factor which cancels. Norms scale real/imaginary components before squaring;
the denominator terms are also scaled before their sum and final division.
No unitful floor, relaxed tolerance, altered eigenvalue, or large-shift
normalization is introduced. **The physical A-sigma B LU and inverse action
are unchanged**, as is the strict stored-A=0 R1 certificate. Original-unit
A_frobenius/B_frobenius metadata is preserved. Residual-only copies add O(nnz)
memory, not dense matrices. Their setup remains outside the LU/iteration
reported timings.

If normalization would erase a nonzero matrix coefficient, raise an explicit
numeric-range ValueError rather than silently changing the pencil. Invalid
candidate vectors, detected nonfinite/range failures in residual arithmetic,
or an underflowed nonzero final ratio fail closed. NumPy errstate alone does
not detect SciPy sparse-kernel underflow; the R3 checks and support limits below
supersede the earlier blanket underflow claim. The degenerate zero normalization
is restricted to the certified A=0, lambda=0 case with nonzero Bv. Arbitrarily
extreme dynamic range is not promised; conservative rejection is preferable
to manufacturing a zero residual.

**Measured raw values.** The executable probe and raw candidate values are
`.validation-scratch/task10_r2_metrics.py` and `task10_r2_metrics.json`.
The historical column deliberately uses the unsafe unscaled NumPy numerator
for comparison; it is not used to accept candidates. The independent oracle
uses the standard-scale diagonal pencil and Python `math.hypot`, not any
production residual/norm helper.

| Case | s | Historical unscaled numerator | Fixed residual | Independent residual | Gate |
|---|---:|---:|---:|---:|---|
| wrong lambda=1.5, e1 | 1 | 0.5 | 0.03783972516125606 | 0.03783972516125606 | reject |
| same | 1e-170 | 0 | 0.03783972516125606 | 0.03783972516125606 | reject |
| same | 1e170 | inf | 0.03783972516125607 | 0.03783972516125606 | reject |
| near-correct lambda=1+1e-10, v=(1,2e-10i,0,...) | 1 | 2.2360680143235242e-10 | 1.8651180816140854e-11 | 1.8651180815767828e-11 | accept |
| same | 1e-170 | 0 | 1.8651180816140854e-11 | 1.8651180815767828e-11 | accept |
| same | 1e170 | inf | 1.8651180816140854e-11 | 1.8651180815767828e-11 | accept |
| real ARPACK strict, first pair | 1e-170 | 0 | 1.2805060245507373e-16 | 1.3256579553441177e-16 | reject at 1e-30 |
| real ARPACK strict, second pair | 1e-170 | 0 | 2.0503159475827967e-16 | 2.0500896942551253e-16 | reject at 1e-30 |

Roundoff-size residuals need not be bit-identical across algebraic reorderings;
the strict real-ARPACK test records every candidate, independently recomputes
its residual, compares reported magnitudes, and checks that the acceptance
counts match that oracle. It does not assert that an accidentally exact
floating-point pair must fail. At s=1 and 1e170 the measured real strict pairs
also all fail at 1e-30, with fixed residuals between 1.12e-16 and 1.16e-16.

**Persistent coverage and GREEN.** The 20 initial common-scale cases passed
after the repair (**20 passed, 50 deselected**). Added coverage also checks
scaled real A=0/non-Hermitian B, mixed valid/invalid ARPACK partial candidates,
and explicit exhausted-range failure. Tests cover vector amplitudes 1,
1e-170 and 1e170 independently of pencil scaling, preserve the nonzero
1e-24 spectrum under both extreme pencil scales, and reject wrong pairs
with either allow_partial setting. Full `.venv/Scripts/python.exe -m pytest
-q --tb=short` returned **385 passed, 3 existing Fourier deprecation warnings,
6.29 s**. `git diff --check` passed (only existing LF/CRLF notices).

Only Task 10 eigensolver/tests and this validation appendix were edited for
R2; the prior finite/__init__.py edit was left intact. No commit/push.
Independent quality re-review remains a separate gate; a green test suite is
not represented as an independent reviewer approval.

### Task 10 quality R3: sparse-product range guard

**Actual RED.** With A=diag(1e-200,0,0,0,0,0), B=I,
v=(1e-150,1,0,0,0,0), controlled mu=1 and sigma=-1, the candidate lambda
is zero. Its true normalized residual is approximately 1e-150, representable
and above residual_tol=1e-200. SciPy computes the nonzero 1e-350 product as
zero without raising under np.errstate(all='raise'), so R2 wrongly accepted
it. The new behavior-first selection `-k 'sparse_residual or
scalar_product_range' --tb=short` returned **25 failed, 73 deselected**:
15 wrong-pair cases failed with DID NOT RAISE EigenSolveError, one downstream
scalar-product case failed with DID NOT RAISE ValueError, and nine safe-product
cases lacked the newly required diagnostic. These are distinct from a
collection/setup failure.

**Conservative policy.** Before residual-only CSR matvecs, inspect all four
real/imaginary component products for each stored coefficient and its indexed
vector entry. frexp supplies exponents without forming a trial product.
Nonzero factors with exponent sum below -1020 are rejected: mantissas in
[1/2,1) otherwise guarantee a product >=2^-1022. This intentionally rejects
some normal boundary products as well as subnormal/zero-underflow products.
Zero factors are exempt, including explicit stored zeros and zero components;
no dense or arbitrary-precision fallback is added. The same pre-product guard
covers (lambda/scale)*Bv and the eigenvalue-dependent denominator product.
Scaling checks retain nonzero components, and subnormal nonzero outputs at
matvec, scaled-action/eigenvalue, difference and denominator-term boundaries
are conservatively rejected. NumPy arithmetic exceptions remain an additional
check, not a substitute for inspecting SciPy kernel operands.

Range-rejected candidates increment both discarded_residual and its new
subset discarded_numeric_range. Exceptions include the numeric-range count;
no residual zero is reported for these candidates. Existing partial/no-pair
rules, physical LU/operator, eigenvalue mapping and user tolerances are
unchanged. This is a normal-product working-range guard, **not** a general
correctly-rounded/cancellation-error certificate for arbitrarily extreme
pencils. It does not inspect every internal sparse accumulator operation.
Scaled norms still permit loss of negligible components relative to their
maximum before squaring; they do not ban all subnormal norm intermediates.
The final nonzero-to-zero ratio check remains in place. Unsupported extreme
ranges may therefore be conservatively rejected rather than solved.

**GREEN and verification.** Expanded coverage exercises real, imaginary and
mixed complex phases independently in coefficients and vectors; CSR, CSC,
COO, DIA and BSR inputs; both partial flags; unchanged np.errstate; exact zero
products; safe 1e-250 products; the downstream complex scalar product; and
the conservative normal/subnormal exponent boundary. Initial implementation
passed all **98** then-present eigensolver tests. After the extra coefficient
phases and boundary coverage, the full command
`.venv/Scripts/python.exe -m pytest -q --tb=short` returned **441 passed,
3 existing Fourier metadata deprecation warnings in 6.44 s**. This includes
the existing 1e-24 spectra and optical chain: dense-oracle eigenvalue error
8.69e-16, maximum returned residual 1.62e-15. `git diff --check` passed with
only existing LF/CRLF notices.

R3 edits are limited to src/cwt3d/finite/eigensolver.py,
tests/test_eigensolver.py and this document. The prior finite/__init__.py edit
is untouched. No commit or push was made; independent quality re-review is
still required.

## Task 11: finite complex fields, signed power and same-grid tracking

### Public API and amplitude convention

```python
from cwt3d.finite import reconstruct_field, reconstruct_mode, reconstruct_modes, track_modes
modes = reconstruct_modes(pencil, eigen_result, coupling=coupling_result)
m = reconstruct_mode(pencil, eigen_result, 0)
t = track_modes(modes, newer_modes, min_overlap=.8)
```

`reconstruct_field(pencil, vector, eigenvalue, coupling=None, phase_fix=True)`
is the single-vector diagnostic primitive. Reconstruction handles only actual
returned columns, including partial results, without padding missing modes.
Negative/out-of-range/bool/noninteger indices, wrong shapes, zero/nonfinite
vectors, and nonfinite eigenvalues are rejected. Optional CouplingResult must
match pencil.C and units exactly; its components must be finite (4,4) arrays
summing to C within component-scaled floating-point error.

Normalize free faces with `Pcenter=hx*hy*sum_cells,sum_4 |U|^2=1`, `U=Bv`.
Remove common input amplitude before squaring (tested factors 1e200/1e-200).
Default global phase makes the largest free component real positive;
phase_fix=False retains the input phase. Never edit lambda or imaginary signs.
The returned immutable FiniteMode exposes:

- free_vector, four full complex faces with exact incoming zeros, four complex
  centers, center_vector, component_intensities (4,Ny,Nx), intensity (Ny,Nx);
- eigenvalue, delta, signed alpha, normalization and read-only units;
- recomputed original-pencil Frobenius normwise residual, boundary_error
  (max incoming amplitude), reconstruction_error (max interpolation-minus-Bv);
- read-only scalar budget. Every ndarray getter uses independent metadata
  directly over immutable bytes, including face/center and tracking arrays.

This chosen amplitude convention gives envelopes um^-1, intensity um^-2,
dimensionless Pcenter, and budget rates um^-1, **not electromagnetic watts**.
Intensity sums squared envelopes, not microscopic interference-resolved fields.
Arbitrary diagnostic vectors are allowed: large residual/balance errors are
reported, not hidden. A forged EigenResult residual of zero cannot bypass
recomputation. Arbitrarily extreme physical units/dynamic ranges are not
certified; detected nonfinite/exhausted normalization or residual arithmetic
raises ValueError. No correctly-rounded arbitrary-range power guarantee.

### Signed power identity and independent physical tests

```text
Pedge = hy*sum(|Rx_right|^2+|Sx_left|^2)
      + hx*sum(|Ry_top|^2+|Sy_bottom|^2)
Pc = 2*hx*hy*Im sum_cells U^H C U
2*Im(lambda)*Pcenter = Pc + Pedge
```

Budget keys: center_power, edge, signed coupling, threshold,
balance_signed_error, balance_absolute_error, balance_relative_error.
The relative denominator is `abs(threshold)+abs(coupling)+edge`, with explicit
zero-scale handling and **no 1.0 floor**. A 1e12 length-rescaling regression
with a deliberately wrong small imaginary eigenvalue checks this distinction.
Generic complex C may contain gain: Pc is never absolutized; no invented
absorption/radiation decomposition is provided.

With matching CouplingResult, add C1D, Crad, C2D signed quadratic powers,
component_sum_error, all_aux_evanescent and radiation_scope. Only under the
actual all-evanescent auxiliary condition is Crad called all radiative coupling
in this lossless TE model. Otherwise scope explicitly states zero-order only,
auxiliary orders may radiate. That label branch uses controlled metadata in a
test, not a new propagating-auxiliary physical experiment.

Full-face integrals are independently computed without B/budget, for actual
finite sparse eigenmodes on 1x1, 1x4, 4x1, 3x2 grids. Hermitian C gives edge-only
threshold; generic gain/loss retains signed coupling. Real dual-PC-layer
waveguide/geometry/Green coupling on 4x3, 15x12 um verifies rad+edge threshold
and negligible C1D/C2D powers under its lossless all-evanescent hypotheses.

### Field overlaps, crossings and degenerate subspaces

`track_modes(old,new,min_overlap=.7,degeneracy_atol=1e-8,degeneracy_rtol=1e-6)`
requires identical grids AND physical domains. It uses normalized complex
center-field amplitude overlap `abs(hx*hy*sum conj(U_old)*U_new)`, not squared
power, frequency ordering or intensity-only overlap. Eligibility is imposed
BEFORE Hungarian optimization, with dummy unmatched entries; maximize match
count first and then summed scores.

- assignment is per-old index or -1; scores is assigned amplitude or NaN;
  overlap is the full pair score matrix. matched, unmatched_old, unmatched_new
  account for differing returned counts and low-overlap missing/new entries.
- aligned_vectors contains only globally phase-adjusted NEW free vectors for
  definite pairs (None otherwise). Preserve the new eigenvalue; actual sweep
  aligned pairs are checked again against the new pencil.
- Connected eigenvalue clusters use complex lambda distance <= atol +
  rtol*max(abs(lambda_i),abs(lambda_j)). Equal-dimensional full-rank clusters
  use SVD principal overlaps; minimum principal overlap must pass min_overlap.
- clusters reports only forced member correspondences/principal_overlaps;
  ambiguous/ambiguous_new combine eigenvalue-cluster and global-assignment
  ambiguity, explicitly denying strong individual identity. Cluster-covered
  and assignment-ambiguous entries are not
  called lost/new. Return no cluster basis rotation and never mix nondegenerate
  vectors. Exact and near-degenerate unitary basis rotations retain subspace
  identity when all individual overlaps fail the threshold. Rank-deficient
  bases cannot certify a full subspace.

Limits: same-grid returned-subset correspondence only, not complete global
mode identification. Cross-grid matching explicitly raises. Connected chains
can conservatively mark broad clusters ambiguous; truncated/unequal clusters
cannot be certified. Ambiguous unmatched entries do not prove physical
mode disappearance. Cross-grid common-sample matching, convergence, far field,
performance benchmarks, D=10 and full-paper/device validation remain separate.

### Executed RED/GREEN and measured evidence

Initial tests failed collection because fields.py was absent: scaffolding RED,
not a numerical failure. Additional malformed-component tests then gave
**3 failed, 29 deselected**: NaN accepted, wrong shape error, inconsistent
component sum accepted. Explicit validation fixed this behavioral RED.
Final targeted suite: **33 passed**, no new warnings. Full
`.venv/Scripts/python.exe -m pytest -q --tb=short`: **474 passed, 3 existing
Fourier metadata-setter warnings, 6.93 s**. `git diff --check` passes.

Real metrics probe: `.validation-scratch/task11_metrics.py`, raw JSON
`task11_metrics.json`. Existing dual-PC-layer physical fixture, D=1,
quadrature order 10, sigma=.025+.12j, k=6, tol=1e-12. Contrast changes 2 to
2.001 at fixed grid/domain; both are real geometry-to-finite-mode solves.

| Six actual finite modes | Maximum error / observed result |
|---|---:|
| Independent full-face integrated normalization error | 6.66e-16 |
| Independent full-face power-balance absolute error | 2.71e-16 um^-1 |
| Reported relative power-balance error | 5.18e-16 |
| Incoming boundary / face-center reconstruction errors | 0 / 0 |
| Recomputed original-pencil residual | 1.33e-16 |
| Absolute C1D / C2D power | 1.63e-18 / 7.62e-20 um^-1 |
| Perturbed contrast matched subset | 6 of 6, none missing/new |
| Minimum complex-field overlap amplitude | 0.999999987316115 |

Reordered power sums differ at rounding scale from reported balance errors;
they need not be bit-identical. The real sweep has no ambiguous clusters at
these tolerances. Degeneracies and frequency crossings are controlled fixtures,
not mislabeled physical-device experiments.

New files: finite/fields.py, finite/tracking.py, tests/test_finite_modes.py;
modified finite/__init__.py exports and this appendix. No results.py needed
merely for layering. Existing numerical implementations/historical source are
unchanged. No commit/push. **Await independent specification review followed
by independent quality review in the parent workflow; tests are not approval.**

### Task 11 quality R1: global assignment ambiguity and unit overlap

**Behavioral RED.** Before editing tracking.py, the new normal-pencil test
on grid 1x1, Lx=Ly=1, C=diag(0,1,2,3), Cnew=Q C Q.T (Hadamard first
two components) failed both reorder/phase variants: the original tracker
called both tied nondegenerate edges definite. These are real valid pencil
eigenpairs (maximum measured residual 1.6738104569903572e-17), not arbitrary
diagnostic vectors. Their 2x2 overlaps are all 0.7071067811865476; eigenvalues
0+2j and 1+2j are not clustered. A third unchanged mode demonstrates that
the forced (2,2) edge survives. The first targeted execution returned
12 failed, 33 deselected: two actual false-identity assertions, nine missing
helper imports, and one missing diagnostic attribute. The import failures
are new-API scaffolding, not additional numerical RED evidence.

**Policy/algorithm.** Keep eligibility-before-optimization and maximum match
count first, summed overlap second. For each edge of one optimal assignment,
forbid it and re-solve; an equal-cardinality alternative within
`64*float64_eps*max(1,matched_count)` of the optimum makes that edge non-forced.
Only forced edges enter matched/assignment/scores/aligned_vectors or clusters.
No frequency tie-break, vector rotation/mixing, or eigenvalue change occurs.
Score sums use math.fsum; the tolerance is a fixed dimensionless roundoff
budget, not a user-defined physical closeness criterion.

Rectangular assignments also force each initially unmatched vertex to
participate (using an objective-dominating bonus) and compare its best
original objective against the optimum. This identifies *all* uncertain
matched/unmatched vertices: merely recording one forbidden-edge witness can
miss a third tied candidate. Both single-mode and principal-subspace cluster
assignments use the same analysis. assignment_ambiguous_old/new isolate this
cause; ambiguous/ambiguous_new combine it with eigenvalue-cluster ambiguity.
Uncertain participants are not asserted lost/new; genuinely never-matched
vertices remain unmatched. A definite cluster still denies individual identity.

**Unit-overlap policy.** Single and subspace eligibility both allow only
`32*float64_eps` absolute amplitude roundoff below min_overlap. No relative
or unrestricted relaxation; a deficit 1e-10 at min_overlap=1 is rejected.
The actual deterministic self-overlap probe measured 0.9999999999999999:
restoring only the original strict comparison in memory gives no match;
the corrected predicate returns (0,0). This controlled predicate comparison
was executed after implementation, not represented as a pre-code TDD run.

**Persistent checks.** Reorder/global phase; valid nondegenerate tied pencil
plus forced edge; unique optimum despite row-local tie; strict maximum
cardinality; rectangular count changes including three tied candidates;
inferior unmatched candidate; near-tie inside/outside score tolerance; exact
degenerate subspace; tied cluster selection plus forced cluster; bounded
single/subspace threshold; and 72 small random score/eligibility matrices
checked against independent exhaustive test-only partial-match enumeration.
The real dual-PC contrast 2 -> 2.001 chain now explicitly requires **6/6**
definite matches and no ambiguity, rather than only at least four matches.

**Cost/support limits.** A score matrix n x m with k optimal eligible pairs
needs at most `1+k+(n-k)+(m-k)` Hungarian solves (often fewer for ineligible
unmatched vertices), each on padded (n+m) x (n+m) weights. Thus O(n+m)
solves, O((n+m)^4) worst-case time, O((n+m)^2) assignment workspace, not
exponential enumeration in production. This is a small returned-mode-set
reference path, not a large-k accelerated tracker or certified interval
assignment solver. Local 30-repeat median timings of assignment analysis
alone, excluding overlap construction, SVD and eigensolves:

| Matrix / case | Hungarian calls | Median seconds |
|---|---:|---:|
| 6 x 6 unique | 7 | 0.0000627 |
| 24 x 24 tied | 25 | 0.00062265 |
| 6 x 24 tied | 25 | 0.00039385 |
| 48 x 48 unique | 49 | 0.00333435 |

Raw executable/output: .validation-scratch/task11_r1_metrics.py and
task11_r1_metrics.json. These are local synthetic assignment timings only;
no mesh/domain/D=10/device convergence or speedup claim follows.

**Final GREEN:** targeted tests/test_finite_modes.py: **49 passed in 0.99 s**;
full `.venv/Scripts/python.exe -m pytest -q --tb=short`: **490 passed,
3 existing Fourier metadata deprecation warnings in 6.94 s**. The physical
contrast chain passes its strengthened 6/6 gate. `git diff --check` passes;
untracked tracking/tests are additionally checked with no-index diffs against
/dev/null so Git's ordinary tracked-only diff cannot omit their whitespace.

R1 edits touch only finite/tracking.py, tests/test_finite_modes.py and this
appendix; fields.py and the pre-existing __init__.py edit remain untouched.
Legacy sources unchanged; no commit/push. Independent quality re-review is
still required.
