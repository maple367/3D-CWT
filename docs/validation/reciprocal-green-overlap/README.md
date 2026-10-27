# Reciprocal Green-overlap reuse

This local optimization changes only the evaluation of the overlap integral. It does not introduce a new physical approximation, impose geometric symmetry, reduce precision, or reduce quadrature order.

## Default-on calculation setting

`build_coupling(..., reciprocity=True)` and `green_overlap(..., reciprocity=True)` enable reuse by default. Set `reciprocity=False` to restore the original two-evaluation expression. Both example CLIs accept `--no-reciprocity` and `--reciprocity`; `liang_70um.run_case(...)` exposes the same boolean. The selected setting is retained in result metadata, provenance, exported calculation inputs and distinct cache keys. Unknown kernels remain on the original path even when reuse is enabled.

After adding the switch, the full suite returned **662 passed**. The added setting tests exercise real enabled/disabled CLI exports, both backends, exact kernel-call counts, invalid boolean values, metadata, cache roundtrips and cross-setting/forged-metadata rejection. The actual API documentation example with reuse disabled was executed successfully.

The current benchmark compares the native public `False` and `True` settings, with three paired repeats per case. Latest raw results are in [with-toggle/benchmark.json](with-toggle/benchmark.json); coupling speedups were 1.85x, 1.59x and 1.80x for CC, asymmetric two-PC and irregular sampled cases. The maximum C difference remained 3.46945e-18 um^-1 and targeted finite-eigenvalue differences remained zero. The earlier report and independent rerun below are preserved under their original measured source digests, not relabelled as measurements of the subsequent configurable implementation.

## Why reciprocity applies generally to the current kernels

For each transverse wavenumber, the scalar TE reference operator is

\[
\mathcal L=\partial_z^2+k_0^2\epsilon(z)-k_\parallel^2,
\qquad \mathcal L G(z,s)=-\delta(z-s).
\]

Let \(u_L\) satisfy the outgoing left boundary condition and \(u_R\) the outgoing right boundary condition. Continuity of the scalar TE field and its derivative across interfaces preserves the Wronskian

\[
W=u_Lu_R'-u_L'u_R.
\]

Where the outgoing problem is invertible,

\[
G(z,s)=-\frac{u_L(\min(z,s))u_R(\max(z,s))}{W}
       =G(s,z).
\]

This is **transpose reciprocity, not Hermitian symmetry**. There is no complex conjugation on the right-hand side. The proof allows different upper/lower halfspaces, arbitrary internal layer widths and scalar permittivities, and passive complex permittivity within the Green-kernel API. It needs no mirror, inversion, rotation, or hole-shape symmetry. Passive lossy Green tests do not extend the real-positive-permittivity domain currently supported by the complete TE coupling pipeline.

Hole geometry enters the Fourier coefficients and averaged reference structure; it does not add a new condition to the identity above. The optimization remains within the existing scalar, reciprocal TE model. It does not assert reciprocity for arbitrary user-supplied kernels, derivative kernels, or a future nonreciprocal/vector backend. At an outgoing pole, no finite Green function exists; the existing `PoleError` behavior is retained and tested.

## What changes, and what does not

For one identical source/receiver interval, the original two Duffy triangles contribute

\[
\Theta^*(x)G(x,y)\Theta(y)+\Theta^*(y)G(y,x)\Theta(x).
\]

Evaluate `g = kernel(x,y)` once and retain

```python
tx, ty = mode.theta(x), mode.theta(y)
g = kernel(x, y)
return np.sum(weights * (tx.conj()*g*ty + ty.conj()*g*tx))
```

Both field weights are retained; the field need not be real. The original `Theta*G*Theta` multiplication order is also retained. Forming `Theta*Theta` first can overflow or underflow when the original integrand is representable; two dedicated scaling regressions exercise these cases.

The shortcut is enabled only for exact built-in `HomogeneousGreen`, `LayeredGreen`, and `PreparedLayeredGreen` types. Unknown callables and subclasses retain two independent kernel evaluations. Distinct source/receiver intervals retain the original tensor quadrature, including overlap validation. Neither cross-layer integral entries nor Fourier coefficients are equated using geometric assumptions. Source hashing already includes `quadrature.py`, so cache keys change with the implementation.

## Verification

The 22 new tests in `tests/test_quadrature_reciprocity.py` cover:

- one native kernel evaluation instead of two, for all three built-in types;
- asymmetric real and passive lossy multilayers;
- internal cutoff and strongly evanescent/thick layers;
- near-guided-pole evaluation and actual pole rejection;
- complex longitudinal fields against an independent adaptive double integral;
- unknown nonreciprocal callables and overridden subclasses falling back;
- distinct source/receiver intervals remaining unchanged;
- extreme field/kernel scales retaining finite nonzero old-formula results;
- two different patterned layers in an asymmetric guide, with an off-center circle and a non-isosceles right triangle, for both Green backends.

The complete local suite returned **644 passed**, with six existing NumPy deprecation warnings. This includes the existing kernel continuity/source-jump/boundary/power and backend-parity gates.

## Same-precision measured comparison

Run the current native-setting comparison from the checkout:

```text
.venv/Scripts/python.exe benchmarks/bench_reciprocity.py --output docs/validation/reciprocal-green-overlap/with-toggle --repeats 3
```

The historical pre-toggle benchmark reported below froze the original two-kernel overlap expression and overrode only that function in the reference run. It alternated old/new execution order, recorded every sample, used no disk coupling cache, and retained the same `complex128`, D/q/F, grid, shift rule, seed, tolerance and requested mode count on both sides. The original override is identified explicitly in that JSON; it must not be mistaken for a separately source-hashed installed build. The current benchmark instead directly compares the public `reciprocity=False` and `True` settings, as recorded in the separate `with-toggle` results above.

Python 3.13.14, NumPy 2.5.4, SciPy 1.18.1, one numerical-library thread. Three paired repeats per structure. Medians in seconds:

| Structure | Controls | Original coupling | Reused coupling | Coupling speedup | Compute speedup |
|---|---|---:|---:|---:|---:|
| Liang chapter-4 CC | N64, D10, q32, F64, k1 | 2.1043 | 1.1267 | 1.87x | 1.80x |
| Asymmetric two-PC circle/scalene triangle | N32, D6, q20, F32, k1 | 0.9335 | 0.5937 | 1.57x | 1.52x |
| Asymmetric sampled bent inclusion | N32, D6, q20, F32, k1 | 0.3759 | 0.2106 | 1.79x | 1.67x |

Compute timing includes coupling, finite-pencil assembly, targeted solve and field/power reconstruction. It excludes imports, reference-guide setup and exports. The benchmark compares one targeted finite eigenpair, not a complete spectrum; all four infinite-C eigenvalues are separately matched. An initial four-finite-mode benchmark attempt timed out without producing a completed checkpoint; the reported run requests the same one finite mode on **both** sides, so reducing mode count is not included in the speedup.

Across the completed three structures and all three repeats:

- maximum absolute C-matrix difference: **3.46945e-18 um^-1**;
- maximum matched targeted finite-eigenvalue difference: **0 um^-1** in these runs;
- maximum eigenpair residual from either path: **5.43722e-14**;
- maximum absolute difference in the recorded edge/coupling/threshold/radiative power-budget or relative balance diagnostics: **6.50522e-19** (budget entries retain their respective units).

All component and intermediate arrays, all infinite eigenvalues, finite residuals and matched-mode power diagnostics are checked. Raw samples, controls and source digest are in [benchmark.json](benchmark.json). The three completed cases each contain exactly three paired samples.

## Independent mathematical and implementation reviews

The separate mathematical/specification review confirmed that the scalar outgoing Green identity requires no layer-stack or hole-shape symmetry. Its baseline-code inspection and small asymmetric lossy-stack/complex-field probes supported the identity; it did not audit the final implementation or run the complete suite. In particular, it confirmed the need for arbitrary-callable/subclass fallback and near-pole regression, both included in the implementation tests above. It also identified the future boundary: derivative kernels exchange the differentiated argument under reciprocity, so they must not inherit this value-kernel shortcut; mixed source/receiver modes or different weights require their own exchanged-field expression. No new pole regularization or gain-material support is introduced.

The independent code review returned APPROVED, reran all 22 new tests (22 passed), and repeated each of the three full benchmark structures for one original/reused pair. The maximum C difference was again 3.46945e-18 um^-1, with zero matched targeted finite-eigenvalue difference in all three. The rerun source digest matches the measured implementation. The reviewer did not rerun the full test suite; the 644-test full-suite execution above is the implementer's verification. This independent one-repeat rerun is not a replacement for the original three-repeat timing table. Raw rerun data are preserved in [independent-rerun.json](independent-rerun.json).

**Applicability is general within the stated model; speedup is workload-dependent.** Only the diagonal same-interval overlap blocks are reduced. Many patterned layers produce more untouched cross-layer work, so a universal twofold end-to-end speedup is not claimed. These comparisons certify implementation parity at fixed controls, not convergence of the underlying physical model or its truncations.
