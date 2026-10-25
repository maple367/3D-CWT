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
