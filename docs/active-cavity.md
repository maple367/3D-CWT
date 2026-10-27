# Active-cavity analysis

This opt-in module adds carrier and gain/absorption physics on top of the existing passive spatial operator. The passive eigenproblem, examples and public interfaces remain unchanged. The module is included in `cwt3d 1.4.0` and later. The source distribution and repository include the executable examples; the wheel installs the library.

## Scope

The first implementation uses a scalar TE reference mode, four spatially varying complex wave envelopes and one cell-centered carrier density per transverse cell. It includes signed material gain/absorption, gain compression, carrier-induced index changes, injection, ABC recombination and lateral carrier diffusion with no-flux boundaries. A fixed-carrier linear solve and a uniform-mode threshold formula bridge passive and self-consistent dynamics.

The deterministic time solver requires an explicit initial field. A zero field remains zero; nonzero emission requires a nonzero optical seed. It has no spontaneous-emission noise, linewidth prediction, thermal transport, Poisson/drift-diffusion electrical solver or external optical feedback. A stationary-intensity time window does not certify a unique, stable, single-frequency CW solution. The four wave components are a spatial basis, not a restriction to four finite-cavity modes.

## Equations, units and normalization

Use lengths in um, time in ns, carrier/photon volume densities in um^-3, material power gain/loss in um^-1 and current density in A/um^2. Convert literature values in cm units explicitly. `group_index` is not the effective phase index. The speed is `vg=c*1e-3/group_index` in um/ns, and the photon energy is `hbar*c*k0*1e6` joules.

The passive pencil is

    A0 v = lambda B v,   A0=C_lift B+iD,   lambda=delta+i*alpha.

The time equation retains the staggered averaging matrix B:

    B dv/dt = i*vg*(A0-reference_detuning*B)*v + M(N,U)*Bv,
    M = vg*((Gamma*g(N,U)-alpha_internal)/2-i*k0*Gamma*Delta_n(N)).

The positive imaginary unit before the coupling follows the mandatory correction in [Inoue et al., PRB99,169904(2019)](https://doi.org/10.1103/PhysRevB.99.169904). Their original Eq.(8) and Appendix(A6)-(A9) omitted that unit. The explicit later field/carrier equations are in [Inoue et al., Nature Communications14,50(2023), Supplementary Note2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-022-35599-2/MediaObjects/41467_2022_35599_MOESM1_ESM.pdf). This module uses the declared linear/compressed material law below, not an unverified implementation of the original paper's complete gain-dispersion model.

`reference_detuning_um_inverse` is an optional rotating-frame offset. It changes optical phase, not photons or carrier physics. For a passive eigenvalue lambda, the envelope evolves as exp(i*vg*(lambda-reference_detuning)*t). Add back the reference-frame frequency before interpreting phase-derived physical frequencies.

Define q=Bv as **photon-number-normalized** center envelopes, S=sum(|q|^2) as sheet photon density and

    U = Gamma*S/d_active,
    total_photons = hx*hy*sum(S).

This is a declared new amplitude convention, not a claim that the old arbitrarily normalized eigenfield is already in V/m or photons. `normalize_seed(v,photons)` sets an initial photon count once; dynamics never renormalizes the field.

The carrier equation is

    dN/dt = J*1e-9/(e*d_active)-R(N)-vg*g(N,U)*U+D_N*laplacian(N).

Multiplying the stimulated carrier term by active volume d_active*cell_area exactly cancels the integrated stimulated photon-production term. Gamma must not be applied a second time to that carrier term. Material loss, edge leakage and radiative coupling are separate channels; do not count losses already present in C twice. Material gain is a **power** gain, so its optical-amplitude contribution contains 1/2. The passive alpha is an amplitude loss.

## APIs

```python
from cwt3d.active import (
    ActiveCavity, CarrierMaterial, ActiveState, simulate,
    solve_active_modes, uniform_threshold_gain,
)
```

Given an existing `FinitePencil`, construct an `ActiveCavity` with explicit group index, k0, total active thickness and scalar confinement. Internal material loss can be scalar or a cell-shaped map. An optional physical `CouplingResult` must match the passive matrix and supplies the provenance needed for radiative-output diagnostics; its `k0` must be a finite positive real scalar exactly equal to `k0_um_inverse` after float conversion (no tolerance), while a bare numeric C has no frequency metadata and uses the user-declared k0. When retaining a physical result named `coupling`, pass `k0_um_inverse=coupling.k0` to avoid a rounded-frequency mismatch.

`CarrierMaterial` evaluates

    g(N,U)=g_N*(N-Ntr)/(1+epsilon*U),
    Delta_n(N)=index_per_density*(N-Ntr),
    R(N)=A*N+B*N**2+C*N**3.

The coefficients can be scalar or cell-shaped snapshots, allowing different gain and absorption regions. Negative gain represents absorption and is not clipped. Finite signed N trial values are permitted inside the nonlinear solver; initial physical densities must be nonnegative. Accepted negative carrier values beyond the declared carrier absolute tolerance are an error. Tiny signed numerical values within that tolerance remain visible, never silently projected to zero.

`solve_active_modes` builds the frozen spatial gain/index perturbation and reuses the existing sparse general-pencil solver. Its output is a near-shift subset with signed Im(lambda), not a complete threshold ranking. For a **specified passive mode**, uniform pumping and fixed reference index,

    g_threshold=(2*alpha_mode+alpha_internal)/Gamma.

With nonuniform gain, index feedback or saturation, recompute the active spatial operator; do not promote the uniform formula into a universal threshold condition.

The time solver uses a real-packed implicit BDF/Radau system and sparse Jacobian structure including the line-wise nonlocality of B inverse. B is factored once for derivative evaluation; no dense inverse is constructed. Check output-memory limits before choosing a large grid/time array. An integration error is not a successful last sample.

## Run a complete example

```bash
python -m pip install -e ".[dev,plot]"
python examples/active_cavity.py --N 8 --D 3 --q 12 --duration-ns 1 --output results/active-demo
```

The example uses the actual Liang chapter4 passive layer stack, solves its reference frequency and one passive mode, and estimates the group index from d(beta)/d(k0) while holding material permittivity fixed. The active thickness, confinement and gain/recombination parameters are **illustrative user-adjustable values**, not calibrated parameters of that device. The selected passive mode is not certified as the global lowest-threshold mode.

The seed is scaled to an explicit photon count, and carriers are initially uniform at `min(pump_ratio,1)*Nth`. This is a prebiased deterministic evolution, not noise-started turn-on from an empty active layer. The pump ratio refers only to the analytic uniform, weak-field threshold of that selected passive mode. The result saves `trajectory.npz` and `summary.json` to a fresh directory, with units, material assumptions, passive controls, reference frame and solver provenance. Material-index feedback can be enabled using `--index-per-density-um3`; its default zero value makes the introductory trajectory simpler, while the material/cavity APIs and tests exercise nonzero feedback.

Interpret final-window photon variation as a diagnostic of the trajectory, not a CW stability certificate. Tighten relative/absolute tolerances and maximum step separately; Fourier/vertical/grid convergence of the passive optical input remains a separate issue. Validate parameters before making an absolute output-power or device-performance claim.

## Diagnostics

The instantaneous photon balance separates edge loss, signed passive-coupling loss, internal material loss and stimulated gain. Generic numeric C matrices provide signed coupling-exchange diagnostics; these are not automatically vertical radiative watts. With a verified physical coupling result, the Crad channel can be reported separately. Its scope is all represented vertical radiation only when auxiliary orders are evanescent; otherwise it is the zeroth-order channel only. No up/down radiation split is inferred.

Numerical acceptance gates include passive frequency/decay recovery, uniform gain/index shifts, carrier-only injection/recombination, diffusion conservation, stimulated carrier/photon bookkeeping, temporal refinement and immutable snapshots. Physical device validation requires independent gain/index data and experimental comparisons beyond these implementation gates.

[Executed validation, time-refinement metrics and trajectory figures](validation/active-cavity/README.md) are archived separately from the API guide.
