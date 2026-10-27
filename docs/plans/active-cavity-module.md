# Active-cavity first-module implementation plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add an opt-in, independent active-cavity module with frozen gain/index analysis and deterministic spatial carrier-photon dynamics, leaving the published passive APIs and numerical results unchanged.

**Architecture:** Reuse the existing sparse pencil and general eigensolver. Use photon-number-normalized four-wave envelopes, local material models and a finite-volume carrier diffusion operator. Preserve the staggered mass matrix and validate passive/small-signal/energy limits before interpreting nonlinear lasing.

**Tech Stack:** Python >=3.10, NumPy/SciPy only; existing pytest and package infrastructure. Local branch only; no version change, merge, push or release in this task.

## Frozen conventions and contracts

All new module units: length um, time ns, carrier/photon volume density um^-3, material power gain/loss um^-1, differential gain um^2, diffusion um^2/ns, current density A/um^2. `vg = scipy.constants.c * 1e-3 / group_index` in um/ns. Photon energy = hbar*c*k0*1e6 joules. Inputs expressed in cm units must be explicitly converted by the caller/examples, not guessed.

Existing `A0 = C_lift B + i D`, lambda=delta+i*alpha and exp(+i omega t). Use `B dv/dt = i*vg*A0*v + diag(m repeated four times)*(B v)`, where `m=vg*((Gamma*g-alpha_internal)/2 - i*k0*Gamma*dn)` in ns^-1. Add no material/radiation loss already present in C twice. No inverse or dense domain matrices in production.

Let q=Bv, S=sum_components |q|^2 (sheet photon density), U=Gamma*S/d_active. Total photons = cell_area*sum(S). With N in um^-3, `dN/dt = J*1e-9/(e*d_active) - R(N) - vg*g*U + D*Laplacian(N)`. This makes volume-integrated stimulated carrier removal equal total stimulated photon production. This is a new explicit photon-envelope normalization; existing arbitrary passive mode amplitudes are not automatically physical. A normalized passive free vector can be scaled to a chosen `initial_photons` using area*||Bv||^2.

Scalar confinement and active thickness initially; spatial gain/index/current/carriers are supported. Material parameters may be scalar or cell-shaped immutable maps to support gain/absorber regions. Group index and k0 are explicit required scalars. Gain may be negative (absorption); all rates/parameters must be finite with physical sign/domain checks. Deterministic first version: explicitly supplied initial field seed, no stochastic-linewidth, thermal, drift-diffusion/Poisson or external-feedback claims.

### Task 1: Material and carrier physics

Create `src/cwt3d/active/materials.py`, `carriers.py`; tests `tests/test_active_materials.py`, `test_active_carriers.py`.

- `CarrierMaterial(differential_gain_um2, transparency_density_um3, *, gain_compression_um3=0., index_per_density_um3=0., recombination_a_ns=0., recombination_b_um3_ns=0., recombination_c_um6_ns=0.)`.
- `gain(N,U) = differential_gain*(N-Ntr)/(1+gain_compression*U)`; `index_change(N) = index_per_density*(N-Ntr)`; `recombination(N)=A*N+B*N**2+C*N**3`. Physical initial/accepted densities nonnegative; nonlinear solver trial N can be negative, so material evaluations accept finite real N and preserve signed gain, never clip. U must be nonnegative. Arrays are immutable snapshots; every public getter is independently bytes-backed. Broadcast validation must not accidentally permit a grid dimension mismatch at cavity evaluation.
- `diffusion_operator(grid, diffusion_um2_ns)` returns real sparse ncell-square finite-volume Laplacian times D with zero-flux boundary conditions. Scalar D initially, >=0. Conservation, negative semidefinite, constant nullspace, anisotropic hx/hy and one-cell axes tested.
- `carrier_rhs(N,U,current_density_A_um2,material,*,vg_um_ns,active_thickness_um,diffusion_matrix)` returns flat derivative or consistent original shape (document it; dynamics uses flattened result). Convert current by 1e-9 to charge/ns. Reject negative injection and nonfinite shape/control values. No clipping.
- Write tests first, run RED, implement, run GREEN. Child owns only these two modules/two tests; no __init__, commits or old-core edits.

### Task 2: Optical adapter, state and frozen-active analysis

Create `src/cwt3d/active/cavity.py`, `threshold.py`; tests `tests/test_active_cavity.py`, `test_active_threshold.py`.

- `ActiveCavity(pencil, *, group_index, k0_um_inverse, active_thickness_um, confinement, internal_loss_um_inverse=0.)`; read-only scalars, immutable pencil. Require 0<Gamma<=1, physical positive controls, internal loss scalar or grid map >=0. Cache A0/B/D copies once internally; public A/B/D are sparse copies. Expose `.grid`, `.vg_um_ns`, `.photon_energy_j`, `.active_thickness_um`, `.confinement`, `.k0_um_inverse`, `.internal_loss_um_inverse`.
- `center_fields(v)` => array(4,Ny,Nx); `photon_density(v)` => U(Ny,Nx); `photon_number(v)` => scalar; `normalize_seed(v, photons)` => free vector with requested total; reject zero vector for positive requested photons; never renormalize during dynamics.
- `material_rate(N,U,material)` => flat m per cell (ns^-1), strict final broadcast to (Ny,Nx); m repeated component-major.
- `frozen_pencil(N,material,*,photon_density_um3=0.)` => sparse `(A,B)` with `A = A0 + diag(-k0*Gamma*dn - 0.5j*(Gamma*g-alpha_internal) repeated4) @ B`.
- `power_budget(v,N,material)` => instantaneous total_photons, edge_loss_per_ns, coupling_loss_per_ns (signed for generic C), internal_loss_per_ns, stimulated_per_ns, dphotons_dt_per_ns, output_power_w. Edge flux from original faces; coupling loss=vg*2*area*Im[q^H C_lift q]; internal=vg*area*sum(alpha_internal*S); stimulated=vg*area*Gamma*sum(g*S). `output_power_w` is coupling-channel photon rate * photon_energy * 1e9, must explicitly label signed/generic and total radiative vs zero-order scope from source; no upper/lower split claim. Safer use nonnegative physical radiative component only if source physical CouplingResult is retained, otherwise report signed coupling exchange and do not claim generic watts are measured radiation. Coordinate with parent if provenance needed.
- `solve_active_modes(cavity,N,material,*,photon_density_um3=0.,**solver_options)` wraps existing `solve_generalized(A,B,...)`, returns unchanged EigenResult with signed Im(lambda).
- `uniform_threshold_gain(alpha,*,confinement,internal_loss_um_inverse=0.) = (2*alpha+alpha_internal)/Gamma`; require nonnegative passive alpha and no inferred global lowest-mode claim.
- Write RED tests, GREEN; compare full arrays vs dense small independent oracle, passive recovery, uniform gain/index shifts and patterned perturbations, mutation safety. Do not edit old finite core, materials, carriers, dynamics or __init__; no commits/push.

### Task 3: Deterministic joint dynamics and immutable results

Create `src/cwt3d/active/dynamics.py`, `results.py`; tests `tests/test_active_dynamics.py`.

- `ActiveState(field_vector, carrier_density_um3)` bytes snapshots, finite complex 1D field and finite nonnegative scalar/2D N (cavity checks exact shape); public `.field_vector`, `.carrier_density_um3` fresh read-only headers.
- `simulate(cavity,material,initial,*,times_ns,current_density_A_um2=0.,diffusion_um2_ns=0.,rtol=1e-6,field_atol=1e-10,carrier_atol=1e-3,max_step_ns=np.inf,method='BDF',max_output_bytes=256*1024**2)` returns immutable `ActiveResult`.
- Real packed state (Re(v),Im(v),Nflat), derivative field computed by a single sparse LU of B reused, preserving B. No explicit inverse/dense Jacobian. Use SciPy solve_ivp BDF/Radau and a *conservative proven sparse Jacobian pattern*, including nonlocal B^-1 triangular line reachability. Restrict methods to implemented/tested ones. Dense Jacobian fallback is unacceptable. Constant/callable(t_ns) current profile accepted; every callback output validated/copied. No noise or silent regularization.
- N trial evaluations may be signed, accepted physical outputs below -carrier_atol raise with failure status/partial diagnostics; tiny negative values within the declared absolute numerical tolerance must remain visible, not clipped. Require time-refinement checks for real accuracy; don't present this tolerance as a carrier positivity proof. All returned samples must be finite. Integration failure must raise, never present last point as success; include useful partial result if feasible.
- Sparse Jacobian construction: B^-1 support within each component's upstream/downstream grid line. field/field support R*(support(A0)+support(B)), field/N support R*T where T stacks 4 identities, carrier/field support T.T*support(B); carrier/N support I+diffusion. Duplicate real/imag block supports and assemble sparse bmat. Bounds/memory accounting before allocation; output sizes explicitly checked.
- Results expose `.times_ns`, `.field_vectors` (nt,4n), `.carrier_density_um3` (nt,Ny,Nx), `.photon_number`, `.metadata`; store budget traces if cavity exposes them. A deterministic zero-field initial condition stays zero, no invented spontaneous start. CW diagnostics may report stationary intensity/carrier maps, not certify spectral stability or uniqueness.
- Tests RED then GREEN: exact no-field carrier injection/recombination, diffusion conservation, passive exp(i vg lambda t), uniform gain factor2, reciprocal stimulated photon/carrier bookkeeping, zero-field, malformed controls, mutation, temporal refinement, jac sparsity finite-difference support. Child owns only four specified files; no commits/old-core edits.

### Task 4: Integration and executable examples (parent after children)

Create `src/cwt3d/active/__init__.py`, `examples/active_cavity.py`, `docs/active-cavity.md`, integration tests if needed. Export actual APIs only. Example constructs actual chapter4 passive coupling with small grid, seeds a physical photon count, performs uniform threshold and bounded pumped transient, exports numeric NPZ/JSON only into fresh output. Every material/normalization parameter carries units and demonstration-not-device-fit provenance. Include time-refinement, carrier/photon balances, final-state diagnostics and version/source provenance. Keep README passive quickstart unchanged except optional link to experimental module docs.

### Task 5: Independent scientific specification review

Review equations/signs/units/photon-carrier reciprocity and synthetic vs physical claims. Check passive limit, uniform thresholds, nonuniform maps, time errors, fixed reference-mode scope and no hidden noise/thermal/transport claims. Fix regressions before quality review.

### Task 6: Independent implementation quality review and delivery

After scientific specification PASS, inspect immutable snapshots, sparse scaling, bad inputs/failure paths, minimum Python3.10, packaging, and full regression. Run full pytest and installed wheel smoke from outside repository. Preserve old passive core source identity, no automatic release. Commit only after reviews pass; final response reports actual examples/results and explicit first-module scope.

## Recorded implementation decisions

- Optional power-budget trajectories are evaluated from returned field/carrier snapshots by the executable example and saved in its numeric export, rather than retained as redundant arrays inside `ActiveResult`. `ActiveCavity.power_budget` remains the public per-state diagnostic; automatic in-result budget storage is deferred. This preserves complete output bookkeeping without expanding the first result API or its storage estimates.
- The proven field/field Jacobian support additionally includes `R*T*T.T*support(B)` for gain-compression coupling across the four waves; the preliminary line52 expression alone is insufficient.
- Physical coupling/reference wave numbers are bound by strict float equality after validation; use `coupling.k0` for that constructor argument. Generic numeric matrices retain explicit caller-supplied reference-frequency semantics.
