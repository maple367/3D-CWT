# A 70 µm square PCSEL

This example follows the circular-hole (CC) device in Chapter 4 of
[Yong Liang's doctoral thesis](https://doi.org/10.14989/doctor.k18283),
Table 4.1 and §4.3. The lattice constant is 295 nm, the air filling fraction is
0.16, and the square cavity has side length 70 µm. The envelope equations use
zero incoming waves at the four edges, as in thesis Eq. (4.22).

![Calculated finite-cavity envelope and line profiles](../images/liang-70um.png)

## Structure

The calculation uses the following layers in the order listed in Table 4.1.
The PC layer contains a circular air hole of radius
`a*sqrt(0.16/pi)` in a GaAs background.

| Layer | Thickness (nm) | Refractive index |
| --- | ---: | ---: |
| n-cladding, AlGaAs | Halfspace | 3.307 |
| Spacer, GaAs | 30 | 3.524 |
| Active region, effective 3QW | 64 | 3.553 |
| Spacer, GaAs | 40 | 3.524 |
| Cladding, AlGaAs | 40 | 3.279 |
| GaAs | 22 | 3.524 |
| Photonic crystal | 118 | GaAs 3.524; air 1 |
| GaAs | 20 | 3.524 |
| p-cladding, AlGaAs | Halfspace | 3.307 |

The thesis lists 1500 nm for each outer cladding. Here each continues as a
homogeneous halfspace; no additional interface or mirror is placed at its end.
The averaged PC permittivity used in the vertical guide is
`0.84*3.524**2 + 0.16 = 10.59160384`.

The reference vacuum wavenumber is found by solving
`beta_fundamental(k0) = 2*pi/a`. This gives `k0 = 6.34040020253 µm^-1`, or
`lambda0 = 0.990976138174 µm`. It is the calculated Bragg reference, not a
measured lasing wavelength.

## Run the example

From the repository root:

```bash
python -m pip install -e ".[plot]"
python examples/liang_70um.py --N 96 --D 32 --q 64 --output results/liang-70um
python examples/plot_liang_70um.py results/liang-70um --output docs/images/liang-70um.png
```

The numerical settings for the figure are:

| Setting | Value |
| --- | --- |
| Cavity size | 70 × 70 µm² |
| Grid | 96 × 96 envelope cells; 36,864 free unknowns |
| Auxiliary cutoff | `D=32` |
| Vertical quadrature | `q=64` |
| Fourier control | `F=64`; circular-hole Fourier coefficients are analytic |
| Backend | `prepared` |
| Eigenpairs | One, near `sigma = -0.0953503743078 + 0.00357142857143j µm^-1` |
| ARPACK / residual tolerances | `1e-10` / `1e-9` |

This shift targets the single-lobed mode below the lowest-frequency periodic
band edge. The result is not a complete spectrum or a proof of the lowest
threshold across all branches.

The same calculation is available as a Python function:

```python
from examples.liang_70um import run_case

case = run_case(N=96, D=32, q=64, output="results/liang-70um-api")
mode = case.modes[0]
print(mode.delta * 70, mode.alpha * 70)
```

`output` is optional; without it, the objects remain in memory.
`build_structure()` returns `(period_um, eps, thickness_um, pc_layers)`.
`solve_reference_mode(period_um, eps, thickness_um)` returns the matched
vertical mode, `k0`, and root bracket. The [API guide](../api.md) shows how to
build the calculation directly with the package interfaces and change the
structure.

## Calculated result

| Quantity | Value |
| --- | ---: |
| Normalized detuning, `deltaL` | -6.64124431 |
| Normalized amplitude loss, `alphaL` | 0.22057143 |
| Amplitude loss, `alpha` | 0.00315102043 µm⁻¹ |
| Power-loss coefficient, `2*alpha` | 0.00630204086 µm⁻¹ |
| Outgoing edge contribution | 0.00424296052 µm⁻¹ |
| Radiative coupling contribution | 0.00205908034 µm⁻¹ |
| Generalized eigenpair residual | 1.67 × 10⁻¹⁸ |
| Relative power-balance error | 2.37 × 10⁻¹⁴ |

The field satisfies `hx*hy*sum(|Rx|²+|Sx|²+|Ry|²+|Sy|²)=1`. The image
shows this envelope intensity divided by its peak; it does not show the fast
unit-cell interference pattern or power in watts. The active material's
threshold gain also depends on its confinement factor and additional losses.

The measured calculation took 62.9 seconds on the recording machine with
one numerical thread: 62.6 seconds for coupling, 0.0082 seconds for sparse
assembly, and 0.217 seconds for the finite eigensolve. This excludes interpreter
startup, exporting files, and plotting. Versions and exact stage timings are
recorded in [summary.json](liang-70um/summary.json). These are timing samples,
not fixed runtime guarantees.

The numerical table is also available as [modes.csv](liang-70um/modes.csv).
The full complex arrays are generated locally as `fields.npz` rather than
committed to the repository.

## Numerical checks

The following checks keep the structure and cavity size fixed. The mesh rows
reuse the same `D32/q64` coupling matrix. They track the selected complex
four-component envelope by interpolation onto shared interior points of the
96 × 96 grid, rather than assuming solver indices identify the same mode.

| Envelope grid | `deltaL` | `alphaL` |
| --- | ---: | ---: |
| 32 × 32 | -6.64182901 | 0.22109496 |
| 64 × 64 | -6.64133563 | 0.22065313 |
| 96 × 96 | -6.64124431 | 0.22057143 |
| 128 × 128 | -6.64121235 | 0.22054285 |

The 96→128 change in `alphaL` is 0.0130%; the normalized complex field overlap
is above 0.999999999. Tightening the vertical quadrature from 48 to 64 at
`D=32` changes the coupling matrix by 1.22 × 10⁻¹⁵ relative Frobenius norm.

Auxiliary-harmonic refinement is slower:

| Cutoff (`q=48`, 64 × 64 grid) | `alphaL` of the selected mode |
| --- | ---: |
| D=8 | 0.22929719 |
| D=16 | 0.22375503 |
| D=24 | 0.22166010 |
| D=32 | 0.22065313 |

The D24→32 step changes the matrix by 0.509% and this mode's `alphaL` by
0.456%. It has not reached a 0.1% auxiliary-cutoff criterion. The extra digits
above identify reproducible numerical outputs, not that level of physical
accuracy. Raw controls, shifts, field overlaps, timings, and differences are
in [refinement.json](liang-70um/refinement.json). The cutoff rows used a broader
12-pair search; the selected D32/N64 result agrees with the focused one-pair
search to numerical roundoff.

Table 4.2 of the thesis gives `alphaL=0.23` for its CC band-edge mode A. The
present value is nearby, but the calculation uses a full layered Green function
and a different numerical discretization. It is not a point-by-point
reproduction of that table. Nor does this one-mode example classify all four
band-edge families or calculate their above-threshold competition.
