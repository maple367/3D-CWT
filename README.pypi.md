# cwt3d

A Python implementation of three-dimensional coupled-wave theory for
square-lattice photonic-crystal surface-emitting lasers (PCSELs).

The package calculates normalized vertical TE waveguide modes, Fourier
coupling coefficients, and finite-cavity envelopes using sparse staggered-grid
eigenproblems. Version1.4.0 also provides `cwt3d.active`: frozen gain/index
analysis and deterministic spatial four-wave/carrier dynamics. Runtime
dependencies are NumPy and SciPy.

## Install

```bash
python -m pip install cwt3d
```

Matplotlib is optional:

```bash
python -m pip install "cwt3d[plot]"
```

## Examples and Python interfaces

The [source repository](https://github.com/maple367/3D-CWT) includes a complete
70 × 70 µm² circular-hole example based on Chapter 4 of Yong Liang's thesis.
It contains the layer stack, runnable calculation and plotting scripts, and
numerical refinement results.

![70 µm PCSEL envelope](https://raw.githubusercontent.com/maple367/3D-CWT/latest/docs/images/liang-70um.png)

The [Python API guide](https://github.com/maple367/3D-CWT/blob/latest/docs/api.md)
shows how to construct a different geometry, calculate its coupling matrix,
solve a finite cavity, and access the complex fields. Examples are included
in the source distribution and repository; the wheel installs the `cwt3d`
library.

## Active cavities

`cwt3d.active` exposes `CarrierMaterial`, `ActiveCavity`, `ActiveState`,
`simulate`, `solve_active_modes`, and `uniform_threshold_gain`. It retains
the passive spatial operator and supports signed gain/absorption, gain
compression, carrier-induced index changes, injection, ABC recombination
and lateral carrier diffusion. Photon and carrier normalization is explicit.

The [active-cavity guide](https://github.com/maple367/3D-CWT/blob/latest/docs/active-cavity.md)
describes units, physics and APIs. A runnable `examples/active_cavity.py` and
`scripts/plot_active_trajectory.py` are shipped in the source distribution.
Their active parameters are illustrative, not measured device fits. A nonzero
optical seed is required for nonzero deterministic emission; noise, linewidth,
thermal transport and full electrical drift-diffusion are outside the model.

## Scope

Reciprocal Green-value reuse is enabled by default. Pass `reciprocity=False`
to `build_coupling(...)` to use the original two-evaluation formula. The
repository examples also accept `--no-reciprocity`. Both settings keep the
same precision and integration orders; selected settings are recorded and
separated in the optional cache. The optimization does not require geometric
symmetry within the supported scalar TE model. See the
[validation and benchmark notes](https://github.com/maple367/3D-CWT/blob/latest/docs/validation/reciprocal-green-overlap/README.md).

The passive coupling builder uses lossless scalar TE reference modes on a
square lattice. Active gain and index feedback are local perturbations of
that fixed vertical reference, not a full-vector or strong-index-rebuilt model.
Eigenanalysis returns modes near a selected complex shift, not the complete
cavity spectrum. The example's auxiliary-harmonic refinement is still finite,
as reported in its notes. Time stepping does not certify spatial convergence,
experimental output or stable single-frequency CW operation. The historical
full electrical transport solver is not included.

The code is distributed under the MIT license. The formulation follows
[Yong Liang's doctoral thesis](https://doi.org/10.14989/doctor.k18283);
references and conventions are listed in the repository documentation.
