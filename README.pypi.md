# cwt3d

A Python implementation of three-dimensional coupled-wave theory for
square-lattice photonic-crystal surface-emitting lasers (PCSELs).

The package calculates normalized vertical TE waveguide modes, Fourier
coupling coefficients, and finite-cavity envelopes using sparse staggered-grid
eigenproblems. Runtime dependencies are NumPy and SciPy.

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

## Scope

The current model uses lossless scalar TE waves on a square lattice.
It returns modes near a selected complex shift, not the complete cavity
spectrum. The example's auxiliary-harmonic refinement is still finite, as
reported in its notes. Carrier transport and above-threshold dynamics from
the historical application are not part of this package.

The code is distributed under the MIT license. The formulation follows
[Yong Liang's doctoral thesis](https://doi.org/10.14989/doctor.k18283);
references and conventions are listed in the repository documentation.
