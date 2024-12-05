# 3D-CWT
 Three Dimension Coupled Wave Theory

# Requirements
- Python 3.12 (my environment)
    - numpy=1.26.4
    - scipy=1.14.0
    - matplotlib
    - mph
    - dill
- Comsol Multiphysics
## Intallation
```bash
conda env create -f environment.yml
```

# TODO
- [x] Add support for titled photonic crystal
- [x] Add support for finite size photonic crystal
- [ ] Add find_beta0 and find_k0 to class TMM

# Release Notes
## Version 1.2.0
 - Consider the symmetry of the fourier coefficients, so that the calculation of the fourier coefficients is faster.
## Version 1.0.0
 - Add support for finite size photonic crystal. Powered by [Comsol Multiphysics](https://www.comsol.com/).
## Version 0.4.1
 - Encapsulated into a function.
## Version 0.4.0
 - Add support for titled photonic crystal.
 - Add comsol semiconductor simulation.
## Version 0.3.0
 - Turn back to Python as numerical computing environment.
## Version 0.2.0
 - Use Matlab as numerical computing environment.



