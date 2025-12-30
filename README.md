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
## Installation
```bash
conda env create -f environment.yml
```

# TODO
- [x] Add support for titled photonic crystal
- [x] Add support for finite size photonic crystal
- [ ] Add find_beta0 and find_k0 to class TMM
- [ ] Pure python implementation

# Release Notes
## Latest - Version 2.0.0
 - Organize the code.
## Version 1.3.0
 - Use high accuracy Green's function.
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

# Citing 3D-CWT
If you use this code in your research, please cite the following papers, this will help me a lot.  
In BibTeX format:  
```bibtex
@misc{huang_Efficient_2025,
  title = {Towards Efficient {{PCSEL}} Design: A Fully {{AI-driven}} Approach},
  shorttitle = {Towards Efficient {{PCSEL}} Design},
  author = {Huang, Hai and Xu, Ziteng and Xin, Qi and Zhang, Zhaoyu},
  year = 2025,
  doi = {10.48550/ARXIV.2503.11022},
  copyright = {arXiv.org perpetual, non-exclusive license},
  langid = {english}
}

@article{huang_Unveiling_2025,
  title = {Unveiling the Potential of Photonic Crystal Surface Emitting Lasers: A Concise Review},
  shorttitle = {Unveiling the Potential of Photonic Crystal Surface Emitting Lasers},
  author = {Huang, Hai and Yang, Chuanyi and Li, Hui and Zhang, Zhaoyu},
  year = 2025,
  month = apr,
  journal = {Semiconductor Science and Technology},
  volume = {40},
  number = {4},
  pages = {43001},
  publisher = {IOP Publishing},
  issn = {0268-1242, 1361-6641},
  doi = {10.1088/1361-6641/adb7fd},
  langid = {english}
}
```
or in following IEEE format:  
[1] H. Huang, C. Yang, H. Li, and Z. Zhang, ‘Unveiling the potential of photonic crystal surface emitting lasers: a concise review’, Semicond. Sci. Technol., vol. 40, no. 4, p. 43001, Apr. 2025, doi: 10.1088/1361-6641/adb7fd.  
[2] H. Huang, Z. Xu, Q. Xin, and Z. Zhang, ‘Towards efficient PCSEL design: a fully AI-driven approach’, 2025. doi: 10.48550/ARXIV.2503.11022.



