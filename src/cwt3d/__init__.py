"""Independent 3D-CWT core; importing it never loads legacy/COMSOL backends.

NumPy/SciPy form the reference backend. Numba is optional and is neither
imported nor required by the core mathematical conventions.
"""

from . import conventions

__all__ = ["conventions"]
