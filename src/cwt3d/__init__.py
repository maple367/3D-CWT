"""Independent 3D-CWT core.

NumPy/SciPy form the reference backend. Numba is optional and is neither
imported nor required by the core mathematical conventions.
"""

from . import conventions

__all__ = ["conventions"]
