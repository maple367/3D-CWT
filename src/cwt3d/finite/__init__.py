"""Finite-domain layout only; no PDE assembly or eigenvalue solver yet."""
from .grid import StaggeredGrid, WaveFields

__all__ = ['StaggeredGrid', 'WaveFields']
