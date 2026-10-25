"""Finite-domain staggered layout and direct sparse pencil (no eigensolver)."""
from .grid import StaggeredGrid, WaveFields
from .assembly import FinitePencil, assemble_pencil

__all__ = ['StaggeredGrid', 'WaveFields', 'FinitePencil', 'assemble_pencil']
