"""Finite-domain staggered layout, sparse pencil, and targeted eigensolver."""
from .grid import StaggeredGrid, WaveFields
from .assembly import FinitePencil, assemble_pencil
from .eigensolver import (
    EigenResult, EigenSolveError, FactorizationError, solve_generalized, solve_finite,
)

__all__ = [
    'StaggeredGrid', 'WaveFields', 'FinitePencil', 'assemble_pencil',
    'EigenResult', 'EigenSolveError', 'FactorizationError',
    'solve_generalized', 'solve_finite',
]
