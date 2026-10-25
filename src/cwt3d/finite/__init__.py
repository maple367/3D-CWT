"""Finite-domain staggered layout, sparse pencil, and targeted eigensolver."""
from .grid import StaggeredGrid, WaveFields
from .assembly import FinitePencil, assemble_pencil
from .eigensolver import (
    EigenResult, EigenSolveError, FactorizationError, solve_generalized, solve_finite,
)
from .fields import FiniteMode, reconstruct_field, reconstruct_mode, reconstruct_modes
from .tracking import ClusterMatch, ModeTracking, track_modes

__all__ = [
    'StaggeredGrid', 'WaveFields', 'FinitePencil', 'assemble_pencil',
    'EigenResult', 'EigenSolveError', 'FactorizationError',
    'solve_generalized', 'solve_finite',
    'FiniteMode', 'reconstruct_field', 'reconstruct_mode', 'reconstruct_modes',
    'ClusterMatch', 'ModeTracking', 'track_modes',
]
