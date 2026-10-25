"""Liang chapter 4 circular-hole finite-domain example (lengths in um).

Install cwt3d first; this script also runs with python -I outside the checkout.
The full-layer scalar TE Green model differs from the thesis approximations.
A single shift returns a nearby subset, not the complete or lowest-loss spectrum.
"""
import argparse
import csv
from dataclasses import dataclass
from importlib.metadata import version
import json
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import brentq

from cwt3d.cache import algorithm_digest
from cwt3d.coupling import build_coupling
from cwt3d.finite.assembly import FinitePencil
from cwt3d.finite.eigensolver import EigenResult, solve_finite
from cwt3d.finite.fields import reconstruct_modes
from cwt3d.finite.grid import StaggeredGrid
from cwt3d.geometry import CircleHole
from cwt3d.waveguide import NoGuidedModeError, TEWaveguide


@dataclass(frozen=True)
class ReferenceMode:
    """Fundamental TE mode at the fixed-period Bragg reference frequency."""
    mode: object
    k0: float
    bracket: tuple


@dataclass(frozen=True)
class CaseResult:
    """Reusable numerical objects; eigenpairs and modes share ascending alpha order."""
    reference: ReferenceMode
    coupling: object
    pencil: FinitePencil
    eigenpairs: EigenResult
    modes: tuple
    summary: dict


def build_structure():
    """Return (period_um, epsilon, internal_thickness_um, PC-layer mapping).

    Table 4.1, CC: replace the two homogeneous 1.5 um outer claddings by
    halfspaces of the same epsilon. PC-layer indices include the lower halfspace.
    """
    period = .295
    hole = CircleHole(period * np.sqrt(.16 / np.pi), (period, period),
                      (period / 2, period / 2), 3.524**2, 1.)
    eps = (3.307**2, 3.524**2, 3.553**2, 3.524**2, 3.279**2,
           3.524**2, hole.mean_epsilon.real, 3.524**2, 3.307**2)
    thickness = (.030, .064, .040, .040, .022, .118, .020)
    return period, eps, thickness, {6: hole}


def solve_reference_mode(period_um, eps, thickness_um):
    """Solve beta_fundamental(k0)=2*pi/a with an automatic physical bracket.

    The search interval follows beta/sqrt(max(eps)) < k0 < beta/n_clad.
    If a sampled frequency has no bound mode, skip it rather than assigning
    a fabricated propagation constant. Brent's method uses only guided modes.
    """
    beta0 = 2 * np.pi / period_um
    lower = beta0 / np.sqrt(max(eps)) * (1 + 1e-9)
    upper = beta0 / np.sqrt(max(eps[0], eps[-1])) * (1 - 1e-9)

    def mode_at(k0):
        return TEWaveguide(eps, thickness_um, k0=k0, period_um=period_um).solve_mode()

    def mismatch(k0):
        return mode_at(k0).beta - beta0

    previous = None
    bracket = None
    for point in np.linspace(lower, upper, 65):
        try:
            value = mismatch(point)
        except NoGuidedModeError:
            continue
        if previous is not None and previous[1] * value <= 0:
            bracket = (float(previous[0]), float(point))
            break
        previous = (point, value)
    if bracket is None:
        raise RuntimeError('No guided-mode Bragg root bracket found')
    k0 = brentq(mismatch, *bracket, xtol=1e-13, rtol=1e-14)
    return ReferenceMode(mode_at(k0), float(k0), bracket)


def _check_output(output):
    if output is None:
        return
    path = Path(output)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('nonempty output or non-directory: choose a fresh directory')


def _validate(N, D, q, F, k, L_um, sigma, backend):
    for name, value in dict(N=N, D=D, q=q, F=F, k=k).items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if k >= 4 * N * N - 1:
        raise ValueError('k must be less than n_dof-1')
    if isinstance(L_um, (bool, np.bool_)) or not isinstance(L_um, Real) or not np.isfinite(L_um) or L_um <= 0:
        raise ValueError('L_um must be positive and finite')
    if sigma is not None and (isinstance(sigma, (bool, np.bool_)) or not np.isscalar(sigma) or not np.isfinite(sigma)):
        raise ValueError('sigma must be a finite complex scalar')
    if backend not in ('reference', 'prepared'):
        raise ValueError('backend must be reference or prepared')


def _json_value(value):
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def run_case(*, L_um=70, N=64, D=16, q=48, F=64, k=1, sigma=None,
             backend='prepared', output=None):
    """Compute one square domain; optionally export into a fresh directory.

    N is cells per side, D auxiliary cutoff, q vertical quadrature, F shape
    Fourier quadrature, k requested eigenpairs. sigma is delta+i*alpha in um^-1.
    No output is written by default. The returned pencil/coupling can be reused.
    """
    _validate(N, D, q, F, k, L_um, sigma, backend)
    _check_output(output)
    started = perf_counter()
    period, eps, thickness, layers = build_structure()
    reference = solve_reference_mode(period, eps, thickness)
    reference_seconds = perf_counter() - started
    stage = perf_counter()
    coupling = build_coupling(reference.mode, period, layers, cutoff=D,
                              quadrature_order=q, fourier_order=F, backend=backend)
    coupling_seconds = perf_counter() - stage
    infinite_values = np.linalg.eigvals(coupling.C)
    automatic_shift = sigma is None
    if automatic_shift:
        sigma = complex(float(np.min(infinite_values.real)) - .6 / L_um, .25 / L_um)
    else:
        sigma = complex(sigma)
    stage = perf_counter()
    pencil = FinitePencil(StaggeredGrid(N, N, L_um, L_um), coupling)
    assembly_seconds = perf_counter() - stage
    stage = perf_counter()
    raw = solve_finite(pencil, sigma=sigma, k=k, tol=1e-10,
                       residual_tol=1e-9, seed=33, maxiter=10000)
    finite_seconds = perf_counter() - stage
    order = np.argsort(raw.eigenvalues.imag, kind='stable')
    eigenpairs = EigenResult(raw.eigenvalues[order], raw.eigenvectors[:, order],
                            raw.residuals[order], dict(raw.metadata))
    stage = perf_counter()
    modes = reconstruct_modes(pencil, eigenpairs, coupling=coupling)
    reconstruction_seconds = perf_counter() - stage
    rows = [dict(index=i, delta_um_inverse=m.delta, alpha_um_inverse=m.alpha,
                 deltaL=m.delta * L_um, alphaL=m.alpha * L_um,
                 residual=m.residual, boundary_error=m.boundary_error,
                 reconstruction_error=m.reconstruction_error, **dict(m.budget))
            for i, m in enumerate(modes)]
    hole = layers[6]
    summary = _json_value(dict(
        schema='cwt3d-liang-70um-v1', status=eigenpairs.status,
        input=dict(L_um=L_um, N=N, D=D, q=q, F=F, k=k, backend=backend),
        source=dict(doi='10.14989/doctor.k18283', chapter=4, table='4.1', geometry='CC circular hole',
                    domain=f'{L_um:g} um square; thesis chapter-4 reference is 70 um',
                    cladding_model='two homogeneous 1.5 um outer claddings replaced by halfspaces',
                    removed_cladding_thickness_um=[1.5, 1.5],
                    approximation='scalar TE full-layer Green; not an exact thesis reproduction',
                    thesis_reference_alphaL=[.23, .52, 2.18, 2.18],
                    reference_scope='literature comparison only, not computed acceptance targets'),
        structure=dict(period_um=period, eps=eps, thickness_um=thickness, pc_layers=[6],
                       radius_um=hole.radius_um, center_um=hole.center_um,
                       fill_fraction=hole.fill_fraction, eps_background=hole.eps_background.real,
                       eps_hole=hole.eps_hole.real),
        reference=dict(k0_um_inverse=reference.k0, wavelength_um=2*np.pi/reference.k0,
                       beta_um_inverse=reference.mode.beta, beta0_um_inverse=coupling.beta0,
                       root_bracket_um_inverse=reference.bracket, root_method='Brent',
                       beta_relative_mismatch=coupling.relative_detuning,
                       boundary_residual=reference.mode.boundary_residual),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         cwt3d=version('cwt3d'), algorithm_digest=algorithm_digest(), precision='complex128',
                         threads={name: os.environ.get(name) for name in (
                             'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')}),
        shift_um_inverse=sigma,
        shift_rule=('minimum real infinite-C band edge - 0.6/L + i*0.25/L' if automatic_shift else 'user supplied'),
        infinite_eigenvalues_um_inverse=infinite_values.tolist(),
        solver=dict(eigenpairs.metadata), ordering='ascending signed Im(lambda), stable ties',
        spectrum_scope='single-shift targeted subset only; not all spectrum or certified lowest threshold',
        normalization=modes[0].normalization, units=dict(modes[0].units), modes=rows,
        timings_seconds=dict(reference=reference_seconds, coupling=coupling_seconds,
                             assembly=assembly_seconds, finite_solve=finite_seconds,
                             lu=raw.lu_seconds, iteration=raw.iteration_seconds,
                             reconstruction=reconstruction_seconds, compute_total=perf_counter()-started),
        cache='disabled'))
    json.dumps(summary, allow_nan=False)
    result = CaseResult(reference, coupling, pencil, eigenpairs, modes, summary)
    if output is not None:
        export_case(result, output)
    return result


def export_case(result, output):
    """Export matching JSON/CSV/NPZ rows; refuse existing nonempty output."""
    _check_output(output)
    text = json.dumps(result.summary, indent=2, allow_nan=False) + '\n'
    modes, coupling = result.modes, result.coupling
    arrays = {name: getattr(coupling, name) for name in ('C', 'C1D', 'Crad', 'C2D')}
    arrays.update(eigenvalues=result.eigenpairs.eigenvalues,
                  eigenvectors=result.eigenpairs.eigenvectors, residuals=result.eigenpairs.residuals,
                  centers=np.array([[getattr(m.centers, n) for n in ('Rx', 'Sx', 'Ry', 'Sy')] for m in modes]))
    for name in ('Rx', 'Sx', 'Ry', 'Sy'):
        arrays['faces_' + name] = np.array([getattr(m.faces, name) for m in modes])
    grid = result.pencil.grid
    arrays.update(x_centers_um=(np.arange(grid.Nx)+.5)*grid.hx_um,
                  y_centers_um=(np.arange(grid.Ny)+.5)*grid.hy_um,
                  x_faces_um=np.arange(grid.Nx+1)*grid.hx_um,
                  y_faces_um=np.arange(grid.Ny+1)*grid.hy_um)
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError('nonfinite numerical export')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'fields.npz', **arrays)
    with (output / 'modes.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.summary['modes'][0]))
        writer.writeheader()
        writer.writerows(result.summary['modes'])
    # Publish the manifest only after both numerical files have been written.
    (output / 'summary.json').write_text(text, encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (('N', 64), ('D', 16), ('q', 48), ('F', 64), ('k', 1)):
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--L-um', type=float, default=70.)
    p.add_argument('--sigma', type=complex, default=None,
                   help='delta+i*alpha in um^-1; default: minimum-real infinite-C edge - 0.6/L + i*0.25/L')
    p.add_argument('--backend', choices=('reference', 'prepared'), default='prepared')
    p.add_argument('--output', type=Path, default=Path('results/liang-70um'))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = run_case(L_um=args.L_um, N=args.N, D=args.D, q=args.q, F=args.F,
                      k=args.k, sigma=args.sigma, backend=args.backend, output=args.output)
    print(f"Reference wavelength: {result.summary['reference']['wavelength_um']:.9f} um")
    print('Modes near the selected shift (ordered by alpha):')
    print('index       deltaL       alphaL       residual')
    for row in result.summary['modes']:
        print(f"{row['index']:5d} {row['deltaL']:12.6g} {row['alphaL']:12.6g} {row['residual']:12.3g}")
    print(json.dumps(dict(status=result.eigenpairs.status, returned=len(result.modes),
                         output=str(args.output)), allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
