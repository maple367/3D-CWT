"""Pure Python finite-domain optical example. All lengths are micrometers.

Run after installing cwt3d. No legacy imports, GPU, COMSOL, or implicit cache.
"""
import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np
import scipy
from scipy.optimize import brentq
from cwt3d.geometry import RightTriangleHole
from cwt3d.waveguide import TEWaveguide
from cwt3d.coupling import build_coupling
from cwt3d.cache import algorithm_digest
from cwt3d.finite.grid import StaggeredGrid
from cwt3d.finite.assembly import FinitePencil
from cwt3d.finite.eigensolver import solve_finite
from cwt3d.finite.fields import reconstruct_modes


def physical():
    """Liang thesis Table 3.1; remove redundant identical cladding segments."""
    a=.295
    leg=a*np.sqrt(2*.16)
    hole=RightTriangleHole((leg,leg),(a,a),(a/2,a/2),12.7449,1.)
    eps=[11.0224,12.8603,hole.mean_epsilon.real,12.7449,11.0224]
    thickness=[.0885,.1180,.0590]
    beta0=2*np.pi/a
    def guide(k0):
        return TEWaveguide(eps,thickness,k0=k0,period_um=a)
    bracket=(beta0/np.sqrt(max(eps))*(1+1e-7),beta0/np.sqrt(11.0224)*(1-1e-7))
    k0=brentq(lambda k:guide(k).solve_mode().beta-beta0,*bracket,xtol=1e-13)
    mode=guide(k0).solve_mode()
    return mode,a,{2:hole},dict(source='Yong Liang thesis DOI 10.14989/doctor.k18283 Table 3.1 (printed p32/PDF36)',
        a_um=a,fill_fraction=.16,eps_air=1.,eps_bulk=12.7449,
        eps=eps,thickness_um=thickness,removed_cladding_thickness_um=[1.5,1.5],
        triangle_legs_um=[leg,leg],triangle_centroid_um=[a/2,a/2],
        k0_um_inverse=k0,wavelength_um=2*np.pi/k0,beta_um_inverse=mode.beta,
        beta0_um_inverse=beta0,root_bracket_um_inverse=list(bracket),
        beta_relative_mismatch=(mode.beta-beta0)/beta0,boundary_residual=mode.boundary_residual)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=['demo','liang'],default='demo',help='same literature structure; liang defaults D10/q24/N16')
    for name,default in [('Nx',None),('Ny',None),('D',None),('q',None),('F',64),('k',8)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--L-um',type=float,default=14.75,help='physical square domain length, chosen 50a; not a verified thesis domain')
    p.add_argument('--sigma',type=complex,default=.03+.03j,help='delta+i*alpha target shift in um^-1 (not frequency)')
    p.add_argument('--backend',choices=['reference','prepared'],default='prepared')
    p.add_argument('--output',type=Path,default=Path('results/v3-demo'))
    p.add_argument('--overwrite',action='store_true',help='replace only the four named output files; requires existing v3 summary')
    return p


def resolve(args):
    for key,small,large in [('Nx',8,16),('Ny',8,16),('D',3,10),('q',12,24)]:
        if getattr(args,key) is None:setattr(args,key,large if args.preset=='liang' else small)
    if any(getattr(args,n)<1 for n in ('Nx','Ny','D','q','F','k')):
        raise ValueError('mesh, orders and k must be positive')
    if not np.isfinite(args.L_um) or args.L_um<=0 or not np.isfinite(args.sigma):
        raise ValueError('L and sigma must be finite; L positive')
    if args.k>=4*args.Nx*args.Ny-1:raise ValueError('k must be less than n_dof-1')
    return args


def compute(args,coupling=None,source=None):
    start=perf_counter()
    if coupling is None:
        mode,a,holes,source=physical()
        coupling=build_coupling(mode,a,holes,cutoff=args.D,quadrature_order=args.q,fourier_order=args.F,backend=args.backend)
    coupling_seconds=perf_counter()-start
    g=StaggeredGrid(args.Nx,args.Ny,args.L_um,args.L_um)
    p=FinitePencil(g,coupling)
    result=solve_finite(p,sigma=args.sigma,k=args.k,tol=1e-10,residual_tol=1e-9,seed=33,maxiter=10000)
    modes=reconstruct_modes(p,result,coupling=coupling)
    rows=[dict(index=i,delta_um_inverse=m.delta,alpha_um_inverse=m.alpha,residual=m.residual,
        boundary_error=m.boundary_error,**dict(m.budget)) for i,m in enumerate(modes)]
    summary=dict(schema='cwt3d-demo-v1',status=result.status,input={**vars(args),'output':str(args.output),'sigma':[args.sigma.real,args.sigma.imag]},
        source=source,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,cwt3d=version('cwt3d'),source_digest=algorithm_digest(),precision='complex128'),
        normalization=modes[0].normalization,units=dict(modes[0].units),modes=rows,
        cache='disabled (no disk reads/writes)',solver={k:([v.real,v.imag] if isinstance(v,complex) else v) for k,v in result.metadata.items()},spectrum_scope='targeted subset only; not a lowest-threshold/all-spectrum certificate',
        coupling_seconds=coupling_seconds,lu_seconds=result.lu_seconds,iteration_seconds=result.iteration_seconds)
    return summary,modes,coupling


def export(output,summary,modes,coupling):
    output.mkdir(parents=True,exist_ok=True)
    arrays={'C':coupling.C,'C1D':coupling.C1D,'Crad':coupling.Crad,'C2D':coupling.C2D,
        'eigenvalues':np.array([m.eigenvalue for m in modes]),
        'centers':np.array([[getattr(m.centers,n) for n in ('Rx','Sx','Ry','Sy')] for m in modes])}
    for n in ('Rx','Sx','Ry','Sy'):arrays['faces_'+n]=np.array([getattr(m.faces,n) for m in modes])
    g=modes[0].grid
    arrays.update(x_centers_um=(np.arange(g.Nx)+.5)*g.hx_um,y_centers_um=(np.arange(g.Ny)+.5)*g.hy_um,
        x_faces_um=np.arange(g.Nx+1)*g.hx_um,y_faces_um=np.arange(g.Ny+1)*g.hy_um)
    np.savez_compressed(output/'fields.npz',**arrays)
    with (output/'modes.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary['modes'][0]));w.writeheader();w.writerows(summary['modes'])
    intensity=modes[0].intensity
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="680" viewBox="0 0 640 680">',
        '<title>Mode 0: incoherent envelope intensity, normalized P=1; um^-2</title>']
    for y in range(g.Ny):
        for x in range(g.Nx):
            v=int(255*intensity[y,x]/intensity.max())
            parts.append(f'<rect x="{x*640/g.Nx}" y="{(g.Ny-1-y)*640/g.Ny}" width="{640/g.Nx+0.1}" height="{640/g.Ny+0.1}" fill="rgb({v},0,{255-v})"/>')
    parts.append(f'<text x="8" y="665">0..{g.Lx_um:g} um; intensity max={intensity.max():.6g} um^-2</text></svg>')
    (output/'mode0.svg').write_text('\n'.join(parts),encoding='utf-8')
    # Write success manifest last, never on a failed solve.
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    args=resolve(parser().parse_args())
    if args.output.exists() and any(args.output.iterdir()):
        marker=args.output/'summary.json'
        if not args.overwrite or not marker.exists() or json.loads(marker.read_text()) .get('schema')!='cwt3d-demo-v1':
            raise ValueError('nonempty output: choose a fresh directory or --overwrite for a prior v3 demo')
    summary,modes,coupling=compute(args)
    export(args.output,summary,modes,coupling)
    print(json.dumps(dict(status=summary['status'],returned=len(modes),output=str(args.output))))

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(1)
