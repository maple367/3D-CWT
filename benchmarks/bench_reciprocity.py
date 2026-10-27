"""Compare reciprocal overlap against the original two-kernel Duffy formula.
Run: .venv/Scripts/python.exe benchmarks/bench_reciprocity.py --output <directory>
No disk coupling cache. Same inputs, nodes, complex128 and solver controls.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import importlib.util
import platform
import sys
from pathlib import Path
from time import perf_counter

import json
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cwt3d.coupling import build_coupling
from cwt3d.geometry import CircleHole,RightTriangleHole,SampledPermittivity
from cwt3d.waveguide import TEWaveguide
from cwt3d.finite.assembly import FinitePencil
from cwt3d.finite.grid import StaggeredGrid
from cwt3d.finite.eigensolver import solve_finite
from cwt3d.finite.fields import reconstruct_modes
from cwt3d.cache import algorithm_digest


def match_error(a,b):
    i,j=linear_sum_assignment(abs(a[:,None]-b[None,:]))
    return float(max(abs(a[i]-b[j])))


def compute(mode,a,holes,N,L,D,q,F,*,reciprocity=True):
    start=perf_counter()
    c=build_coupling(mode,a,holes,cutoff=D,quadrature_order=q,fourier_order=F,backend='prepared',reciprocity=reciprocity)
    coupling=perf_counter()-start
    pencil=FinitePencil(StaggeredGrid(N,N,L,L),c)
    sigma=complex(min(np.linalg.eigvals(c.C).real)-.6/L,.25/L)
    eig=solve_finite(pencil,sigma=sigma,k=1,tol=1e-10,residual_tol=1e-9,seed=33,maxiter=2000)
    modes=reconstruct_modes(pencil,eig,coupling=c)
    return c,eig,modes,dict(coupling=coupling,total=perf_counter()-start)


def cases():
    spec=importlib.util.spec_from_file_location('liang_reciprocity_bench',ROOT/'examples/liang_70um.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    a,eps,widths,holes=m.build_structure()
    ref=m.solve_reference_mode(a,eps,widths)
    yield 'liang_CC',ref.mode,a,holes,64,70.,10,32,64
    guide=TEWaveguide([1.,8.,10.,2.25],[.13,.19],k0=6.)
    mode=guide.solve_mode();a=2*np.pi/mode.beta
    f=np.pi*.13**2
    first=CircleHole(.13*a,(a,a),(.36*a,.47*a),8+2*f,8-2*(1-f))
    f=.35*.27/2
    second=RightTriangleHole((.35*a,.27*a),(a,a),(.63*a,.54*a),10+2*f,10-2*(1-f))
    yield 'asymmetric_two_PC_circle_scalene_triangle',mode,a,{1:first,2:second},32,30.,6,20,32
    y,x=np.mgrid[:128,:128]/128
    mask=(((x-.31)**2+((y-.42)*1.27)**2<.13**2)|((x>.37)&(x<.69)&(y>.39)&(y<.48))|((x>.58)&(x<.67)&(y>.48)&(y<.63)))
    values=np.where(mask,1.,9.)
    guide=TEWaveguide([1.,float(values.mean()),2.25],[.25],k0=6.)
    mode=guide.solve_mode();a=2*np.pi/mode.beta
    yield 'asymmetric_sampled_bent_inclusion',mode,a,{1:SampledPermittivity(values,(a,a))},32,30.,6,20,32


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats<1:p.error('repeats must be positive')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,mode,a,holes,N,L,D,q,F in cases():
        samples=[]
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic warm-state bias.
            pair={}
            for label in (('original','reciprocal') if repeat%2==0 else ('reciprocal','original')):
                pair[label]=compute(mode,a,holes,N,L,D,q,F,reciprocity=label!='original')
            old,oe,om,ot=pair['original'];new,ne,nm,nt=pair['reciprocal']
            errors={key:float(np.max(abs(getattr(old,key)-getattr(new,key)))) for key in ('C1D','Crad','C2D','C2D_green','C2D_local','C','I','J','K','xi','mu','nu','varsigma')}
            for key in errors:
                np.testing.assert_allclose(getattr(old,key),getattr(new,key),rtol=5e-12,atol=2e-13)
            finite_error=match_error(oe.eigenvalues,ne.eigenvalues)
            assert finite_error < 1e-10
            i,j=linear_sum_assignment(abs(oe.eigenvalues[:,None]-ne.eigenvalues[None,:]))
            budgets={key:max(abs(float(om[u].budget[key])-float(nm[v].budget[key])) for u,v in zip(i,j)) for key in ('edge','coupling','threshold','balance_relative_error','Crad')}
            assert max(budgets.values()) < 1e-9
            samples.append(dict(original=ot,reciprocal=nt,array_max_errors=errors,infinite_eigenvalue_error=match_error(np.linalg.eigvals(old.C),np.linalg.eigvals(new.C)),finite_eigenvalue_error=finite_error,max_residual=float(max(max(oe.residuals),max(ne.residuals))),budget_errors=budgets))
        med={label:{key:float(np.median([s[label][key] for s in samples])) for key in ('coupling','total')} for label in ('original','reciprocal')}
        row=dict(case=name,controls=dict(N=N,L_um=L,D=D,q=q,F=F,k=1,pc_layers=list(holes),period_um=a,eps=list(np.asarray(mode.guide.eps).real),thickness_um=list(mode.guide.thickness_um)),medians=med,coupling_speedup=med['original']['coupling']/med['reciprocal']['coupling'],total_speedup=med['original']['total']/med['reciprocal']['total'],max_C_error=max(s['array_max_errors']['C'] for s in samples),max_finite_eigenvalue_error=max(s['finite_eigenvalue_error'] for s in samples),samples=samples)
        rows.append(row)
        report=dict(scope='Same-precision implementation parity, not physical/truncation convergence certification; original is native reciprocity=False, reciprocal is native reciprocity=True; no disk cache; total includes coupling, pencil, solve and field reconstruction, excludes guide setup/imports/exports',reciprocity_settings=dict(original=False,reciprocal=True),environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,precision='complex128',threads=1,algorithm_digest=algorithm_digest()),rows=rows)
        (output/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({key:row[key] for key in ('case','medians','coupling_speedup','total_speedup','max_C_error','max_finite_eigenvalue_error')}),flush=True)
    assert len(rows)==3


if __name__=='__main__':main()
