"""Recompute literature-parameter convergence axes; analysis helper, not tracker API."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linear_sum_assignment

spec=importlib.util.spec_from_file_location('demo',Path(__file__).resolve().parents[1]/'examples/square_finite.py')
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)


def match(previous,current):
    """Common interior samples; assignment has no claim to handle degenerate subspaces."""
    xy=np.linspace(.1,.9,19)*previous[0].grid.Lx_um
    yy,xx=np.meshgrid(xy,xy,indexing='ij');points=np.column_stack((yy.ravel(),xx.ravel()))
    def vectors(modes):
        out=[]
        for m in modes:
            g=m.grid
            axes=((np.arange(g.Ny)+.5)*g.hy_um,(np.arange(g.Nx)+.5)*g.hx_um)
            v=np.concatenate([RegularGridInterpolator(axes,getattr(m.centers,n),bounds_error=True)(points) for n in ('Rx','Sx','Ry','Sy')])
            out.append(v/np.linalg.norm(v))
        return np.array(out)
    a,b=vectors(previous),vectors(current)
    overlap=abs(a.conj()@b.T)
    i,j=linear_sum_assignment(-overlap)
    rows=[]
    for x,y in zip(i,j):
        alternatives=np.delete(overlap[x],y)
        gap=float(overlap[x,y]-max(alternatives,default=0))
        # Intentionally conservative: tied/clustered candidates remain ambiguous.
        ambiguous=bool(overlap[x,y]<.8 or gap<.1 or any(abs(current[y].eigenvalue-m.eigenvalue)<1e-7 for z,m in enumerate(current) if z!=y))
        rows.append(dict(previous_index=int(x),current_index=int(y),overlap=float(overlap[x,y]),row_gap=gap,
            ambiguous=ambiguous,delta_lambda_abs_um_inverse=float(abs(previous[x].eigenvalue-current[y].eigenvalue))))
    return rows


def encoded(a):
    a=np.asarray(a)
    return np.stack((a.real,a.imag),axis=-1).tolist()


def coupling_comparison(old,new):
    """Assignment of four infinite-domain eigenvalues, not finite-grid modes."""
    result={}
    for name in ('C','C1D','Crad','C2D'):
        a,b=np.asarray(old[name]),np.asarray(new[name])
        delta=b-a
        scale=float(np.linalg.norm(b))
        result[name]=dict(delta_norm=float(np.linalg.norm(delta)),
            relative_norm=float(np.linalg.norm(delta)/scale) if scale else None,
            delta_entries=encoded(delta),max_entry_absolute=float(np.max(abs(delta))))
    a,b=np.linalg.eigvals(old['C']),np.linalg.eigvals(new['C'])
    i,j=linear_sum_assignment(abs(a[:,None]-b[None,:]))
    delta=b[j]-a[i]
    centered=(b[j]-b.mean())-(a[i]-a.mean())
    scale=float(np.linalg.norm(b))
    alpha_scale=float(np.linalg.norm(b.imag))
    result['infinite_eigenset']=dict(previous_indices=i.tolist(),current_indices=j.tolist(),
        delta=encoded(delta),relative_norm=float(np.linalg.norm(delta)/scale),
        max_absolute=float(np.max(abs(delta))),common_shift=encoded(b.mean()-a.mean()),
        centered_delta=encoded(centered),centered_max_absolute=float(np.max(abs(centered))),
        centered_relative_norm=float(np.linalg.norm(centered)/np.linalg.norm(b-b.mean())),
        alpha_delta=(b[j].imag-a[i].imag).tolist(),
        alpha_max_absolute=float(np.max(abs(b[j].imag-a[i].imag))),
        alpha_relative_norm=float(np.linalg.norm(b[j].imag-a[i].imag)/alpha_scale) if alpha_scale else None)
    return result


def extended(output,orders,q,F,refined_q,refined_F):
    from time import perf_counter
    import platform
    import scipy
    from cwt3d.cache import algorithm_digest
    paths=[output/'extended_coupling.json',output/'extended_coupling.csv']
    if any(p.exists() for p in paths):raise ValueError('extended report already exists; choose fresh output')
    output.mkdir(parents=True,exist_ok=True)
    mode,a,holes,source=demo.physical()
    cases=[('D',d,q,F) for d in orders]+[('q',orders[-1],refined_q,F),('F',orders[-1],refined_q,refined_F)]
    rows=[];arrays=[]
    for axis,d,nq,nf in cases:
        start=perf_counter()
        c=demo.build_coupling(mode,a,holes,cutoff=d,quadrature_order=nq,fourier_order=nf,backend='prepared')
        seconds=perf_counter()-start
        current={name:np.asarray(getattr(c,name)) for name in ('C','C1D','Crad','C2D')}
        eigenvalues=np.linalg.eigvals(c.C)
        row=dict(run=len(rows),axis=axis,D=d,q=nq,F=nf,seconds=seconds,
            normC=float(np.linalg.norm(c.C)),matrices={k:encoded(v) for k,v in current.items()},
            infinite_eigenvalues=encoded(eigenvalues),alpha=eigenvalues.imag.tolist(),
            comparison_previous=coupling_comparison(arrays[-1],current) if arrays else None,
            comparison_same_D_base=coupling_comparison(arrays[len(orders)-1],current) if axis!='D' else None)
        arrays.append(current);rows.append(row)
        report=dict(schema='cwt3d-extended-coupling-v1',source=source,
            environment=dict(source_digest=algorithm_digest(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,threads=1,backend='prepared'),
            rows=rows,thresholds=dict(matrix_relative_change=1e-3),
            limitations='Coupling-only; four infinite-domain eigenvalues, alpha=Im(lambda) in um^-1. No finite eigensolves. Successive q then F controls separate errors locally at highest D; not an exact-reference error bound. Adjacent D stability is not a tail bound. No publication curve certification.')
        paths[0].write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        with paths[1].open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
            w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
        print(json.dumps({k:row[k] for k in ('axis','D','q','F','seconds','normC')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extended-coupling-only',action='store_true',help='independent high-D report; leaves all old scans untouched')
    p.add_argument('--extended-D',type=int,nargs='+',default=[10,16,24,32])
    p.add_argument('--extended-q',type=int,default=48);p.add_argument('--extended-F',type=int,default=128)
    p.add_argument('--refined-q',type=int,default=64);p.add_argument('--refined-F',type=int,default=192)
    args=p.parse_args()
    if args.extended_coupling_only:
        if min(*args.extended_D,args.extended_q,args.extended_F,args.refined_q,args.refined_F)<1:raise ValueError('orders must be positive')
        if args.extended_D!=sorted(set(args.extended_D)):raise ValueError('D orders must strictly increase')
        if args.refined_q<=args.extended_q or args.refined_F<=args.extended_F:raise ValueError('refined orders must increase')
        return extended(args.output,args.extended_D,args.extended_q,args.extended_F,args.refined_q,args.refined_F)
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('output must be fresh')
    args.output.mkdir(parents=True,exist_ok=True)
    mode,a,holes,source=demo.physical()
    rows=[];couplings={};base=dict(Nx=16,Ny=16,D=10,q=24,F=64,k=12,L_um=14.75,sigma=.03+.03j,backend='prepared',preset='liang',overwrite=False)
    previous={}
    scans=[('D',v) for v in (1,3,6,8,10)]+[('q',v) for v in (8,12,18,24)]+[('mesh',v) for v in (8,16,32,64)]+[('shift',v) for v in (.03+.03j,.04+.03j)]
    for index,(axis,value) in enumerate(scans):
        config=dict(base)
        if axis=='mesh':config.update(Nx=value,Ny=value)
        elif axis=='shift':config['sigma']=value
        else:config[axis]=value
        config['output']=args.output/f'{index:02d}-{axis}'
        key=(config['D'],config['q'])
        if key not in couplings:
            couplings[key]=demo.build_coupling(mode,a,holes,cutoff=key[0],quadrature_order=key[1],fourier_order=64,backend='prepared')
        c=couplings[key]
        summary,modes,_=demo.compute(argparse.Namespace(**config),c,source)
        demo.export(config['output'],summary,modes,c)
        row=dict(run=index,axis=axis,value=[value.real,value.imag] if isinstance(value,complex) else value,
            input=summary['input'],status=summary['status'],returned=len(modes),
            eigenvalues=[[m.delta,m.alpha] for m in modes],max_residual=max(m.residual for m in modes),
            max_closure=max(m.budget['balance_relative_error'] for m in modes),coupling_norm=float(np.linalg.norm(c.C)),
            comparisons=[],coupling_relative_change=None)
        if axis in previous:
            oldc,oldm=previous[axis]
            row['coupling_relative_change']=float(np.linalg.norm(c.C-oldc.C)/np.linalg.norm(c.C))
            row['comparisons']=match(oldm,modes)
        previous[axis]=(c,modes);rows.append(row)
        report=dict(schema='cwt3d-convergence-v1',source=source,environment=summary['environment'],rows=rows,
            thresholds=dict(coupling_relative_change=1e-3,matched_lambda_absolute_um_inverse=1e-4,minimum_overlap=.8),
            limitations='Fixed L and coupling per mesh/shift axis. Target subsets, not full spectrum. Interpolated analysis assignment; ambiguous pairs are retained, not certified. No published curve data available.')
        (args.output/'convergence.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        with (args.output/'convergence.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
        print(json.dumps({k:row[k] for k in ('axis','value','returned','max_residual','max_closure','coupling_relative_change')}),flush=True)
    assert len(rows)==15

if __name__=='__main__':main()
