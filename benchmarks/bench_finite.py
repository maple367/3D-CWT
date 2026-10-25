"""Sparse finite-domain scaling; no dense or legacy mathematical baseline."""
from common import *
import argparse
import tracemalloc
from cwt3d.coupling import build_coupling
from cwt3d.finite.grid import StaggeredGrid
from cwt3d.finite.assembly import FinitePencil
from cwt3d.finite.eigensolver import solve_finite


def run(grid,coupling):
    start=perf_counter(); pencil=FinitePencil(grid,coupling); assembly=perf_counter()-start
    start=perf_counter()
    eigen=solve_finite(pencil,sigma=.03+.03j,k=6,tol=1e-9,residual_tol=1e-9,seed=33,ncv=30,maxiter=5000)
    solve=perf_counter()-start
    a,b=pencil.A,pencil.B
    shifted=(a-(.03+.03j)*b).tocsc()
    return dict(assembly_seconds=assembly,solve_total_seconds=solve,
                lu_seconds=eigen.lu_seconds,iteration_seconds=eigen.iteration_seconds,
                max_residual=float(max(eigen.residuals)),nnz_A=a.nnz,nnz_B=b.nnz,
                nnz_shift=shifted.nnz,lu_nnz=eigen.lu_nnz,lu_fill_ratio=eigen.lu_nnz/shifted.nnz,
                linear_solves=eigen.linear_solves,status=eigen.status,
                eigenvalues=[[float(x.real),float(x.imag)] for x in eigen.eigenvalues])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/benchmarks')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats < 1: parser.error('repeats must be positive')
    m,a,h=physical(True)
    c=build_coupling(m,a,h,cutoff=3,quadrature_order=12,fourier_order=64,backend='prepared')
    rows=[]
    for n in (8,16,32):
        grid=StaggeredGrid(n,n,40.,40.)
        run(grid,c)  # warm-up independent of timing/memory samples
        samples=[run(grid,c) for _ in range(args.repeats)]
        tracemalloc.start()
        memory_run=run(grid,c)
        _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
        row=dict(Nx=n,Ny=n,n_dof=grid.n_dof,Lx_um=40.,Ly_um=40.,sigma=[.03,.03],
                 k=6,tol=1e-9,residual_tol=1e-9,seed=33,ncv=30,maxiter=5000,
                 repeats=args.repeats,warmup=1,samples=samples,
                 tracemalloc_peak_bytes=peak,rss_peak_bytes=None,
                 memory_note='Separate traced run: Python/registered NumPy allocations, NOT process RSS or full native LU memory.',
                 memory_run_max_residual=memory_run['max_residual'])
        for key in ('assembly_seconds','solve_total_seconds','lu_seconds','iteration_seconds','lu_fill_ratio'):
            row[key+'_median']=float(np.median([sample[key] for sample in samples]))
        for key in ('nnz_A','nnz_B','nnz_shift','lu_nnz','linear_solves','max_residual','status'):
            row[key]=samples[len(samples)//2][key]
        rows.append(row)
        save(args.output,'finite',rows,coupling=dict(geometry='dual_circle_triangle',cutoff=3,quadrature_order=12,fourier_order=64,backend='prepared'),
             scope='Sparse scaling only, same target controls; no dense/legacy speed claim, no cross-grid mode matching or mesh-convergence claim.')
        print(json.dumps({k:row[k] for k in ('Nx','assembly_seconds_median','lu_seconds_median','iteration_seconds_median','max_residual','tracemalloc_peak_bytes')}),flush=True)
    assert len(rows)==3

if __name__=='__main__': main()
