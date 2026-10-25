"""Run from repository: .venv/Scripts/python.exe benchmarks/bench_coupling.py --cache-dir <explicit path>."""
from common import *
import argparse
from scipy.optimize import linear_sum_assignment
from cwt3d.coupling import build_coupling
from cwt3d.greens import LayeredGreen, PreparedLayeredGreen
from cwt3d.cache import CouplingCache


def eigen_error(a,b):
    x,y=np.linalg.eigvals(a),np.linalg.eigvals(b)
    i,j=linear_sum_assignment(abs(x[:,None]-y))
    return float(max(abs(x[i]-y[j])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--output',default='docs/benchmarks')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats < 1: parser.error('repeats must be positive')
    rows=[]
    cache=CouplingCache(args.cache_dir)
    for dual in (False,True):
        mode,a,holes=physical(dual)
        # Explicit, measured initialization and steady evaluation, not JIT compile.
        z=np.linspace(.01,.12,12)[:,None]; s=np.linspace(.015,.125,12)[None,:]
        kernel_times={}
        for name,cls in [('reference',LayeredGreen),('prepared',PreparedLayeredGreen)]:
            kernel,init=timed(lambda:cls.from_structure(mode.guide),args.repeats)
            kernel(z,s)
            _,steady=timed(lambda:kernel(z,s),args.repeats)
            _,scalar=timed(lambda:cls.from_structure(mode.guide)(.05,.07),args.repeats)
            kernel_times[name]=dict(init_samples_seconds=init,init_median_seconds=float(np.median(init)),
                                    steady_samples_seconds=steady,steady_median_seconds=float(np.median(steady)),
                                    scalar_cold_samples_seconds=scalar,scalar_cold_median_seconds=float(np.median(scalar)))
        for D in (1,3,10):
            options=dict(cutoff=D,quadrature_order=12,fourier_order=64)
            call=lambda backend,q=12:build_coupling(mode,a,holes,**dict(options,quadrature_order=q),backend=backend)
            # Each backend gets one unrecorded warm-up; all reported samples retained.
            call('reference'); call('prepared')
            ref,rs=timed(lambda:call('reference'),args.repeats)
            fast,fs=timed(lambda:call('prepared'),args.repeats)
            # Independent higher-q reference, never lower accuracy to gain speed.
            high,hs=timed(lambda:call('reference',24),1)
            middle=call('reference',18)
            key=cache.key(mode,a,holes,backend='prepared',**options)
            cache.clear(key)  # exact owned key only, ensures actual cold miss
            cold,cs=timed(lambda:cache.build(mode,a,holes,backend='prepared',**options),1)
            hit,ch=timed(lambda:cache.build(mode,a,holes,backend='prepared',**options),args.repeats)
            assert cold == hit == fast
            row=dict(geometry='dual_circle_triangle' if dual else 'single_circle',cutoff=D,
                     quadrature_order=12,fourier_order=64,convergence_reference_q=24,
                     repeats=args.repeats,warmup_per_backend=1,
                     reference_samples_seconds=rs,prepared_samples_seconds=fs,
                     reference_median_seconds=float(np.median(rs)),prepared_median_seconds=float(np.median(fs)),
                     speedup=float(np.median(rs)/np.median(fs)),
                     matrix_max_error=float(np.max(abs(ref.C-fast.C))),eigen_set_max_error=eigen_error(ref.C,fast.C),
                     q12_vs_q24_matrix_max_error=float(np.max(abs(ref.C-high.C))),
                     q18_vs_q24_matrix_max_error=float(np.max(abs(middle.C-high.C))),
                     q12_vs_q24_eigen_set_max_error=eigen_error(ref.C,high.C),
                     q24_reference_samples_seconds=hs,
                     cache_miss_seconds=cs[0],cache_hit_samples_seconds=ch,cache_hit_median_seconds=float(np.median(ch)),
                     kernel_timings=kernel_times,radial_count=len(ref.radial_orders),
                     cache_cleanup='exact key removed after measurement')
            rows.append(row)
            cache.clear(key)
            save(args.output,'coupling',rows,accuracy='Same D/q/F/geometry and Duffy triangles; eigenvalues Hungarian matched.',
                 cache_policy='Explicit directory only; measured keys deleted. Scratch host policy prunes entries idle 24h.')
            print(json.dumps({k:row[k] for k in ('geometry','cutoff','speedup','matrix_max_error','q12_vs_q24_matrix_max_error')}),flush=True)
    assert len(rows)==6

if __name__=='__main__': main()
