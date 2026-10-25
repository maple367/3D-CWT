"""Shared deterministic fixtures and non-identifying benchmark output."""
import os
# Set before importing NumPy/SciPy. Override inherited multi-thread settings.
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
for name in THREAD_VARIABLES:
    os.environ[name] = '1'
import csv
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import scipy
from cwt3d.geometry import CircleHole, RightTriangleHole
from cwt3d.waveguide import TEWaveguide


def physical(dual=False):
    means = [8.,10.] if dual else [9.]
    guide = TEWaveguide([1., *means, 1.], [.13,.19] if dual else [.32], k0=6.)
    mode = guide.solve_mode()
    a = 2*np.pi/mode.beta
    fill = np.pi*.13**2
    first = CircleHole(.13*a,(a,a),(.36*a,.47*a), means[0]+2*fill, means[0]-2*(1-fill))
    holes = {1:first}
    if dual:
        fill = .35*.27/2
        holes[2] = RightTriangleHole((.35*a,.27*a),(a,a),(.63*a,.54*a),
                                     means[1]+2*fill,means[1]-2*(1-fill))
    return mode,a,holes


def environment():
    from cwt3d.cache import algorithm_digest
    return dict(source_digest=algorithm_digest(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                system=platform.system(),release=platform.release(),machine=platform.machine(),
                cpu=platform.processor(),logical_cpus=os.cpu_count(),
                threads={name:os.environ[name] for name in THREAD_VARIABLES},
                precision='complex128',jit='N/A (no JIT)',rss_peak_bytes=None,
                note='Thread environment fixed before importing NumPy; RSS not measured.')


def timed(call, repeats):
    samples=[]
    for _ in range(repeats):
        start=perf_counter(); result=call(); samples.append(perf_counter()-start)
    return result,samples


def save(output, kind, rows, **extra):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    data=dict(kind=kind,environment=environment(),rows=rows,**extra)
    (output/f'{kind}.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (output/f'{kind}.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
