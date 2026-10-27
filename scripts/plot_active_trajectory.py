"""Plot an exported active example and compare matched time-refinement runs.

The displayed envelope is not the optical interference inside a unit cell.
Use only matched input/initialization cases for a convergence comparison.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cwt3d.finite.grid import StaggeredGrid


def _difference_metrics(candidate, reference):
    """Return absolute peak error and reference-peak-normalized error.

    Identical zero trajectories have zero error; a nonzero error against a
    zero reference has undefined normalization (JSON null), not a floored ratio.
    """
    absolute = float(np.max(abs(candidate-reference)))
    peak = float(np.max(reference))
    normalized = absolute/peak if peak != 0. else (0. if absolute == 0. else None)
    return absolute, normalized


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--compare',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metrics',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists() or args.metrics.exists():
        raise ValueError('choose fresh figure and metrics paths')
    cases=[]
    ignored={'rtol','max_step_ns'}
    for directory in [args.reference]+args.compare:
        summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        if summary.get('schema')!='cwt3d-active-demo-v1' or summary.get('status')!='completed':
            raise ValueError('requires a completed active example export')
        with np.load(directory/'trajectory.npz',allow_pickle=False) as archive:
            arrays={key:archive[key] for key in ('times_ns','field_vectors','carrier_density_um3','photon_number')}
        if any(not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError('trajectory contains nonfinite values')
        cases.append((directory.name,summary,arrays))
    _,reference,r=cases[0]
    comparable=lambda s:{k:v for k,v in s['input'].items() if k not in ignored}
    metrics=[]
    for name,summary,arrays in cases:
        if comparable(summary)!=comparable(reference) or not np.array_equal(arrays['times_ns'],r['times_ns']):
            raise ValueError('comparison requires identical physical inputs and output times')
        if summary['active_parameters']!=reference['active_parameters']:
            raise ValueError('active material parameters differ')
        for key in ('algorithm_digest','active_algorithm_digest','example_sha256'):
            if not reference['environment'].get(key) or summary['environment'].get(key)!=reference['environment'][key]:
                raise ValueError('comparison requires identical fingerprinted source: '+key)
        for key in ('reference','initialisation','passive_optical_inputs'):
            if summary.get(key)!=reference.get(key):
                raise ValueError('reference optics or initialization differ: '+key)
        photon_absolute,photon_relative=_difference_metrics(arrays['photon_number'],r['photon_number'])
        carrier_absolute,carrier_relative=_difference_metrics(arrays['carrier_density_um3'],r['carrier_density_um3'])
        metrics.append(dict(case=name,rtol=summary['input']['rtol'],
            max_step_ns=summary['input']['max_step_ns'],
            max_photon_difference_over_reference_peak=photon_relative,
            max_carrier_difference_over_reference_peak=carrier_relative,
            max_photon_absolute_difference=photon_absolute,
            max_carrier_absolute_difference=carrier_absolute,
            final_photon_number=float(arrays['photon_number'][-1]),
            final_window_relative_photon_variation=summary['final_window_relative_photon_variation']))
    # Validate all exported JSON before creating a figure or writing either file.
    metrics_json=json.dumps(dict(schema='cwt3d-active-refinement-v1',
        scope='Pairwise trajectory agreement only; the reference is not an exact solution or spatial-convergence certificate',
        reference_environment=reference['environment'],rows=metrics),indent=2,allow_nan=False)
    report_json=json.dumps(dict(figure=str(args.output),rows=metrics),allow_nan=False)
    n=reference['input']['N']
    grid=StaggeredGrid(n,n,70.,70.)
    peak=int(np.argmax(r['photon_number']))
    center=grid.interpolate(grid.unpack(r['field_vectors'][peak]))
    sheet=sum(abs(getattr(center,key))**2 for key in ('Rx','Sx','Ry','Sy'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7.6),layout='constrained')
    for name,summary,arrays in reversed(cases):
        axes[0,0].plot(arrays['times_ns'],arrays['photon_number'],label=f"rtol={summary['input']['rtol']:g}",lw=1.5)
    axes[0,0].set(xlabel='Time [ns]',ylabel='Total photons',title='Prebiased, explicitly seeded dynamics')
    axes[0,0].legend(fontsize=8)
    density=r['carrier_density_um3']
    axes[0,1].fill_between(r['times_ns'],density.min(axis=(1,2)),density.max(axis=(1,2)),alpha=.2,label='Spatial min-max')
    axes[0,1].plot(r['times_ns'],density.mean(axis=(1,2)),label='Spatial mean')
    axes[0,1].set(xlabel='Time [ns]',ylabel='Carrier density [um$^{-3}$]',title='Spatial carrier evolution')
    axes[0,1].legend(fontsize=8)
    for ax,data,title,label in (
        (axes[1,0],sheet,'Envelope at photon peak','Sheet photons [um$^{-2}$]'),
        (axes[1,1],density[peak],'Carriers at photon peak','Carrier density [um$^{-3}$]')):
        im=ax.imshow(data,extent=(0,70,0,70),origin='lower',interpolation='nearest',aspect='equal')
        ax.set(xlabel='x [um]',ylabel='y [um]',title=title)
        fig.colorbar(im,ax=ax,label=label,shrink=.82)
    fig.suptitle(f"Active-cavity numerical example | N={n}, D={reference['input']['D']}, q={reference['input']['q']}\nIllustrative active parameters; not a calibrated device or CW certificate",fontsize=12)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.metrics.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)
    args.metrics.write_text(metrics_json,encoding='utf-8')
    print(report_json)


if __name__=='__main__':
    main()
