"""Deterministic active-cavity demonstration on the chapter-4 passive structure.

The active material law is illustrative, not calibrated to Liang's device.
All lengths are um, times ns, densities um^-3 and current densities A/um^2.
Install cwt3d before running this source-distribution example; no noise/thermal model.
"""
import argparse
import hashlib
import cwt3d.active as active_module
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import sys
import numpy as np
from scipy.constants import elementary_charge
from cwt3d.active import (ActiveCavity, ActiveState, CarrierMaterial, simulate,
                         uniform_threshold_gain)
from cwt3d.coupling import build_coupling
from cwt3d.waveguide import TEWaveguide
from cwt3d.finite.assembly import FinitePencil
from cwt3d.finite.grid import StaggeredGrid
from cwt3d.finite.eigensolver import solve_finite
from cwt3d.cache import algorithm_digest


def _active_source_digest():
    """Fingerprint all shipped active modules; passive cache hashes omit them."""
    root=Path(active_module.__file__).parent
    digest=hashlib.sha256(b'cwt3d-active-v1\0')
    for name in ('__init__.py','carriers.py','cavity.py','dynamics.py','materials.py','results.py','threshold.py'):
        digest.update(name.encode('ascii')+b'\0'+(root/name).read_bytes()+b'\0')
    return digest.hexdigest()


def _chapter4():
    path=Path(__file__).with_name('liang_70um.py')
    spec=importlib.util.spec_from_file_location('active_chapter4_reference',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    a,eps,widths,holes=module.build_structure()
    reference=module.solve_reference_mode(a,eps,widths)
    return a,eps,widths,holes,reference


def run_case(*,N=8,D=3,q=12,F=32,duration_ns=1.,samples=101,pump_ratio=1.5,
             initial_photons=1.,rtol=1e-6,max_step_ns=.01,
             index_per_density_um3=0.,output=None):
    """One pumped trajectory, not a fitted device or certified CW solution."""
    for name,value in dict(N=N,D=D,q=q,F=F,samples=samples).items():
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value < (2 if name=='samples' else 1):
            raise ValueError(name+' must be a positive integer (samples >=2)')
    if not np.isfinite(duration_ns) or duration_ns<=0 or not np.isfinite(pump_ratio) or pump_ratio<0:
        raise ValueError('positive finite duration and nonnegative pump_ratio required')
    if output is not None:
        output=Path(output)
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('choose a fresh output directory')
    a,eps,widths,holes,reference=_chapter4()
    coupling=build_coupling(reference.mode,a,holes,cutoff=D,quadrature_order=q,
                            fourier_order=F,backend='prepared')
    pencil=FinitePencil(StaggeredGrid(N,N,70.,70.),coupling)
    sigma=complex(min(np.linalg.eigvals(coupling.C).real)-.6/70.,.25/70.)
    passive=solve_finite(pencil,sigma=sigma,k=1,tol=1e-10,residual_tol=1e-9,seed=33)
    selected=complex(passive.eigenvalues[0])
    dk=reference.k0*1e-5
    beta_plus=TEWaveguide(eps,widths,k0=reference.k0+dk).solve_mode().beta
    beta_minus=TEWaveguide(eps,widths,k0=reference.k0-dk).solve_mode().beta
    group_index=float((beta_plus-beta_minus)/(2*dk))
    cavity=ActiveCavity(pencil,group_index=group_index,k0_um_inverse=reference.k0,
                        active_thickness_um=.02,confinement=.04,
                        internal_loss_um_inverse=.0003,coupling=coupling)
    material=CarrierMaterial(2e-7,1.3e6,gain_compression_um3=2.45e-5,
                             index_per_density_um3=index_per_density_um3,
                             recombination_a_ns=.5)
    gain_threshold=float(uniform_threshold_gain(selected.imag,confinement=.04,
                                                internal_loss_um_inverse=.0003))
    nth=1.3e6+gain_threshold/2e-7
    # Reference-only uniform weak-field threshold; no nonuniform threshold claim.
    jth=elementary_charge*.02*float(material.recombination(nth))*1e9
    current_density=pump_ratio*jth
    initial=ActiveState(cavity.normalize_seed(passive.eigenvectors[:,0],initial_photons),
                        np.full((N,N),min(pump_ratio,1.)*nth))
    result=simulate(cavity,material,initial,times_ns=np.linspace(0,duration_ns,samples),
                    current_density_A_um2=current_density,diffusion_um2_ns=10.,
                    rtol=rtol,field_atol=1e-10,carrier_atol=1e-3,max_step_ns=max_step_ns,
                    reference_detuning_um_inverse=selected.real)
    budgets=[cavity.power_budget(v,n,material)
             for v,n in zip(result.field_vectors,result.carrier_density_um3)]
    tail=result.photon_number[-max(3,samples//10):]
    scale=float(np.mean(abs(tail)))
    variation=float(np.ptp(tail)/scale) if scale else 0.
    summary=dict(schema='cwt3d-active-demo-v1',status='completed',
        scope='Deterministic illustrative active law on Liang chapter-4 passive structure; not a calibrated device reproduction, complete modal survey or CW-stability certificate; no thermal transport or spontaneous-emission noise.',
        input=dict(N=N,D=D,q=q,F=F,duration_ns=duration_ns,samples=samples,
                   pump_ratio=pump_ratio,initial_photons=initial_photons,
                   rtol=rtol,max_step_ns=max_step_ns,index_per_density_um3=index_per_density_um3),
        units=dict(length='um',time='ns',volume_density='um^-3',power_gain='um^-1',
                   current_density='A/um^2',photon_number='1',power='W'),
        environment=dict(cwt3d=version('cwt3d'),algorithm_digest=algorithm_digest(),
            active_algorithm_digest=_active_source_digest(),
            example_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        passive_optical_inputs=json.loads(coupling.provenance),
        reference_k0_um_inverse=float(reference.k0),
        active_parameters=dict(differential_gain_um2=2e-7,transparency_density_um3=1.3e6,
            gain_compression_um3=2.45e-5,index_per_density_um3=index_per_density_um3,
            recombination_a_ns=.5,diffusion_um2_ns=10.,active_thickness_um=.02,
            confinement=.04,internal_loss_um_inverse=.0003,group_index=group_index,
            group_index_scope='finite-difference d(beta)/d(k0) with fixed material permittivity',
            provenance='Illustrative user-adjustable active parameters; not measured/extracted for this passive structure'),
        reference=dict(passive_eigenvalue_um_inverse=[selected.real,selected.imag],
            passive_residual=float(passive.residuals[0]),reference_detuning_um_inverse=selected.real,
            uniform_small_signal_material_gain_threshold_um_inverse=gain_threshold,
            uniform_small_signal_carrier_threshold_um3=nth,
            uniform_current_density_reference_A_um2=jth,
            applied_current_A=current_density*70.**2),
        initialisation=dict(photon_number=float(initial_photons),
            carrier_density_um3=float(min(pump_ratio,1.)*nth),
            scope='Uniform prebiased carriers with an explicit deterministic optical seed; not noise-started turn-on from an empty active layer'),
        solver=dict(result.metadata),radiation_scope=cavity.radiation_scope,
        initial_budget=budgets[0],final_budget=budgets[-1],
        final_photon_number=float(result.photon_number[-1]),
        final_carrier_min_um3=float(np.min(result.carrier_density_um3[-1])),
        final_carrier_max_um3=float(np.max(result.carrier_density_um3[-1])),
        final_window_relative_photon_variation=variation,
        steady_scope='Window variation is a diagnostic, not a spectral/unique stable CW certificate')
    if output is not None:
        output.mkdir(parents=True,exist_ok=True)
        arrays=dict(times_ns=result.times_ns,field_vectors=result.field_vectors,
                    carrier_density_um3=result.carrier_density_um3,photon_number=result.photon_number)
        for key in budgets[0]:
            if all(np.isscalar(b[key]) and isinstance(b[key],(int,float,np.number)) for b in budgets):
                arrays['budget_'+key]=np.array([b[key] for b in budgets],dtype=float)
        np.savez_compressed(output/'trajectory.npz',**arrays)
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return result,summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('N',8),('D',3),('q',12),('F',32),('samples',101)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--duration-ns',type=float,default=1.)
    p.add_argument('--pump-ratio',type=float,default=1.5)
    p.add_argument('--initial-photons',type=float,default=1.)
    p.add_argument('--rtol',type=float,default=1e-6)
    p.add_argument('--max-step-ns',type=float,default=.01)
    p.add_argument('--index-per-density-um3',type=float,default=0.)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    _,summary=run_case(**vars(args))
    print(json.dumps(dict(status=summary['status'],final_photon_number=summary['final_photon_number'],
                         final_window_relative_photon_variation=summary['final_window_relative_photon_variation'],
                         output=str(args.output)),allow_nan=False))


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr);sys.exit(1)
