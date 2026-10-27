"""Cross-module gates on actual passive structures and public active APIs."""
import importlib.util
import json
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import pytest
from scipy.constants import elementary_charge
from cwt3d.active import (ActiveCavity,ActiveState,CarrierMaterial,simulate,
                         carrier_rhs,diffusion_operator,solve_active_modes)
from cwt3d.coupling import build_coupling
from cwt3d.finite.assembly import FinitePencil
from cwt3d.finite.eigensolver import solve_finite
from cwt3d.finite.grid import StaggeredGrid


@pytest.fixture(scope='module')
def physical_active():
    path=Path(__file__).resolve().parents[1]/'examples/liang_70um.py'
    spec=importlib.util.spec_from_file_location('active_integration_chapter4',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    a,eps,widths,holes=module.build_structure()
    reference=module.solve_reference_mode(a,eps,widths)
    coupling=build_coupling(reference.mode,a,holes,cutoff=1,quadrature_order=6,fourier_order=16,backend='prepared')
    pencil=FinitePencil(StaggeredGrid(3,3,70.,70.),coupling)
    shift=complex(min(np.linalg.eigvals(coupling.C).real)-.6/70.,.25/70.)
    eig=solve_finite(pencil,sigma=shift,k=1,seed=33,tol=1e-11)
    cavity=ActiveCavity(pencil,group_index=3.5,k0_um_inverse=reference.k0,
                        active_thickness_um=.02,confinement=.04,coupling=coupling)
    return cavity,pencil,eig,shift


def test_real_passive_structure_recovers_frequency_decay_and_photons(physical_active):
    cavity,pencil,eig,shift=physical_active
    material=CarrierMaterial(2e-7,0.)
    active=solve_active_modes(cavity,np.zeros((3,3)),material,sigma=shift,k=1,seed=33,tol=1e-11)
    np.testing.assert_allclose(active.eigenvalues,eig.eigenvalues,rtol=1e-10,atol=1e-12)
    seed=cavity.normalize_seed(eig.eigenvectors[:,0],7.)
    assert cavity.photon_number(seed)==pytest.approx(7.)
    initial=ActiveState(seed,np.zeros((3,3)))
    times=np.linspace(0,1e-3,9)
    result=simulate(cavity,material,initial,times_ns=times,rtol=1e-8,field_atol=1e-12,
                    carrier_atol=1e-5,reference_detuning_um_inverse=float(eig.eigenvalues[0].real))
    expected=7*np.exp(-2*cavity.vg_um_ns*eig.eigenvalues[0].imag*times)
    np.testing.assert_allclose(result.photon_number,expected,rtol=2e-6,atol=1e-10)
    np.testing.assert_allclose(result.carrier_density_um3,0.,atol=1e-12)


def test_nonuniform_injection_exact_limit_without_optical_seed(physical_active):
    cavity,_,_,_=physical_active
    a=.7
    material=CarrierMaterial(2e-7,1e6,recombination_a_ns=a)
    current=np.arange(1,10).reshape(3,3)*1e-7
    times=np.linspace(0,.2,9)
    initial=ActiveState(np.zeros(cavity.grid.n_dof,complex),np.zeros((3,3)))
    result=simulate(cavity,material,initial,times_ns=times,current_density_A_um2=current,
                    rtol=1e-8,carrier_atol=1e-4)
    expected=(current*1e-9/(elementary_charge*cavity.active_thickness_um))[None,:,:]*(1-np.exp(-a*times[:,None,None]))/a
    np.testing.assert_allclose(result.carrier_density_um3,expected,rtol=2e-6,atol=1e-3)
    np.testing.assert_array_equal(result.field_vectors,0.)
    np.testing.assert_array_equal(result.photon_number,0.)


def test_stimulated_photon_carrier_reciprocity_including_absorption(physical_active):
    cavity,_,eig,_=physical_active
    material=CarrierMaterial(2e-7,1e6,gain_compression_um3=2e-5)
    v=cavity.normalize_seed(eig.eigenvectors[:,0],13.)
    density=np.linspace(.1e6,3e6,9).reshape(3,3)
    photons=cavity.photon_density(v)
    dn=carrier_rhs(density,photons,0.,material,vg_um_ns=cavity.vg_um_ns,
        active_thickness_um=cavity.active_thickness_um,
        diffusion_matrix=diffusion_operator(cavity.grid,0.))
    carrier_change=float(np.sum(dn)*cavity.grid.hx_um*cavity.grid.hy_um*cavity.active_thickness_um)
    stimulated=cavity.power_budget(v,density,material)['stimulated_per_ns']
    assert carrier_change==pytest.approx(-stimulated,rel=2e-12,abs=1e-10)


def test_active_example_actual_cli_export(tmp_path):
    script=Path(__file__).resolve().parents[1]/'examples/active_cavity.py'
    output=tmp_path/'demo'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,str(script),'--N','3','--D','1','--q','6','--F','16',
        '--duration-ns','.001','--samples','5','--output',str(output)],
        check=True,env=env,capture_output=True,text=True,timeout=60)
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert summary['status']=='completed'
    assert 'not a calibrated' in summary['scope']
    assert summary['environment']['algorithm_digest']
    assert len(summary['environment']['active_algorithm_digest'])==64
    assert len(summary['environment']['example_sha256'])==64
    assert summary['passive_optical_inputs']
    assert summary['reference_k0_um_inverse']>0
    with np.load(output/'trajectory.npz',allow_pickle=False) as arrays:
        assert arrays['field_vectors'].shape==(5,36)
        assert arrays['carrier_density_um3'].shape==(5,3,3)
        assert np.isfinite(arrays['photon_number']).all()
        assert arrays['photon_number'][0]==pytest.approx(1.)
