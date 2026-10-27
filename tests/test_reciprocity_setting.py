"""Default-on reciprocity setting, explicit old path and cache isolation."""
from dataclasses import replace
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys
import numpy as np
import pytest
from cwt3d.coupling import build_coupling
from cwt3d.quadrature import green_overlap
from cwt3d.greens import HomogeneousGreen
from cwt3d.cache import CouplingCache, CacheError
from test_coupling import physical
from test_quadrature_reciprocity import ComplexField, two_triangle_reference


def example(name):
    spec=importlib.util.spec_from_file_location('setting_'+name,Path(__file__).resolve().parents[1]/'examples'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('flag,calls_expected',[(True,1),(False,2)])
def test_low_level_switch_controls_evaluation(flag,calls_expected,monkeypatch):
    kernel=HomogeneousGreen(1.3,2.3-.2j,.7)
    original=HomogeneousGreen.__call__;calls=[]
    def counted(self,z,s):
        calls.append(1);return original(self,z,s)
    monkeypatch.setattr(HomogeneousGreen,'__call__',counted)
    got=green_overlap(ComplexField(),(.02,.27),(.02,.27),kernel,8,reciprocity=flag)
    assert len(calls)==calls_expected
    old=two_triangle_reference(ComplexField(),(.02,.27),(.02,.27),kernel,8)
    np.testing.assert_allclose(got,old,rtol=5e-12,atol=2e-14)


@pytest.mark.parametrize('backend',['reference','prepared'])
def test_default_and_explicit_settings_have_correct_metadata(backend):
    mode,a,h=physical(dual=True)
    options=dict(backend=backend,quadrature_order=6,fourier_order=24)
    default=build_coupling(mode,a,h,**options)
    on=build_coupling(mode,a,h,reciprocity=True,**options)
    off=build_coupling(mode,a,h,reciprocity=False,**options)
    assert default==on
    assert on.reciprocity is True and off.reciprocity is False
    assert json.loads(on.provenance)['options']['reciprocity'] is True
    assert json.loads(off.provenance)['options']['reciprocity'] is False
    for name in ('C','C1D','Crad','C2D','K','mu','nu','varsigma'):
        np.testing.assert_allclose(getattr(on,name),getattr(off,name),rtol=5e-12,atol=2e-13)


def test_setting_is_forwarded_to_every_overlap(monkeypatch):
    import cwt3d.coupling as module
    original=module.green_overlap;flags=[]
    def recording(*args,**kwargs):
        flags.append(kwargs['reciprocity']);return original(*args,**kwargs)
    monkeypatch.setattr(module,'green_overlap',recording)
    mode,a,h=physical(dual=True)
    result=build_coupling(mode,a,h,quadrature_order=4,reciprocity=False)
    assert len(flags)==len(result.radial_orders)*len(result.pc_layers)**2
    assert all(flag is False for flag in flags)


@pytest.mark.parametrize('bad',[0,1,'false',None,np.array(True)])
def test_nonboolean_setting_rejected_without_cache_write(bad,tmp_path):
    mode,a,h=physical();cache=CouplingCache(tmp_path/'cache')
    for fn in (build_coupling,cache.key,cache.build):
        with pytest.raises(ValueError,match='reciprocity must be boolean'):
            fn(mode,a,h,reciprocity=bad)
    with pytest.raises(ValueError,match='reciprocity must be boolean'):
        green_overlap(ComplexField(),(0.,.2),(0.,.2),HomogeneousGreen(1.,2.),4,reciprocity=bad)
    assert not cache.directory.exists()


@pytest.mark.parametrize('flag',[True,False])
def test_cache_setting_roundtrip_and_numpy_boolean(flag,tmp_path):
    mode,a,h=physical();cache=CouplingCache(tmp_path/'cache')
    opts=dict(quadrature_order=4,reciprocity=flag)
    key=cache.key(mode,a,h,**opts)
    assert key==cache.key(mode,a,h,quadrature_order=4,reciprocity=np.bool_(flag))
    assert key!=cache.key(mode,a,h,quadrature_order=4,reciprocity=not flag)
    result=cache.build(mode,a,h,**opts)
    assert result.reciprocity is flag
    assert cache.load(key)==result
    assert cache.build(mode,a,h,**opts)==result
    opposite=cache.document(mode,a,h,quadrature_order=4,reciprocity=not flag)
    with pytest.raises(CacheError,match='provenance'):
        cache.store(cache.key(mode,a,h,quadrature_order=4,reciprocity=not flag),result,document=opposite)
    document=cache.document(mode,a,h,**opts)
    with pytest.raises(CacheError,match='numerical controls'):
        cache.store(key,replace(result,reciprocity=not flag),document=document)
    with pytest.raises(CacheError,match='boolean'):
        cache.store(key,replace(result,reciprocity=int(flag)),document=document)


@pytest.mark.parametrize('name',['liang_70um','square_finite'])
def test_example_cli_switches(name):
    m=example(name)
    assert m.parser().parse_args([]).reciprocity is True
    assert m.parser().parse_args(['--no-reciprocity']).reciprocity is False
    assert m.parser().parse_args(['--reciprocity']).reciprocity is True


def test_liang_run_setting_and_export(tmp_path):
    m=example('liang_70um')
    result=m.run_case(N=4,D=1,q=6,F=16,k=1,reciprocity=False,output=tmp_path/'out')
    assert result.coupling.reciprocity is False
    assert result.summary['input']['reciprocity'] is False
    assert json.loads((tmp_path/'out'/'summary.json').read_text())['input']['reciprocity'] is False


@pytest.mark.parametrize('name',['liang_70um','square_finite'])
def test_actual_cli_exports_on_and_off(name,tmp_path):
    script=Path(__file__).resolve().parents[1]/'examples'/(name+'.py')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    results=[]
    for flag in (True,False):
        out=tmp_path/str(flag)
        grid=['--N','4'] if name=='liang_70um' else ['--Nx','4','--Ny','4']
        command=[sys.executable,str(script),*grid,'--D','1','--q','6','--F','16','--k','1','--output',str(out)]
        if not flag:command.append('--no-reciprocity')
        subprocess.run(command,env=env,check=True,capture_output=True,text=True,timeout=30)
        summary=json.loads((out/'summary.json').read_text())
        assert summary['input']['reciprocity'] is flag
        with np.load(out/'fields.npz',allow_pickle=False) as data:
            results.append((data['C'].copy(),data['eigenvalues'].copy()))
    for a,b in zip(results[0],results[1]):
        np.testing.assert_allclose(a,b,rtol=5e-12,atol=2e-13)


def test_precomputed_coupling_cannot_ignore_setting():
    m=example('square_finite')
    args=m.resolve(m.parser().parse_args(['--no-reciprocity']))
    mode,a,h=physical()
    coupling=build_coupling(mode,a,h,quadrature_order=4)
    with pytest.raises(ValueError,match='precomputed coupling reciprocity'):
        m.compute(args,coupling=coupling)
