"""Exercise the standalone CLI rather than importing its implementation."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'examples/square_finite.py'

def run(output, *extra):
    guard = "import sys,runpy;sys.meta_path.insert(0,type('Guard',(),{'find_spec':lambda s,n,*a: (_ for _ in ()).throw(ImportError(n)) if n.split('.')[0] in ('model','mph','numba') else None})());sys.argv="+repr([str(SCRIPT),'--output',str(output),*extra])+";runpy.run_path(sys.argv[0],run_name='__main__')"
    return subprocess.run([sys.executable,'-I','-c',guard],cwd=output.parent,capture_output=True,text=True)

def test_cli_export(tmp_path):
    out=tmp_path/'demo'
    proc=run(out,'--Nx','4','--Ny','5','--k','3')
    assert proc.returncode==0,proc.stderr
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='converged'
    from importlib.metadata import version
    assert summary['environment']['cwt3d'] == version('cwt3d')
    assert summary['solver']['residual_tolerance']==1e-9
    assert summary['solver']['requested']==3
    assert len(summary['modes'])==3
    with np.load(out/'fields.npz',allow_pickle=False) as z:
        assert z['centers'].shape==(3,4,5,4)
        assert z['faces_Rx'].shape==(3,5,5)
        assert z['faces_Ry'].shape==(3,6,4)
        power=summary['input']['L_um']**2/(4*5)*np.sum(abs(z['centers'])**2,axis=(1,2,3))
        np.testing.assert_allclose(power,1,atol=1e-12)
        assert np.max(abs(z['faces_Rx'][:,:,0]))==0
        assert np.max(abs(z['faces_Sx'][:,:,-1]))==0
        for i,m in enumerate(summary['modes']):
            u=z['centers'][i].reshape(4,-1)
            coupling=2*summary['input']['L_um']**2/20*np.vdot(u,z['C']@u).imag
            assert coupling==pytest.approx(m['coupling'],abs=1e-12)
            assert m['residual']<1e-8
            assert m['balance_relative_error']<1e-7
    assert '<svg' in (out/'mode0.svg').read_text()
    assert run(out).returncode!=0

def test_extended_coupling_cli(tmp_path):
    script=SCRIPT.parents[1]/'scripts/validate_convergence.py'
    out=tmp_path/'extended'
    command=[sys.executable,'-I',str(script),'--extended-coupling-only','--output',str(out),
        '--extended-D','1','2','--extended-q','8','--extended-F','16','--refined-q','12','--refined-F','24']
    proc=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    report=json.loads((out/'extended_coupling.json').read_text())
    assert len(report['rows'])==4
    assert [r['axis'] for r in report['rows']]==['D','D','q','F']
    for r in report['rows']:
        matrices={k:np.array(v)[...,0]+1j*np.array(v)[...,1] for k,v in r['matrices'].items()}
        np.testing.assert_allclose(matrices['C'],sum(matrices[k] for k in ('C1D','Crad','C2D')),atol=1e-14)
        assert r['normC']==pytest.approx(np.linalg.norm(matrices['C']))
        assert len(r['infinite_eigenvalues'])==4 and r['seconds']>0
    assert not (out/'convergence.json').exists()
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode!=0


@pytest.mark.parametrize('extra',[['--Nx','0'],['--D','0'],['--sigma','nan'],['--k','999999']])
def test_cli_failure(tmp_path,extra):
    out=tmp_path/'bad'
    assert run(out,*extra).returncode!=0
    assert not (out/'summary.json').exists()
