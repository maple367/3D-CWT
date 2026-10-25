"""Audit actual runs and package an allowlisted portable source/results delivery.

Run from repository root after building the wheel and executing both examples
and validate_convergence.py. No original data/history is copied or modified.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from cwt3d.cache import algorithm_digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'3D-CWT-deliverables'


def main():
    report=json.loads((OUT/'convergence/convergence.json').read_text())
    rows=report['rows']
    assert len(rows)==15
    expected={'D':[1,3,6,8,10],'q':[8,12,18,24],'mesh':[8,16,32,64],'shift':[[.03,.03],[.04,.03]]}
    for axis,values in expected.items():
        selected=[r for r in rows if r['axis']==axis]
        assert [r['value'] for r in selected]==values
        for r in selected:
            inp=r['input']
            assert inp['L_um']==14.75 and inp['F']==64 and inp['k']==12
            assert inp['D']==(r['value'] if axis=='D' else 10)
            assert inp['q']==(r['value'] if axis=='q' else 24)
            assert inp['Nx']==inp['Ny']==(r['value'] if axis=='mesh' else 16)
            assert inp['sigma']==(r['value'] if axis=='shift' else [.03,.03])
    flat=list(csv.DictReader((OUT/'convergence/convergence.csv').open()))
    assert len(flat)==len(rows) and sum(r['returned'] for r in rows)==180
    for x,y in zip(rows,flat):
        assert x['run']==int(y['run']) and x['max_residual']==float(y['max_residual'])
    digest=algorithm_digest()
    extended=json.loads((OUT/'convergence/extended_coupling.json').read_text())
    extrows=extended['rows']
    assert extended['environment']['source_digest']==digest
    assert [(r['D'],r['q'],r['F']) for r in extrows]==[(10,48,128),(16,48,128),(24,48,128),(32,48,128),(32,64,128),(32,64,192)]
    extcsv=list(csv.DictReader((OUT/'convergence/extended_coupling.csv').open()))
    assert len(extcsv)==len(extrows)==6
    previous=None
    for row,flatrow in zip(extrows,extcsv):
        for key,value in row.items():
            if isinstance(value,(dict,list)):assert json.loads(flatrow[key])==value
            elif value is None:assert flatrow[key]==''
            else:assert str(value)==flatrow[key]
        matrices={k:np.asarray(v)[...,0]+1j*np.asarray(v)[...,1] for k,v in row['matrices'].items()}
        for matrix in matrices.values():assert matrix.shape==(4,4) and np.all(np.isfinite(matrix))
        np.testing.assert_allclose(matrices['C'],sum(matrices[k] for k in ('C1D','Crad','C2D')),atol=1e-14)
        np.testing.assert_allclose(row['normC'],np.linalg.norm(matrices['C']),rtol=1e-14)
        eigen=np.asarray(row['infinite_eigenvalues']);eigen=eigen[:,0]+1j*eigen[:,1]
        # Characteristic polynomial is independent of the eigensolver's ordering.
        np.testing.assert_allclose(np.poly(eigen),np.poly(matrices['C']),rtol=1e-12,atol=1e-15)
        np.testing.assert_array_equal(row['alpha'],eigen.imag)
        assert row['seconds']>0
        if previous is not None:
            for name,matrix in matrices.items():
                delta=matrix-previous[name];comparison=row['comparison_previous'][name]
                np.testing.assert_allclose(comparison['delta_norm'],np.linalg.norm(delta),rtol=1e-14,atol=1e-30)
                np.testing.assert_allclose(comparison['relative_norm'],np.linalg.norm(delta)/np.linalg.norm(matrix),rtol=1e-14,atol=1e-30)
        previous=matrices
    summaries=list((OUT/'convergence').glob('*/summary.json'))+[OUT/'demo/summary.json',OUT/'liang/summary.json']
    assert len(summaries)==17
    for path in summaries:
        s=json.loads(path.read_text());folder=path.parent
        assert s['environment']['source_digest']==digest and s['status']=='converged'
        assert s['solver']['returned']==len(s['modes'])
        with np.load(folder/'fields.npz',allow_pickle=False) as z:
            n=s['input']['Nx'];m=s['input']['Ny'];k=len(s['modes']);area=s['input']['L_um']**2/(n*m)
            assert z['centers'].shape==(k,4,m,n)
            for a in z.values():assert a.dtype.kind in 'iufc' and np.all(np.isfinite(a))
            np.testing.assert_allclose(area*np.sum(abs(z['centers'])**2,axis=(1,2,3)),1,atol=1e-12)
            for name in ('Rx','Sx'):assert z['faces_'+name].shape==(k,m,n+1)
            for name in ('Ry','Sy'):assert z['faces_'+name].shape==(k,m+1,n)
            assert not np.any(z['faces_Rx'][:,:,0]) and not np.any(z['faces_Sx'][:,:,-1])
            assert not np.any(z['faces_Ry'][:,0,:]) and not np.any(z['faces_Sy'][:,-1,:])
            for j,mode in enumerate(s['modes']):
                edge=s['input']['L_um']/m*(np.sum(abs(z['faces_Rx'][j,:,-1])**2)+np.sum(abs(z['faces_Sx'][j,:,0])**2))+s['input']['L_um']/n*(np.sum(abs(z['faces_Ry'][j,-1,:])**2)+np.sum(abs(z['faces_Sy'][j,0,:])**2))
                np.testing.assert_allclose(edge,mode['edge'],atol=1e-12)
                u=z['centers'][j].reshape(4,-1)
                pc=2*area*np.vdot(u,z['C']@u).imag
                np.testing.assert_allclose(pc,mode['coupling'],atol=1e-12)
                assert abs(2*z['eigenvalues'][j].imag-edge-pc)<1e-8
        assert len(list(csv.DictReader((folder/'modes.csv').open())))==len(s['modes'])
    benchmark=json.loads((ROOT/'docs/benchmarks/coupling.json').read_text())
    assert benchmark['environment']['source_digest']==digest
    assert subprocess.check_output(['git','diff','--','src'],cwd=ROOT,text=True)==''
    source=[]
    for directory in ('src/cwt3d','tests','examples','scripts','benchmarks'):
        source.extend(p for p in (ROOT/directory).rglob('*.py') if '__pycache__' not in p.parts)
    source.extend((ROOT/'docs').rglob('*.md'))
    source.extend(p for p in (ROOT/'docs/benchmarks').rglob('*') if p.is_file() and p.suffix in ('.json','.csv'))
    source.extend(ROOT/p for p in ('README.md','LICENSE','pyproject.toml','model/_Model.py','calculator/_calc.py'))
    assert (ROOT/'model/_Model.py') in source and (ROOT/'calculator/_calc.py') in source
    with zipfile.ZipFile(OUT/'cwt3d-task13-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(source)):z.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(OUT/'cwt3d-task13-results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('demo','liang','convergence'):
            for path in sorted((OUT/folder).rglob('*')):
                if path.is_file():z.write(path,path.relative_to(OUT).as_posix())
    artifacts=[]
    for path in sorted(OUT.glob('*')):
        if path.suffix not in ('.zip','.whl'):continue
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            names=z.namelist()
            assert not any(name.startswith('/') or '..' in Path(name).parts or any(x in Path(name).parts for x in ('.git','.venv','__pycache__')) or name.endswith(('.mph','.pth')) for name in names)
            manifest=[dict(path=info.filename,bytes=info.file_size,sha256=hashlib.sha256(z.read(info)).hexdigest()) for info in z.infolist()]
            for entry in manifest:
                if path.suffix=='.zip':
                    target=(ROOT if 'source' in path.name else OUT)/entry['path']
                    assert target.stat().st_size==entry['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256']
            if path.suffix=='.whl':
                assert 'cwt3d/finite/eigensolver.py' in names
                metadata=z.read('cwt3d-0.1.0.dist-info/METADATA').decode()
                assert 'Requires-Dist: numpy' in metadata and 'Requires-Dist: scipy' in metadata
                assert not any(x in metadata.lower() for x in ('requires-dist: mph','requires-dist: comsol'))
        artifacts.append(dict(name=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),entries=len(names),manifest=manifest))
    audit=dict(schema='cwt3d-delivery-audit-v1',source_digest=digest,runs=15,scan_modes=180,export_directories=17,
        max_residual=max(r['max_residual'] for r in rows),max_closure=max(r['max_closure'] for r in rows),
        extended_coupling_runs=len(extrows),extended_coupling_seconds=sum(r['seconds'] for r in extrows),
        extended_coupling_finite_modes=0,extended_final_D_relative_change=extrows[3]['comparison_previous']['C']['relative_norm'],
        validation_status='pre-release; matrix D convergence gate remains unmet; no publication curve certification',
        axes=expected,artifacts=artifacts,core_unchanged=True,benchmark_digest_matches=True,
        source_scope='new core/tests/examples/scripts/docs/benchmarks, pyproject/license/README, two legacy AST regression sources')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
