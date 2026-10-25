"""Safe explicit persistence; no pickle, hidden writes or shared headers."""
from dataclasses import replace
import json
import os
import numpy as np
import pytest
from cwt3d.coupling import build_coupling
from cwt3d.geometry import SampledPermittivity
from test_coupling import physical


def cache_class():
    from cwt3d.cache import CouplingCache
    return CouplingCache


def test_roundtrip_and_explicit_constructor(tmp_path, monkeypatch):
    cache = cache_class()(tmp_path/'new')
    assert not cache.directory.exists()
    m, a, h = physical(dual=True)
    r = cache.build(m, a, h, backend='prepared', quadrature_order=6, fourier_order=24)
    key = cache.key(m, a, h, backend='prepared', quadrature_order=6, fourier_order=24)
    assert len(list(cache.directory.iterdir())) == 1
    import cwt3d.cache as module
    monkeypatch.setattr(module, 'build_coupling', lambda *a, **k: pytest.fail('cache hit recomputed'))
    other = cache_class()(cache.directory, readonly=True)
    s = other.build(m, a, h, backend='prepared', quadrature_order=6, fourier_order=24)
    assert s == r  # immutable bytes and scalar tuples, not ndarray equality
    x = s.C
    assert isinstance(x.base, bytes)
    with pytest.raises(ValueError):
        x.setflags(write=True)
    x.shape = (16,)
    assert s.C.shape == r.C.shape == (4, 4)
    assert other.load(key) == r


@pytest.mark.parametrize('backend', ['reference', 'prepared'])
@pytest.mark.parametrize('scalar', [float, np.float64, np.array],
                         ids=['python-float', 'numpy-float64', 'numpy-0d'])
def test_scalar_period_canonical_roundtrip(tmp_path, monkeypatch, backend, scalar):
    import cwt3d.cache as module
    m, a, h = physical()
    period = scalar(a)
    options = dict(backend=backend, quadrature_order=4)
    cache = cache_class()(tmp_path/'entries')
    direct = build_coupling(m, period, h, **options)
    canonical = build_coupling(m, float(a), h, **options)
    assert cache.build(m, period, h, **options) == direct
    assert direct == canonical  # numerical identity, not raw input dtype identity
    key = cache.key(m, period, h, **options)
    assert key == cache.key(m, float(a), h, **options)
    assert cache.document(m, period, h, **options) == cache.document(m, float(a), h, **options)
    cache.store(key, direct, document=cache.document(m, float(a), h, **options))
    assert cache.load(key) == direct
    cache.clear(key)
    assert cache.load(key) is None
    assert cache.build(m, period, h, **options) == canonical
    monkeypatch.setattr(module, 'build_coupling', lambda *a, **k: pytest.fail('hit recomputed'))
    assert cache.build(m, float(a), h, **options) == canonical
    unrelated = cache.directory/'unrelated.txt'
    unrelated.write_bytes(b'keep')
    cache.clear(key)
    assert cache.load(key) is None
    assert unrelated.read_bytes() == b'keep'
    assert list(cache.directory.iterdir()) == [unrelated]


@pytest.mark.parametrize('period', [[1.], np.array([1.]), 1+0j, np.array(1+0j),
                                    True, np.array(True), 0., -1., np.nan, np.inf])
def test_cache_period_rejects_invalid_scalars(tmp_path, period):
    m, _, h = physical()
    cache = cache_class()(tmp_path/'entries')
    for call in (build_coupling, cache.key, cache.document, cache.build):
        with pytest.raises(ValueError, match='finite positive real scalar'):
            call(m, period, h)
    assert not cache.directory.exists()


def test_prune_recognized_checksum_damage_policy(tmp_path):
    from cwt3d.cache import CacheError
    cache = cache_class()(tmp_path)
    m, a, h = physical()
    result = cache.build(m, a, h, quadrature_order=4)
    key = cache.key(m, a, h, quadrature_order=4)
    path = cache.path(key)
    with np.load(path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    payload['C'][0, 0] += .01  # valid ZIP/layout/key, invalid numerical checksum
    with path.open('wb') as stream:
        np.savez(stream, **payload)
    with pytest.raises(CacheError, match='checksum'):
        cache.load(key)
    assert cache.prune() == []  # fresh recognized entry retained by age/size
    unrelated = tmp_path/'unrelated.npz'
    unrelated.write_bytes(b'keep')
    malformed = cache.path('b'*64)
    malformed.write_bytes(b'not zip')
    os.utime(path, (1, 1))
    assert cache.prune() == [key]  # prune is ownership/retention, not checksum audit
    assert cache.load(key) is None
    assert unrelated.read_bytes() == b'keep'
    assert malformed.read_bytes() == b'not zip'
    assert result.C.shape == (4, 4)


def test_complete_keys(tmp_path, monkeypatch):
    import cwt3d.cache as module
    cache = cache_class()(tmp_path)
    m, a, h = physical(dual=True)
    key = cache.key(m, a, h)
    assert key == cache.key(m, a, dict(reversed(list(h.items()))))
    assert key == cache.key(m, a, h, cutoff=1, backend='reference')
    changes = [dict(cutoff=3), dict(quadrature_order=13), dict(fourier_order=32),
               dict(beta_rtol=.01), dict(backend='prepared')]
    assert len({key, *(cache.key(m, a, h, **k) for k in changes)}) == 6
    assert key != cache.key(m, a*1.001, h)
    assert key != cache.key(m, a, {1:h[1]})
    shifted = dict(h); shifted[1] = replace(h[1], center_um=(a*.4, a*.5))
    assert key != cache.key(m, a, shifted)
    complex_h = dict(h); complex_h[1] = replace(h[1], eps_hole=h[1].eps_hole-.01j)
    assert key != cache.key(m, a, complex_h)
    v = np.full((4, 5), 8., complex)
    k1 = cache.key(m, a, {1:SampledPermittivity(v, (a,a))})
    v[0,0] += 1
    assert k1 != cache.key(m, a, {1:SampledPermittivity(v, (a,a))})
    assert k1 != cache.key(m, a, {1:SampledPermittivity(np.full((5,4),8.), (a,a))})
    from cwt3d.waveguide import TEWaveguide
    modes = TEWaveguide([1.,9.,1.], [.8], k0=6).solve_modes()
    assert len(modes) > 1
    assert cache.key(modes[0], a, {}) != cache.key(modes[1], a, {})
    # Same beta but different actual solved field must not collide.
    import copy
    altered = copy.copy(m); altered._field = m._field.copy(); altered._field[1] *= .99
    assert key != cache.key(altered, a, h)
    old = module.algorithm_digest()
    monkeypatch.setattr(module, 'algorithm_digest', lambda: old+'changed')
    assert key != cache.key(m, a, h)


def test_corrupt_mismatched_stale_and_untrusted(tmp_path):
    from cwt3d.cache import CacheError
    cache = cache_class()(tmp_path)
    m, a, h = physical()
    key = cache.key(m,a,h,quadrature_order=4)
    r = cache.build(m,a,h,quadrature_order=4)
    path = cache.path(key)
    path.write_bytes(b'not npz')
    with pytest.raises(CacheError):
        cache.load(key)
    cache.store(key,r,document=cache.document(m,a,h,quadrature_order=4))
    other = 'a'*64
    cache.path(other).write_bytes(path.read_bytes())
    with pytest.raises(CacheError):
        cache.load(other)
    # Object payload cannot be loaded even if a file has the right filename.
    with path.open('wb') as f:
        np.savez(f, manifest=np.array([{'execute':'never'}], dtype=object))
    with pytest.raises(CacheError):
        cache.load(key)
    cache.store(key,r,document=cache.document(m,a,h,quadrature_order=4))
    os.utime(path, (1,1))
    assert cache.load(key) is None
    assert cache.build(m,a,h,quadrature_order=4) == r


def test_atomic_failure_readonly_and_scoped_cleanup(tmp_path, monkeypatch):
    import cwt3d.cache as module
    cache = cache_class()(tmp_path)
    m,a,h = physical()
    key = cache.key(m,a,h,quadrature_order=4)
    r = build_coupling(m,a,h,quadrature_order=4)
    cache.store(key,r,document=cache.document(m,a,h,quadrature_order=4))
    before = cache.path(key).read_bytes()
    def fail(*args):
        raise OSError('injected replace failure')
    with monkeypatch.context() as context:
        context.setattr(module.os, 'replace', fail)
        with pytest.raises(OSError):
            cache.store(key,r,document=cache.document(m,a,h,quadrature_order=4))
    assert cache.path(key).read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))
    readonly = cache_class()(tmp_path, readonly=True)
    assert readonly.load(key) == r
    with pytest.raises(PermissionError):
        readonly.store(key,r,document=cache.document(m,a,h,quadrature_order=4))
    with pytest.raises(PermissionError):
        readonly.clear(key)
    unrelated = tmp_path/'unrelated.npz'; unrelated.write_bytes(b'keep')
    impostor = tmp_path/('cwt3d-'+ 'b'*64 + '.npz'); impostor.write_bytes(b'keep')
    os.utime(cache.path(key),(1,1))
    removed = cache.prune()
    assert removed == [key]
    assert unrelated.read_bytes() == impostor.read_bytes() == b'keep'
    assert not cache.path(key).exists()
    with pytest.raises(ValueError):
        cache.path('../unsafe')


def test_size_prune_and_readonly_miss(tmp_path):
    from cwt3d.cache import CacheError
    cache = cache_class()(tmp_path, max_age=None)
    m,a,h = physical()
    keys=[]
    for q in (4,5):
        cache.build(m,a,h,quadrature_order=q)
        keys.append(cache.key(m,a,h,quadrature_order=q))
    sizes=[cache.path(key).stat().st_size for key in keys]
    os.utime(cache.path(keys[0]),(100,100))
    limited=cache_class()(tmp_path,max_age=None,max_bytes=max(sizes))
    assert limited.prune() == [keys[0]]
    assert cache.path(keys[1]).exists()
    readonly=cache_class()(tmp_path,readonly=True)
    with pytest.raises(PermissionError):
        readonly.build(m,a,h,quadrature_order=8)
    with pytest.raises(PermissionError):
        readonly.prune()
    tiny=cache_class()(tmp_path,max_bytes=10)
    with pytest.raises(CacheError,match='budget'):
        tiny.load(keys[1])


def test_checksum_and_forged_huge_header_rejected(tmp_path):
    from cwt3d.cache import CacheError
    import io
    import zipfile
    cache=cache_class()(tmp_path)
    m,a,h=physical()
    cache.build(m,a,h,quadrature_order=4)
    key=cache.key(m,a,h,quadrature_order=4)
    path=cache.path(key)
    with np.load(path,allow_pickle=False) as source:
        arrays={name:source[name] for name in source.files}
    arrays['C']=arrays['C'].copy(); arrays['C'][0,0] += .01
    with path.open('wb') as stream:
        np.savez(stream,**arrays)
    with pytest.raises(CacheError,match='checksum'):
        cache.load(key)
    # Valid ZIP byte budget but enormous declared ndarray: must fail before allocation.
    buffer=io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer,dict(descr='<c16',fortran_order=False,shape=(2**50,)))
    with zipfile.ZipFile(path,'w') as archive:
        for name,array in arrays.items():
            item=io.BytesIO(); np.save(item,array,allow_pickle=False)
            archive.writestr(name+'.npy',buffer.getvalue() if name=='C' else item.getvalue())
    with pytest.raises(CacheError,match='payload'):
        cache.load(key)


def test_store_rejects_wrong_precision_provenance(tmp_path):
    from cwt3d.cache import CacheError
    m,a,h=physical()
    cache=cache_class()(tmp_path)
    key=cache.key(m,a,h,quadrature_order=4)
    wrong=build_coupling(m,a,h,quadrature_order=5)
    with pytest.raises(CacheError,match='metadata'):
        cache.store(key,wrong,document=cache.document(m,a,h,quadrature_order=4))
    assert not cache.path(key).exists()


@pytest.mark.parametrize('change', ['geometry', 'field', 'period', 'precision', 'tolerance'])
def test_store_rejects_real_input_mismatch(tmp_path, change):
    import copy
    from cwt3d.cache import CacheError
    m, a, h = physical()
    cache = cache_class()(tmp_path/'entries')
    options = dict(quadrature_order=4)
    other_m, other_a, other_h = m, a, dict(h)
    other_options = dict(options)
    if change == 'geometry':
        other_h[1] = replace(h[1], center_um=(.41*a, .47*a))
    elif change == 'field':
        other_m = copy.copy(m)
        other_m._field = m._field.copy()
        other_m._field[1] *= .99
        assert other_m.beta == m.beta
    elif change == 'period':
        # Geometry/guide are scaled consistently by the real fixture.
        other_m, other_a, other_h = physical(scale=1.001)
    elif change == 'precision':
        other_options['quadrature_order'] = 5
    else:
        other_options['beta_rtol'] = .01
    requested = build_coupling(m, a, h, **options)
    wrong = build_coupling(other_m, other_a, other_h, **other_options)
    if change in ('geometry', 'field'):
        assert np.max(np.abs(requested.C-wrong.C)) > 1e-8
    key = cache.key(m, a, h, **options)
    assert key != cache.key(other_m, other_a, other_h, **other_options)
    with pytest.raises(CacheError, match='metadata|provenance'):
        cache.store(key, wrong, document=cache.document(m, a, h, **options))
    assert not cache.directory.exists()
    cache.store(key, requested, document=cache.document(m, a, h, **options))
    assert cache.build(m, a, h, **options) == requested
    unrelated = cache.directory/'unrelated.txt'
    unrelated.write_bytes(b'keep')
    cache.clear(key)
    assert not cache.path(key).exists()
    assert unrelated.read_bytes() == b'keep'


def test_provenance_snapshot_readonly_and_load_validation(tmp_path):
    from dataclasses import FrozenInstanceError
    from hashlib import sha256
    from cwt3d.cache import CacheError
    m, a, h = physical()
    cache = cache_class()(tmp_path)
    options = dict(quadrature_order=4)
    result = build_coupling(m, a, h, **options)
    document = cache.document(m, a, h, **options)
    key = cache.key(m, a, h, **options)
    canonical = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert result.provenance == canonical
    assert isinstance(result.provenance, str)
    with pytest.raises(FrozenInstanceError):
        result.provenance = ''
    # Public document edits do not mutate the result's immutable snapshot.
    document['geometry'].clear()
    assert result.provenance == canonical
    document = cache.document(m, a, h, **options)
    with pytest.raises(CacheError, match='provenance'):
        cache.store(key, replace(result, provenance=''), document=document)
    cache.store(key, result, document=document)
    restored = cache_class()(tmp_path, readonly=True).load(key)
    assert restored == result
    assert restored._arrays == result._arrays
    assert isinstance(restored.C.base, bytes)
    path = cache.path(key)
    with np.load(path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    manifest = json.loads(payload['manifest'].tobytes())
    assert manifest['metadata']['provenance'] == canonical
    # An internally consistent metadata checksum alone must not authenticate
    # provenance for different inputs. This is a defensive read-path regression,
    # not a claim to resist a writer who rewrites provenance AND the document.
    manifest['metadata']['provenance'] = '{}'
    encoded = json.dumps(manifest['metadata'], sort_keys=True, separators=(',', ':')).encode()
    manifest['metadata_sha256'] = sha256(encoded).hexdigest()
    payload['manifest'] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    with path.open('wb') as stream:
        np.savez(stream, **payload)
    with pytest.raises(CacheError, match='provenance'):
        cache.load(key)
    # Without updating checksum, provenance damage is caught at checksum gate.
    manifest['metadata']['provenance'] = canonical
    payload['manifest'] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    with path.open('wb') as stream:
        np.savez(stream, **payload)
    with pytest.raises(CacheError, match='checksum'):
        cache.load(key)
    cache.clear(key)


def test_sampled_metadata_and_numeric_version_keys(tmp_path, monkeypatch):
    import cwt3d.cache as module
    m,a,_=physical()
    h={1:SampledPermittivity(np.tile([[8.,10.],[10.,8.]],(4,4)),(a,a))}
    cache=cache_class()(tmp_path)
    key=cache.key(m,a,h,backend='prepared',quadrature_order=4,fourier_order=17)
    r=cache.build(m,a,h,backend='prepared',quadrature_order=4,fourier_order=17)
    s=cache.load(key)
    assert s == r
    assert s.sampling_shapes == ((1,(8,8)),)
    assert s.fourier_order == 17
    with monkeypatch.context() as context:
        context.setattr(module.np,'__version__',module.np.__version__+'-changed')
        assert cache.key(m,a,h,backend='prepared',quadrature_order=4,fourier_order=17) != key
        assert cache.load(key) is None
    # Actual stack/k0/thickness changes, not only selected order controls.
    for scale in (1.1,2.):
        changed,b,geometry=physical(scale=scale)
        assert cache.key(changed,b,geometry) != cache.key(m,a,h)
