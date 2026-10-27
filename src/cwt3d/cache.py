"""Opt-in, local NPZ+JSON coupling snapshots; never deserialize Python objects.

Construction/key/load do not create directories. Only explicit build misses or
store write. Corrupt/mismatched entries raise CacheError (never silent success).
Expiration is a miss. No automatic prune: invoke prune/clear explicitly.
Checksums detect damage, not an adversary able to rewrite both data and manifest;
use a directory whose writers you trust. Reading cannot execute pickled code.
"""
from dataclasses import fields
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import zipfile

import numpy as np
import scipy
from .coupling import CouplingResult, build_coupling, _pack
from .geometry import CircleHole, RightTriangleHole, SampledPermittivity
from .waveguide import TEMode, _positive_scalar
from .quadrature import _reciprocity_option

SCHEMA = 'cwt3d-coupling-v1'
DEFAULT_MAX_BYTES = 256*1024*1024
DEFAULT_MAX_AGE = 7*24*60*60
_BUILD_SIGNATURE = inspect.signature(build_coupling)
_ARRAY_NAMES = ('C1D', 'Crad', 'C2D', 'C2D_green', 'C2D_local', 'C', 'I', 'J',
                'K', 'xi', 'xi_orders', 'aux_orders', 'mu', 'nu', 'varsigma')


class CacheError(ValueError):
    """Malformed, damaged or mismatched cache entry; explicit clear to rebuild."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _array(value):
    a = np.ascontiguousarray(value)
    if a.dtype.kind not in 'iufc':
        raise TypeError('cache arrays must be numeric')
    return {'shape': list(a.shape), 'dtype': a.dtype.str,
            'sha256': sha256(a.tobytes()).hexdigest()}


def _canonical(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return {'float': value.hex()}
    if isinstance(value, complex):
        return {'complex': [value.real.hex(), value.imag.hex()]}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return _array(value)
    raise TypeError(f'unsupported cache key type: {type(value).__name__}')


def algorithm_digest():
    """Actual source digest, not callable identity/repr or a manual version only."""
    root = Path(__file__).parent
    digest = sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _document(mode, period_um, pc_layers, options):
    if not isinstance(mode, TEMode):
        raise TypeError('mode must be a normalized TEMode')
    bound = _BUILD_SIGNATURE.bind(mode, period_um, pc_layers, **options)
    bound.apply_defaults()
    bound.arguments['reciprocity'] = _reciprocity_option(bound.arguments['reciprocity'])
    # Match the real build's scalar semantics, not the caller's container/dtype.
    # Do not generalize this normalization to geometry or solved-field arrays.
    period_um = _positive_scalar(period_um, 'period_um')
    guide = mode.guide
    geometry = []
    for index, shape in sorted(dict(pc_layers).items()):
        if isinstance(shape, SampledPermittivity):
            state = {'period_um': shape.period_um, 'values': shape.values}
        elif isinstance(shape, (CircleHole, RightTriangleHole)):
            state = {f.name: getattr(shape, f.name) for f in fields(shape)}
        else:
            raise TypeError('unsupported cache geometry')
        geometry.append([_canonical(index), type(shape).__name__,
                         {k: _canonical(v) for k, v in state.items()}])
    # Include the actual interpolation mesh/materials and normalized field,
    # not just beta or a caller-assigned mode index. Altered solved shapes miss.
    state = {'eps': guide.eps, 'thickness_um': guide.thickness_um,
             'interfaces_um': guide.interfaces_um, 'k0': guide.k0,
             'period_um': guide.period_um, 'nodes': guide._nodes,
             'lengths': guide._lengths, 'materials': guide._materials,
             'beta': mode.beta, 'value': mode._value, 'field': mode._field}
    return {'schema': SCHEMA, 'algorithm': algorithm_digest(),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'byteorder': sys.byteorder,
            'guide_mode': {k: _canonical(v) for k,v in state.items()},
            'geometry': geometry, 'period_um': _canonical(period_um),
            'options': {k: _canonical(v) for k,v in bound.arguments.items()
                        if k not in ('mode', 'period_um', 'pc_layers')}}


def _tuple(value):
    return tuple(_tuple(v) for v in value) if isinstance(value, list) else value


def _validate_metadata(metadata, document):
    """Reject accidental public store mis-keying before writing/returning a hit."""
    if metadata.get('provenance') != _json(document):
        raise CacheError('result metadata provenance does not match complete key inputs')
    options = document['options']
    if type(metadata.get('reciprocity')) is not bool:
        raise CacheError('result metadata reciprocity must be boolean')
    for name in ('cutoff', 'quadrature_order', 'fourier_order', 'backend', 'reciprocity'):
        if _canonical(metadata[name]) != options[name]:
            raise CacheError('result metadata does not match key numerical controls')
    def real(value):
        return float.fromhex(value['float']) if isinstance(value, dict) else float(value)
    period = real(document['period_um'])
    beta0 = 2*np.pi/period
    beta = real(document['guide_mode']['beta'])
    expected = dict(period_um=period, beta0=beta0,
                    k0=real(document['guide_mode']['k0']),
                    relative_detuning=(beta-beta0)/beta0, units='um^-1')
    if any(metadata[name] != value for name, value in expected.items()):
        raise CacheError('result metadata does not match key structure/period')
    ids = [item[0] for item in document['geometry']]
    if _json(metadata['pc_layers']) != _json(ids):
        raise CacheError('result metadata does not match PC layer indices')
    shapes = [[item[0], item[2]['values']['shape']] for item in document['geometry']
              if item[1] == 'SampledPermittivity']
    if _json(metadata['sampling_shapes']) != _json(shapes):
        raise CacheError('result metadata does not match sampled dimensions')


def _check_archive(path, max_bytes):
    """Bound expanded bytes AND NPY declared shapes before np.load allocates."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) != len(_ARRAY_NAMES)+1
                or {x.filename for x in members} != {n+'.npy' for n in (*_ARRAY_NAMES, 'manifest')}
                or sum(x.file_size for x in members) > max_bytes):
            raise CacheError('entry exceeds cache schema/expanded byte budget')
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise CacheError('unsupported NPY format')
                if (dtype.hasobject or dtype.kind not in 'iufc'
                        or any(d < 0 for d in shape)
                        or math.prod(shape)*dtype.itemsize != member.file_size-stream.tell()):
                    raise CacheError('unsafe or mismatched NPY payload size/type')


class CouplingCache:
    """Explicit directory ownership prefix cwt3d-<SHA256>.npz.

    Default age 7 days, size 256 MiB. Size limits bound each expanded entry and
    explicit prune's total retained storage. readonly forbids every write/delete.
    Concurrent misses may duplicate work; atomic replace keeps complete snapshots,
    not guaranteed concurrent success. Windows open readers may block replacement
    with PermissionError/WinError 5. Callers must serialize access or retry errors;
    no cross-process lock or automatic retry is provided.
    Private temp files are always in directory and cleaned on normal failure.
    """
    def __init__(self, directory, *, readonly=False, max_age=DEFAULT_MAX_AGE,
                 max_bytes=DEFAULT_MAX_BYTES):
        self.directory = Path(directory)
        self.readonly = bool(readonly)
        if max_age is not None and (isinstance(max_age, bool) or not np.isfinite(max_age) or max_age < 0):
            raise ValueError('max_age must be None or finite nonnegative seconds')
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError('max_bytes must be a positive integer')
        self.max_age, self.max_bytes = max_age, max_bytes

    def key(self, mode, period_um, pc_layers, **options):
        return sha256(_json(_document(mode, period_um, pc_layers, options)).encode()).hexdigest()

    def document(self, mode, period_um, pc_layers, **options):
        """Canonical JSON-compatible provenance for explicit store calls."""
        return _document(mode, period_um, pc_layers, options)

    def path(self, key):
        if not isinstance(key, str) or re.fullmatch('[0-9a-f]{64}', key) is None:
            raise ValueError('key must be lowercase SHA256 hex')
        return self.directory/f'cwt3d-{key}.npz'

    def _writable(self):
        if self.readonly:
            raise PermissionError('cache is readonly')

    def load(self, key):
        path = self.path(key)
        if not path.exists():
            return None
        if path.is_symlink():
            raise CacheError('cache entry must not be a symlink')
        if self.max_age is not None and time.time()-path.stat().st_mtime > self.max_age:
            return None
        try:
            _check_archive(path, self.max_bytes)
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {*_ARRAY_NAMES, 'manifest'}:
                    raise CacheError('unexpected array names')
                raw = data['manifest']
                if raw.dtype != np.uint8 or raw.ndim != 1:
                    raise CacheError('manifest must be JSON bytes')
                manifest = json.loads(raw.tobytes())
                if manifest['schema'] != SCHEMA or manifest['key'] != key:
                    raise CacheError('cache schema/key mismatch')
                if sha256(_json(manifest['document']).encode()).hexdigest() != key:
                    raise CacheError('cache key document mismatch')
                if (manifest['document']['algorithm'] != algorithm_digest()
                        or manifest['document']['numpy'] != np.__version__
                        or manifest['document']['scipy'] != scipy.__version__
                        or manifest['document']['byteorder'] != sys.byteorder):
                    return None
                arrays = {name: data[name] for name in _ARRAY_NAMES}
                if any(_array(a) != manifest['arrays'][name] for name,a in arrays.items()):
                    raise CacheError('array checksum/shape/dtype mismatch')
                metadata = manifest['metadata']
                check = sha256(_json(metadata).encode()).hexdigest()
                if check != manifest['metadata_sha256']:
                    raise CacheError('metadata checksum mismatch')
                _validate_metadata(metadata, manifest['document'])
                return CouplingResult(_pack(**arrays), **{k:_tuple(v) for k,v in metadata.items()})
        except CacheError:
            raise
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError,
                zipfile.BadZipFile, EOFError, RuntimeError) as exc:
            raise CacheError('invalid coupling cache entry') from exc

    def store(self, key, result, *, document):
        self._writable()
        target = self.path(key)

        if sha256(_json(document).encode()).hexdigest() != key:
            raise CacheError('store document/key mismatch')
        arrays = {name: getattr(result, name) for name in _ARRAY_NAMES}
        metadata = {f.name: getattr(result, f.name) for f in fields(result) if f.name != '_arrays'}
        _validate_metadata(metadata, document)
        manifest = {'schema': SCHEMA, 'key': key, 'document': document,
                    'metadata': metadata, 'metadata_sha256':sha256(_json(metadata).encode()).hexdigest(),
                    'arrays': {name:_array(a) for name,a in arrays.items()}}
        raw = _json(manifest).encode()
        if sum(a.nbytes for a in arrays.values())+len(raw)+16384 > self.max_bytes:
            raise CacheError('entry exceeds cache byte budget')
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='.cwt3d-', suffix='.tmp', delete=False) as stream:
                temp = Path(stream.name)
                np.savez(stream, manifest=np.frombuffer(raw, dtype=np.uint8), **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    def build(self, mode, period_um, pc_layers, **options):
        document = _document(mode, period_um, pc_layers, options)
        key = sha256(_json(document).encode()).hexdigest()
        result = self.load(key)
        if result is not None:
            return result
        self._writable()
        result = build_coupling(mode, period_um, pc_layers, **options)
        self.store(key, result, document=document)
        return result

    def clear(self, key):
        self._writable()
        self.path(key).unlink(missing_ok=True)

    def prune(self):
        """Delete recognized owned entries, expired then oldest.

        Recognition checks ZIP/NPY layout and schema/key document, not array or
        metadata checksums/provenance. Thus recognized checksum-damaged entries
        can be deleted by age/size policy. Unrecognized files, symlinks and
        crash-left temp files are retained; max_bytes counts recognized files only.
        """
        self._writable()
        entries = []
        if not self.directory.exists():
            return []
        for path in self.directory.iterdir():
            match = re.fullmatch(r'cwt3d-([0-9a-f]{64})\.npz', path.name)
            if not match or path.is_symlink() or not path.is_file():
                continue
            try:
                _check_archive(path, max(self.max_bytes, DEFAULT_MAX_BYTES))
                with np.load(path, allow_pickle=False) as data:
                    manifest = json.loads(data['manifest'].tobytes())
                key = match[1]
                if (manifest['schema'] != SCHEMA or manifest['key'] != key or
                    sha256(_json(manifest['document']).encode()).hexdigest() != key):
                    continue
            except (OSError, ValueError, KeyError, TypeError, AttributeError,
                    OverflowError, zipfile.BadZipFile, EOFError, RuntimeError):
                continue
            stat = path.stat()
            entries.append((stat.st_mtime, stat.st_size, key, path))
        total = sum(e[1] for e in entries)
        removed = []
        now = time.time()
        for mtime, size, key, path in sorted(entries):
            if (self.max_age is not None and now-mtime > self.max_age) or total > self.max_bytes:
                path.unlink()
                total -= size
                removed.append(key)
        return removed
