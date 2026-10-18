"""Calibration session: config, samples, revisions and on-disk persistence.

Layout:
  <session_dir>/session.yaml
  <session_dir>/samples/<sample_id>.png   raw image, no overlay
  <session_dir>/samples/<sample_id>.yaml  per-sample metadata
  <session_dir>/results/<name>/extrinsics.yaml, report.yaml
Floats are stored at full float64 precision; display rounding is never written back.
"""
import os
import time
import uuid

import cv2
import numpy as np
import yaml

SCHEMA_VERSION = 1


def atomic_write_text(path, text):
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def to_plain(obj):
    """Convert numpy scalars/arrays (recursively) to plain Python for safe_dump."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def atomic_write_yaml(path, data):
    atomic_write_text(path, yaml.safe_dump(to_plain(data), sort_keys=False,
                                           default_flow_style=None))


def atomic_write_png(path, image):
    tmp = f'{path}.tmp.{os.getpid()}.png'
    if not cv2.imwrite(tmp, image):
        raise IOError(f'failed to write {tmp}')
    os.replace(tmp, path)


class Session:
    def __init__(self, session_dir, session_id, config, frames, versions):
        self.dir = session_dir
        self.id = session_id
        self.config = config            # dict: columns, rows, square_size_mm, mode, intrinsics, ...
        self.frames = frames            # dict: base, eef, camera
        self.versions = versions
        self.config_revision = 1
        self.dataset_revision = 0
        self.next_sequence = 1
        self.samples = []               # list of dicts, ordered by sequence

    @staticmethod
    def create(data_root, name, config, frames, versions):
        sid = name.strip() if name and name.strip() else time.strftime('session_%Y%m%d_%H%M%S')
        sdir = os.path.join(data_root, sid)
        if os.path.exists(sdir):
            sid = f'{sid}_{uuid.uuid4().hex[:4]}'
            sdir = os.path.join(data_root, sid)
        os.makedirs(os.path.join(sdir, 'samples'))
        os.makedirs(os.path.join(sdir, 'results'))
        s = Session(sdir, sid, config, frames, versions)
        s.flush()
        return s

    # ---------- samples ----------
    def valid_samples(self):
        return [s for s in self.samples if not s['excluded']]

    def add_sample(self, sample, image):
        seq = self.next_sequence
        sample_id = f's{seq:04d}_{uuid.uuid4().hex[:6]}'
        sample = dict(sample, sample_id=sample_id, sequence=seq, excluded=False,
                      image_file=f'samples/{sample_id}.png',
                      config_revision=self.config_revision)
        atomic_write_png(os.path.join(self.dir, sample['image_file']), image)
        atomic_write_yaml(os.path.join(self.dir, 'samples', f'{sample_id}.yaml'), sample)
        self.samples.append(sample)
        self.next_sequence += 1
        self.dataset_revision += 1
        self.flush()
        return sample

    def exclude(self, sample_id):
        if sample_id == 'last':
            valid = self.valid_samples()
            if not valid:
                return None, 'no valid samples'
            target = max(valid, key=lambda s: s['sequence'])
        else:
            target = next((s for s in self.samples if s['sample_id'] == sample_id), None)
            if target is None:
                return None, f'unknown sample {sample_id}'
            if target['excluded']:
                return None, f'sample {sample_id} already excluded'
        target['excluded'] = True
        atomic_write_yaml(os.path.join(self.dir, 'samples', f"{target['sample_id']}.yaml"), target)
        self.dataset_revision += 1
        self.flush()
        return target, ''

    # ---------- persistence ----------
    def flush(self):
        atomic_write_yaml(os.path.join(self.dir, 'session.yaml'), {
            'schema_version': SCHEMA_VERSION,
            'session_id': self.id,
            'config': self.config,
            'frames': self.frames,
            'config_revision': self.config_revision,
            'dataset_revision': self.dataset_revision,
            'next_sequence': self.next_sequence,
            'sample_index': [s['sample_id'] for s in self.samples],
            'versions': self.versions,
        })

    @staticmethod
    def load(session_dir):
        path = os.path.join(session_dir, 'session.yaml')
        with open(path) as f:
            d = yaml.safe_load(f)
        if d.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {d.get('schema_version')}")
        s = Session(session_dir, d['session_id'], d['config'], d['frames'], d.get('versions', {}))
        s.config_revision = int(d['config_revision'])
        s.dataset_revision = int(d['dataset_revision'])
        s.next_sequence = int(d['next_sequence'])
        missing = []
        for sid in d['sample_index']:
            with open(os.path.join(session_dir, 'samples', f'{sid}.yaml')) as f:
                smp = yaml.safe_load(f)
            if not os.path.isfile(os.path.join(session_dir, smp['image_file'])):
                missing.append(smp['image_file'])
            s.samples.append(smp)
        if missing:
            raise ValueError(f'missing images: {missing[:3]}')
        return s

    def save_result(self, name, overwrite, extrinsics, report):
        name = name.strip() if name and name.strip() else time.strftime('result_%Y%m%d_%H%M%S')
        if os.sep in name or name in ('.', '..'):
            raise ValueError('result name must be a plain directory name')
        rdir = os.path.join(self.dir, 'results', name)
        if os.path.exists(rdir) and not overwrite:
            raise FileExistsError(f'{rdir} exists; enable overwrite or choose another name')
        os.makedirs(rdir, exist_ok=True)
        atomic_write_yaml(os.path.join(rdir, 'extrinsics.yaml'), extrinsics)
        atomic_write_yaml(os.path.join(rdir, 'report.yaml'), report)
        return rdir


def T_to_list(T):
    return np.asarray(T, float).tolist()


def list_to_T(lst):
    return np.array(lst, dtype=float)
