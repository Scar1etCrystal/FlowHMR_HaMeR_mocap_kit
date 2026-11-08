"""Minimal stand-ins for the toyxyz app framework.

The vendored modules import a handful of helpers that belong to their desktop
application (project storage, scene camera validation, artifact bookkeeping).
The export adapter only needs identity-like behaviour from them, so they are
reimplemented here instead of vendoring the whole app.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def validate_transform(matrix):
    """Return a finite rigid 4x4 transform; the writer applies it to the root."""
    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError('Invalid scene transform')
    if np.max(np.abs(value[:3, :3] @ value[:3, :3].T - np.eye(3))) > 1e-5:
        raise ValueError('Scene transform must be rigid')
    return value


def resolve_reference(path):
    return Path(path)


def file_stat(path):
    st = os.stat(path)
    return [int(st.st_size), int(st.st_mtime_ns)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def validate_name(name):
    text = str(name).strip()
    if not text or not text.isascii() or any(c in text for c in ':\x00\x01'):
        raise ValueError('Invalid rig name')
    return text


def reference_frame(direction):
    """Orthonormal frame whose Y axis follows ``direction`` (bone overlay basis)."""
    v = np.asarray(direction, dtype=float)
    length = np.linalg.norm(v)
    if length < 1e-12:
        raise ValueError('Zero-length bone direction')
    y = v / length
    reference = np.array([0.0, 1.0, 0.0]) if abs(y[1]) < 0.99 else np.array([0.0, 0.0, 1.0])
    x = np.cross(reference, y)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(np.array([1.0, 0.0, 0.0]), y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack((x, y, z), axis=1)
