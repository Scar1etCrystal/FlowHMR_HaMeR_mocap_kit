"""Structural validation for the Humanoid52 FBX written by the vendored writer.

The toyxyz writer targets Blender's importer and produces a file the Autodesk
SDK rejects, so validation parses the binary records directly and, when a
Blender executable is available, also re-imports the file and checks bones.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import struct
import tempfile
from pathlib import Path

EXPECTED_BONES = 52
EXPECTED_MARKERS = 13
EXPECTED_CURVES = 159

BLENDER_CANDIDATES = (
    r'D:\blender-3.2.1-windows-x64\blender.exe',
    r'C:\Program Files\Blender Foundation\Blender\blender.exe',
    r'C:\Program Files\Blender Foundation\Blender 4.0\blender.exe',
)


class Record:
    __slots__ = ('name', 'props', 'children')

    def __init__(self, name, props, children):
        self.name = name
        self.props = props
        self.children = children


def _read_prop(data, offset):
    kind = data[offset:offset + 1]
    offset += 1
    if kind == b'Y':
        return struct.unpack_from('<h', data, offset)[0], offset + 2
    if kind == b'C':
        return data[offset] != 0, offset + 1
    if kind == b'I':
        return struct.unpack_from('<i', data, offset)[0], offset + 4
    if kind == b'F':
        return struct.unpack_from('<f', data, offset)[0], offset + 4
    if kind == b'D':
        return struct.unpack_from('<d', data, offset)[0], offset + 8
    if kind == b'L':
        return struct.unpack_from('<q', data, offset)[0], offset + 8
    if kind == b'S':
        size = struct.unpack_from('<I', data, offset)[0]
        return data[offset + 4:offset + 4 + size], offset + 4 + size
    if kind in b'fdlib':
        count, encoding, size = struct.unpack_from('<III', data, offset)
        raw = data[offset + 12:offset + 12 + size]
        fmt = {b'f': '<f', b'd': '<d', b'l': '<q', b'i': '<i', b'b': '<?'}[kind]
        if encoding:
            import zlib
            raw = zlib.decompress(raw)
        values = struct.unpack('<%d%s' % (count, fmt[1]), raw)
        return values, offset + 12 + size
    raise ValueError(f'Unsupported FBX property type {kind!r} at {offset}')


def _read_records(data, offset, end):
    out = []
    while offset < end - 13:
        rec_end, nprops, prop_len, name_len = struct.unpack_from('<IIIB', data, offset)
        if rec_end == 0:
            break
        pos = offset + 13
        name = data[pos:pos + name_len].decode('utf-8', 'replace')
        pos += name_len
        prop_end = pos + prop_len
        props = []
        while pos < prop_end:
            value, pos = _read_prop(data, pos)
            props.append(value)
        children = []
        if pos < rec_end:
            children = _read_records(data, pos, rec_end)
        out.append(Record(name, props, children))
        offset = rec_end
    return out


def parse(path: Path):
    data = Path(path).read_bytes()
    if not data.startswith(b'Kaydara FBX Binary'):
        raise ValueError('not a binary FBX')
    version = struct.unpack_from('<I', data, 23)[0]
    return version, _read_records(data, 27, len(data))


def _walk(records):
    for record in records:
        yield record
        yield from _walk(record.children)


def check_structure(path: Path) -> dict:
    version, records = parse(path)
    models = [r for r in _walk(records) if r.name == 'Model']
    bones, markers = [], []
    for model in models:
        kind = model.props[2] if len(model.props) > 2 else ''
        if isinstance(kind, (bytes, bytearray)):
            kind = kind.decode('utf-8', 'replace')
        label = model.props[1] if len(model.props) > 1 else b''
        if isinstance(label, (bytes, bytearray)):
            label = label.split(b'\x00\x01')[0].decode('utf-8', 'replace')
        if kind == 'LimbNode':
            (markers if str(label).endswith('_end') else bones).append(str(label))
    curves = [r for r in _walk(records) if r.name == 'AnimationCurve']
    flags = set()
    for curve in curves:
        for child in curve.children:
            if child.name == 'KeyAttrFlags' and child.props:
                flags.add(int(child.props[0][0]) if isinstance(child.props[0], tuple) else int(child.props[0]))
    poses = [r for r in _walk(records) if r.name == 'Pose']
    settings = {}
    for record in _walk(records):
        if record.name == 'Properties70' and record.children:
            for prop in record.children:
                if prop.name == 'P' and len(prop.props) >= 4 and prop.props[0] == b'UnitScaleFactor':
                    settings['UnitScaleFactor'] = prop.props[4] if len(prop.props) > 4 else None
                if prop.name == 'P' and len(prop.props) >= 4 and prop.props[0] == b'TimeMode':
                    settings['TimeMode'] = prop.props[4] if len(prop.props) > 4 else None
    return dict(version=version, bones=len(bones), markers=len(markers),
                bone_names=bones, curves=len(curves), key_flags=sorted(flags),
                bind_pose=bool(poses), settings=settings)


def blender_available() -> str | None:
    for candidate in BLENDER_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return shutil.which('blender')


def check_blender(path: Path) -> dict | None:
    blender = blender_available()
    if not blender:
        return None
    script = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8')
    script.write(
        "import bpy, json, sys\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        "bpy.ops.import_scene.fbx(filepath=sys.argv[-1])\n"
        "arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)\n"
        "out = {'armature': bool(arm)}\n"
        "if arm:\n"
        "    out['bones'] = len(arm.data.bones)\n"
        "    act = arm.animation_data.action if arm.animation_data else None\n"
        "    out['curves'] = len(act.fcurves) if act else 0\n"
        "    out['interp'] = sorted({k.interpolation for fc in act.fcurves for k in fc.keyframe_points[:2]}) if act else []\n"
        "    out['frames'] = [float(act.frame_range[0]), float(act.frame_range[1])] if act else None\n"
        "print('RESULT', json.dumps(out))\n")
    script.close()
    try:
        proc = subprocess.run([blender, '-b', '-noaudio', '--python', script.name, '--', str(path)],
                              capture_output=True, text=True, encoding='utf-8', errors='replace',
                              timeout=600)
        for line in proc.stdout.splitlines():
            if line.startswith('RESULT '):
                return json.loads(line[len('RESULT '):])
        return {'error': 'no result from blender'}
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}
    finally:
        Path(script.name).unlink(missing_ok=True)


def validate(fbx_path: Path, frames: int) -> dict:
    info = check_structure(fbx_path)
    problems = []
    if info['bones'] != EXPECTED_BONES:
        problems.append(f"bones {info['bones']} != {EXPECTED_BONES}")
    if info['markers'] != EXPECTED_MARKERS:
        problems.append(f"markers {info['markers']} != {EXPECTED_MARKERS}")
    if info['curves'] != EXPECTED_CURVES:
        problems.append(f"curves {info['curves']} != {EXPECTED_CURVES}")
    if info['key_flags'] != [4]:
        problems.append(f"interpolation flags {info['key_flags']} != [4] (linear)")
    if not info['bind_pose']:
        problems.append('missing bind pose')
    if info['bone_names'][:3] != ['Hips', 'LeftUpLeg', 'RightUpLeg']:
        problems.append(f"bone order {info['bone_names'][:3]}")
    blender = check_blender(fbx_path)
    if blender and blender.get('bones') != EXPECTED_BONES + EXPECTED_MARKERS:
        problems.append(f"blender bones {blender.get('bones')}")
    if blender and blender.get('interp') and 'LINEAR' not in blender['interp']:
        problems.append(f"blender interpolation {blender.get('interp')}")
    return dict(frames=frames, fps=30, skeletons=f"{EXPECTED_BONES}+{EXPECTED_MARKERS}",
                curves=info['curves'], interpolation='LINEAR', unit='meter',
                fbx_version=info['version'], bind_pose=info['bind_pose'],
                blender=blender, problems=problems)
