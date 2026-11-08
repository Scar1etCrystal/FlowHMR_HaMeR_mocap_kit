"""Multi-shot handling for the FlowHMR runner.

A camera cut (close-up, angle change) gives ByteTrack a new id, so the official
pipeline keeps only one track and silently drops the rest of the video. These
helpers plan shot segments from the tracker output, build per-segment inputs and
stitch the per-segment SMPL arrays back into one continuous take.

Nothing here runs the network; it only consumes the existing npz/json formats.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
FFMPEG = PACK / 'envs' / 'Lib' / 'site-packages' / 'imageio_ffmpeg' / 'binaries' / 'ffmpeg-win-x86_64-v7.1.exe'


def _ffmpeg() -> str:
    return str(FFMPEG) if FFMPEG.is_file() else 'ffmpeg'


# --------------------------------------------------------------------------- #
# tracking pre-pass
# --------------------------------------------------------------------------- #
def save_tracks(path: Path, tracks: list[dict], *, total: int, width: int, height: int) -> None:
    payload = {'total_frames': total, 'width': width, 'height': height, 'count': len(tracks)}
    for index, track in enumerate(tracks):
        payload[f'track{index}_id'] = np.asarray(track['id'], np.int64)
        payload[f'track{index}_frames'] = np.asarray(track['frames'], np.int64)
        payload[f'track{index}_bboxes'] = np.asarray(track['bboxes'], np.float32)
    np.savez(path, **payload)


def load_tracks(path: Path) -> dict:
    with np.load(path) as data:
        total = int(data['total_frames'])
        width = int(data['width'])
        height = int(data['height'])
        count = int(data['count'])
        tracks = [{
            'id': int(data[f'track{i}_id']),
            'frames': data[f'track{i}_frames'].astype(np.int64),
            'bboxes': data[f'track{i}_bboxes'].astype(np.float32),
        } for i in range(count)]
    return {'tracks': tracks, 'total_frames': total, 'width': width, 'height': height}


# --------------------------------------------------------------------------- #
# shot planning
# --------------------------------------------------------------------------- #
def _area_fractions(track: dict, width: int, height: int) -> np.ndarray:
    boxes = track['bboxes']
    w = np.clip(boxes[:, 2] - boxes[:, 0], 1.0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 1.0, None)
    return (w * h) / float(width * height)


def _median_fraction(track: dict, width: int, height: int) -> float:
    frames = track['frames']
    if frames.size == 0:
        return 0.0
    return float(np.median(_area_fractions(track, width, height)))


def _boundary_fraction(track: dict, width: int, height: int, *, first: bool) -> float:
    frames = track['frames']
    if frames.size == 0:
        return 0.0
    order = np.argsort(frames)
    index = order[0] if first else order[-1]
    box = track['bboxes'][index]
    area = max(float(box[2] - box[0]), 1.0) * max(float(box[3] - box[1]), 1.0)
    return area / float(width * height)


def plan_segments(
    tracks: list[dict],
    total: int,
    width: int,
    height: int,
    *,
    min_frames: int = 24,
    min_share: float = 0.02,
    gap_frames: int = 24,
    max_segments: int = 12,
    switch_ratio: float = 0.30,
) -> list[dict]:
    """Turn tracker tracks into contiguous, full-coverage shot segments.

    A new ByteTrack id on a *different* frame range is a camera cut, so each
    such range becomes its own inference run (and therefore keeps its frames).
    Two ids are treated as one shot only when their ranges overlap a lot or they
    are frame-contiguous with nearly the same box size at the seam (an id switch
    inside a continuous shot).

    ``switch_ratio`` is the relative box-area change allowed at a contiguous
    seam before it is called a cut.
    """
    if not tracks:
        return [{'start': 0, 'end': total, 'frames': total, 'tracks': [], 'bbox_fraction': 0.0}]

    infos = []
    for index, track in enumerate(tracks):
        frames = track['frames']
        if frames.size == 0:
            continue
        infos.append({
            'index': index,
            'id': int(track['id']),
            'start': int(frames.min()),
            'end': int(frames.max()) + 1,
            'length': int(frames.size),
            'median': _median_fraction(track, width, height),
            'first': _boundary_fraction(track, width, height, first=True),
            'last': _boundary_fraction(track, width, height, first=False),
        })
    if not infos:
        return [{'start': 0, 'end': total, 'frames': total, 'tracks': [], 'bbox_fraction': 0.0}]

    keep = [item for item in infos if item['length'] >= min_frames
            and item['length'] / max(total, 1) >= min_share]
    if not keep:
        keep = [max(infos, key=lambda item: item['length'])]

    keep.sort(key=lambda item: (item['start'], -item['length']))
    shots: list[dict] = []
    for item in keep:
        attached = False
        for shot in shots:
            overlap = min(shot['end'], item['end']) - max(shot['start'], item['start'])
            shorter = min(shot['end'] - shot['start'], item['length'])
            if overlap > 0.5 * max(shorter, 1):
                shot['end'] = max(shot['end'], item['end'])
                shot['start'] = min(shot['start'], item['start'])
                shot['items'].append(item)
                attached = True
                break
        if attached:
            continue
        if shots:
            previous = shots[-1]
            seam_gap = item['start'] - previous['end']
            if 0 <= seam_gap <= 2 and previous['items']:
                a = previous['items'][-1]['last']
                b = item['first']
                scale = max(a, b, 1e-6)
                if abs(a - b) / scale <= switch_ratio:
                    previous['end'] = max(previous['end'], item['end'])
                    previous['items'].append(item)
                    continue
        shots.append({'start': item['start'], 'end': item['end'], 'items': [item]})

    # Absorb fragments that are not worth an inference run.
    changed = True
    while changed and len(shots) > 1:
        changed = False
        for i, shot in enumerate(shots):
            length = shot['end'] - shot['start']
            if length >= min_frames and length / max(total, 1) >= min_share:
                continue
            neighbour = i - 1 if i > 0 else i + 1
            target = shots[neighbour]
            target['start'] = min(target['start'], shot['start'])
            target['end'] = max(target['end'], shot['end'])
            target['items'] = (target['items'] + shot['items']) if neighbour < i else (shot['items'] + target['items'])
            shots.pop(i)
            changed = True
            break

    while len(shots) > max_segments:
        index = min(range(len(shots)), key=lambda i: shots[i]['end'] - shots[i]['start'])
        neighbour = index - 1 if index > 0 else index + 1
        target = shots[neighbour]
        target['start'] = min(target['start'], shots[index]['start'])
        target['end'] = max(target['end'], shots[index]['end'])
        target['items'] += shots[index]['items']
        shots.pop(index)

    shots.sort(key=lambda shot: shot['start'])
    segments = []
    cursor = 0
    for i, shot in enumerate(shots):
        start = 0 if i == 0 else cursor
        end = total if i == len(shots) - 1 else min(shot['end'], total)
        if end <= start:
            continue
        ids = [item['id'] for item in shot['items']]
        median = float(np.mean([item['median'] for item in shot['items']])) if shot['items'] else 0.0
        segments.append({
            'start': int(start),
            'end': int(end),
            'frames': int(end - start),
            'track_ids': [int(value) for value in ids],
            'bbox_fraction': round(median, 4),
        })
        cursor = end
    if not segments:
        segments = [{'start': 0, 'end': total, 'frames': total, 'track_ids': [], 'bbox_fraction': 0.0}]
    return segments


def describe_plan(tracks: list[dict], total: int, width: int, height: int, segments: list[dict]) -> dict:
    rows = []
    for index, track in enumerate(tracks):
        frames = track['frames']
        if frames.size == 0:
            continue
        area = _area_fractions(track, width, height)
        rows.append({
            'track_id': int(track['id']),
            'start': int(frames.min()),
            'end': int(frames.max()) + 1,
            'length': int(frames.size),
            'avg_bbox_fraction': round(float(np.mean(area)), 4),
            'max_bbox_fraction': round(float(np.max(area)), 4),
        })
    rows.sort(key=lambda row: -row['length'])
    return {'total_frames': total, 'width': width, 'height': height,
            'tracks': rows, 'segments': segments}


# --------------------------------------------------------------------------- #
# per-segment inputs
# --------------------------------------------------------------------------- #
def trim_video(src: Path, dst: Path, first: int, last: int) -> None:
    """Copy frame range ``first..last`` (inclusive) as a 30 fps clip."""
    print(f'  trimming frames {first}..{last} -> {dst.name}', flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        _ffmpeg(), '-y', '-i', str(src),
        '-vf', f"select='between(n\\,{first}\\,{last})',setpts=N/30/TB",
        '-r', '30', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17',
        '-pix_fmt', 'yuv420p', '-an', str(dst),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def concat_videos(parts: list[Path], dst: Path) -> None:
    if len(parts) == 1:
        shutil.copyfile(parts[0], dst)
        return
    listing = dst.with_suffix('.concat.txt')
    listing.write_text(''.join(f"file '{part.as_posix()}'\n" for part in parts), encoding='utf-8')
    subprocess.run([
        _ffmpeg(), '-y', '-f', 'concat', '-safe', '0', '-i', str(listing),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', '-r', '30',
        '-pix_fmt', 'yuv420p', '-an', str(dst),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    listing.unlink(missing_ok=True)


def segment_bbox(tracks: list[dict], start: int, end: int, width: int, height: int) -> np.ndarray:
    """Per-frame xyxy boxes for frames ``[start, end)`` from the planned tracks.

    Uses the track that covers most of the window, with linear interpolation for
    frames where it has no detection.
    """
    length = end - start
    best, best_score = None, -1
    for track in tracks:
        frames = track['frames']
        if frames.size == 0:
            continue
        overlap = int(np.count_nonzero((frames >= start) & (frames < end)))
        score = overlap
        if score > best_score:
            best, best_score = track, score
    if best is None or best_score <= 0:
        full = np.array([0.0, 0.0, float(width), float(height)], np.float32)
        return np.tile(full[None], (length, 1))

    order = np.argsort(best['frames'])
    frames = best['frames'][order].astype(np.float32)
    boxes = best['bboxes'][order]
    wanted = np.arange(start, end, dtype=np.float32)
    out = np.empty((length, 4), np.float32)
    for axis in range(4):
        out[:, axis] = np.interp(wanted, frames, boxes[:, axis].astype(np.float32),
                                 left=float(boxes[0, axis]), right=float(boxes[-1, axis]))
    out[:, 0] = np.clip(out[:, 0], 0, width - 1)
    out[:, 1] = np.clip(out[:, 1], 0, height - 1)
    out[:, 2] = np.clip(out[:, 2], 1, width)
    out[:, 3] = np.clip(out[:, 3], 1, height)
    return out


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #
def _transition_offset(prev: dict, cur: dict) -> np.ndarray:
    """Translation offset that makes the new shot start where the old one ended."""
    return np.asarray(prev['trans'][-1], np.float64) - np.asarray(cur['trans'][0], np.float64)


def _yaw_of(rotvec: np.ndarray) -> float:
    """Yaw of a root orientation around the world up axis (radians)."""
    from scipy.spatial.transform import Rotation

    matrix = Rotation.from_rotvec(np.asarray(rotvec, np.float64)).as_matrix()
    return float(np.arctan2(matrix[0, 2], matrix[2, 2]))


def _yaw_rotate(rotvecs: np.ndarray, delta: float) -> np.ndarray:
    """Rotate every root orientation by ``delta`` around the world up axis."""
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec(np.asarray(rotvecs, np.float64))
    turn = Rotation.from_euler('Y', delta)
    return (turn * rotation).as_rotvec().astype(np.float32)


def merge_segment_arrays(parts: list[dict]) -> dict:
    """Concatenate per-shot SMPL arrays with root-position continuity.

    ``parts`` is a list of npz payloads (poses/betas/trans/Rh/foot_contacts) in
    temporal order. Each new shot is shifted so its first root position matches
    the previous shot's last one; betas come from the first shot.
    """
    if not parts:
        raise ValueError('no segment arrays to merge')
    poses = [np.asarray(part['poses'], np.float32) for part in parts]
    trans = [np.asarray(part['trans'], np.float64) for part in parts]
    rh = [np.asarray(part['Rh'], np.float32) for part in parts]
    contacts = [np.asarray(part['foot_contacts'], np.float32) if 'foot_contacts' in part else None
                for part in parts]

    aligned = [trans[0]]
    headings = [rh[0]]
    for i in range(1, len(parts)):
        previous = {'trans': aligned[i - 1]}
        offset = _transition_offset(previous, {'trans': trans[i]})
        print(f'  segment {i}: root offset {np.round(offset, 4).tolist()} m', flush=True)
        aligned.append(trans[i] + offset)
        # A camera cut can flip the estimated world facing; keep the seam
        # orientation continuous by yawing the whole shot (constant turn, so no
        # extra step appears inside the shot).
        previous_yaw = _yaw_of(headings[-1][-1])
        current_yaw = _yaw_of(rh[i][0])
        delta = float(np.arctan2(np.sin(previous_yaw - current_yaw),
                                 np.cos(previous_yaw - current_yaw)))
        if abs(delta) > np.radians(10.0):
            print(f'  segment {i}: heading correction {np.degrees(delta):.1f} deg', flush=True)
            rh[i] = _yaw_rotate(rh[i], delta)
            poses[i] = poses[i].copy()
            poses[i][:, 0, :] = rh[i]
        headings.append(rh[i])

    merged = {
        'poses': np.concatenate(poses, axis=0).astype(np.float32),
        'trans': np.concatenate(aligned, axis=0).astype(np.float32),
        'Rh': np.concatenate([np.asarray(part, np.float32) for part in rh]).astype(np.float32),
        'betas': np.asarray(parts[0]['betas'], np.float32).reshape(1, -1),
        'mocap_framerate': np.asarray(parts[0].get('mocap_framerate', 30)),
        'num_frames': int(sum(len(part['poses']) for part in parts)),
        'gender': np.asarray(parts[0].get('gender', 'neutral')),
    }
    if all(part is not None for part in contacts):
        merged['foot_contacts'] = np.concatenate(contacts, axis=0).astype(np.float32)
    return merged


def merge_reports(reports: list[dict], contacts: list[dict], offsets: list[int], total: int) -> tuple[dict, dict]:
    """Shift each shot's contact plan into the merged timeline."""
    if not reports:
        return {}, {}
    report = json.loads(json.dumps(reports[0], default=float))
    report['frames'] = int(total)
    segments = []
    for shot, offset in zip(reports, offsets):
        for segment in shot.get('segments') or []:
            row = dict(segment)
            row['start'] = int(row.get('start', 0)) + offset
            row['end'] = int(row.get('end', 0)) + offset
            if 'handoff' in row:
                row['handoff'] = [int(row['handoff'][0]) + offset, int(row['handoff'][1]) + offset]
            segments.append(row)
    report['segments'] = segments
    report['elapsed_seconds'] = float(sum(shot.get('elapsed_seconds') or 0.0 for shot in reports))
    warning = ' '.join(str(shot.get('warning') or '') for shot in reports).strip()
    report['warning'] = warning

    merged_contacts: dict = {}
    if contacts and all(item for item in contacts):
        tables, automatic, handoffs = [], [], []
        anchors, weights = [], []
        for shot, offset in zip(contacts, offsets):
            table = np.array(shot['segments'], copy=True)
            if table.size:
                table[:, 1] += offset
                table[:, 2] += offset
            tables.append(table.reshape(-1, 6))
            auto = np.array(shot['automatic'], copy=True)
            if auto.size:
                auto[:, 1] += offset
                auto[:, 2] += offset
            automatic.append(auto.reshape(-1, 3))
            hand = np.array(shot['handoffs'], copy=True)
            if hand.size:
                hand[:, 1] += offset
                hand[:, 2] += offset
            handoffs.append(hand.reshape(-1, 3))
            anchors.append(np.asarray(shot['anchor_trajectory']))
            weights.append(np.asarray(shot['support_weights']))
        merged_contacts = {
            'segments': np.concatenate(tables, axis=0) if tables else np.zeros((0, 6)),
            'automatic': np.concatenate(automatic, axis=0) if automatic else np.zeros((0, 3), np.int64),
            'anchor_frames': np.arange(total, dtype=np.int64),
            'anchor_trajectory': np.concatenate(anchors, axis=0) if anchors else np.zeros((total, 2, 3), np.float32),
            'support_weights': np.concatenate(weights, axis=0) if weights else np.zeros((total, 2), np.float32),
            'handoffs': np.concatenate(handoffs, axis=0) if handoffs else np.zeros((0, 3), np.int64),
            'planner_version': int(contacts[0].get('planner_version', 1)),
            'handoff_frames': int(sum(int(item.get('handoff_frames') or 0) for item in contacts)),
        }
    return report, merged_contacts
