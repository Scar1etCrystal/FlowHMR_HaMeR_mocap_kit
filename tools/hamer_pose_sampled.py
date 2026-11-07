"""Locked HaMeR in two memory passes (Windows/WDDM friendly).

Pass 1 keeps only ViTPose on the GPU and records the hand boxes.
Pass 2 keeps only HaMeR on the GPU and crops those boxes.

The single-pass original holds both models at once (~5.2 GiB peak). On a 12 GiB
card whose per-process budget is squeezed by desktop apps that fails while
loading HaMeR, so the two models are never resident together here.

Interpolation and smoothing match the original: ViTPose and HaMeR run on
f % N == 0 only, raw rotations are SLERP-interpolated, then the existing
full-rate export/smoothing is applied. N=2 default; N=1 is the full-rate baseline.
"""
from pathlib import Path
import argparse, sys, os, time, json, pickle, gc
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

PACK = Path(os.environ.get('FLOWHMR_PACK_ROOT', r'F:\flowhmr_pack'))
ROOT = PACK / 'app/hamer'
if not ROOT.is_dir():
    raise SystemExit(f'FLOWHMR_PACK_ROOT does not point at the kit: {PACK}')
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
os.environ.setdefault('HAMER_THREADS', '8')
import hamer_locked as H
import cv2, torch


def interpolate_tasks(tasks, T, max_gap):
    result = []; provenance = np.zeros((T, 2), np.uint8)
    for side in (0, 1):
        obs = sorted((x for x in tasks if x['is_right'] == side), key=lambda x: x['frame_idx'])
        if not obs:
            continue
        idx = np.array([x['frame_idx'] for x in obs]); by = {x['frame_idx']: x for x in obs}
        for f in range(T):
            if f in by:
                result.append({'frame_idx': f, 'is_right': side,
                               'results': {k: np.asarray(v).copy() for k, v in by[f]['results'].items()}})
                provenance[f, side] = 1
                continue
            hi = int(np.searchsorted(idx, f)); lo = hi - 1
            if lo < 0 or hi >= len(idx):
                nearest = obs[0] if lo < 0 else obs[-1]
                if abs(f - nearest['frame_idx']) <= max_gap:
                    result.append({'frame_idx': f, 'is_right': side,
                                   'results': {k: np.asarray(v).copy() for k, v in nearest['results'].items()}})
                    provenance[f, side] = 3
                continue
            a, b = obs[lo], obs[hi]; fa, fb = a['frame_idx'], b['frame_idx']
            if fb - fa > max_gap:
                continue
            alpha = (f - fa) / (fb - fa); r = {}
            for key in ('pose', 'orient'):
                A = np.asarray(a['results'][key]); B = np.asarray(b['results'][key]); shape = A.shape
                out = []
                for ma, mb in zip(A.reshape(-1, 3, 3), B.reshape(-1, 3, 3)):
                    out.append(Slerp([fa, fb], Rotation.from_matrix(np.stack([ma, mb])))([f]).as_matrix()[0])
                r[key] = np.array(out, dtype=np.float32).reshape(shape)
            for key in ('betas', 'cam_t'):
                r[key] = ((1 - alpha) * np.asarray(a['results'][key]) + alpha * np.asarray(b['results'][key])).astype(np.float32)
            result.append({'frame_idx': f, 'is_right': side, 'results': r}); provenance[f, side] = 2
    return result, provenance


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def load_hamer_cpu_first():
    """Load HaMeR weights on the CPU, then move them to the GPU once.

    ``H.load_hamer`` runs Lightning's ``load_from_checkpoint`` without a
    map_location, so the checkpoint tensors are restored straight onto the GPU
    and then copied into the model: a ~4.9 GiB transient for a 2.4 GiB model.
    That spike is what fails on a 12 GiB card whose per-process budget is
    squeezed by desktop apps, so load on the CPU and move once instead.
    """
    from hamer.configs import get_config
    from hamer.models import DEFAULT_CHECKPOINT, HAMER

    ckpt = Path(DEFAULT_CHECKPOINT)
    cfg = get_config(str(ckpt.parent.parent / 'model_config.yaml'), update_cachedir=True)
    if (cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in cfg.MODEL):
        cfg.defrost(); cfg.MODEL.BBOX_SHAPE = [192, 256]; cfg.freeze()
    if 'PRETRAINED_WEIGHTS' in cfg.MODEL.BACKBONE:
        cfg.defrost(); cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS'); cfg.freeze()
    model = HAMER.load_from_checkpoint(str(ckpt), strict=False, cfg=cfg, map_location='cpu')
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True); ap.add_argument('--main_track', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--detect_interval', type=int, default=2, help='Run BOTH models only on f %% N == 0 (default 2)')
    ap.add_argument('--batch_size', type=int, default=16); ap.add_argument('--window', type=int, default=11)
    ap.add_argument('--max_frames', type=int, default=0); ap.add_argument('--max_interp_gap', type=int, default=15, help='Do not bridge a missing-hand interval longer than this many frames')
    ap.add_argument('--no_smooth', action='store_true', help='Diagnostic: export interpolated raw poses without Savgol')
    ap.add_argument('--single_pass', action='store_true', help='Old behaviour: keep ViTPose and HaMeR resident together')
    a = ap.parse_args()
    if a.detect_interval < 1:
        ap.error('detect_interval must be >=1')
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
    torch.set_num_threads(H.THREADS); cv2.setNumThreads(H.THREADS)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(a.video)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); T = min(T, a.max_frames) if a.max_frames else T
    W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    mt = np.load(a.main_track); boxes = mt['bbox_xyxy']; assert len(boxes) >= T, (len(boxes), T)

    obs = []; sampled = []; nvit = 0; skipped = [0]
    torch.cuda.reset_peak_memory_stats()
    infer_start = time.perf_counter()

    # ---------------- pass 1: hand boxes from ViTPose ----------------
    vp = H.ViTPoseModel(dev)
    cap = cv2.VideoCapture(a.video)
    for f in range(T):
        ok, frame = cap.read()
        if not ok:
            T = f; break
        if f % a.detect_interval:
            continue
        sampled.append(f)
        bb = boxes[f]
        if not np.isfinite(bb).all() or min(bb[2] - bb[0], bb[3] - bb[1]) <= 4:
            continue
        try:
            res = vp.predict_pose(frame[:, :, ::-1], [np.r_[bb, 1.0][None].astype(np.float32)]); nvit += 1
        except Exception as exc:
            skipped[0] += 1
            print(f'  [hamer] vitpose skip frame {f}: {type(exc).__name__}: {str(exc)[:90]}', flush=True)
            continue
        if not res:
            continue
        for side, sl in H.HAND_SLICES:
            box = H.hand_box(res[0]['keypoints'][sl])
            if box is None:
                continue
            obs.append({'frame_idx': f, 'is_right': side, 'bbox': box})
    cap.release()
    peak_vp = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f'  [hamer] pass 1: {nvit} ViTPose calls, {len(obs)} hand boxes, peak {peak_vp:.2f} GiB', flush=True)
    if not a.single_pass:
        del vp; free_cuda()

    # ---------------- pass 2: HaMeR on those boxes ----------------
    try:
        model, cfg = load_hamer_cpu_first()
    except Exception as exc:
        print(f'  [hamer] CPU-first load unavailable ({type(exc).__name__}); using the original loader', flush=True)
        model, cfg = H.load_hamer(H.DEFAULT_CHECKPOINT)
    model = model.to(dev).eval()
    if a.single_pass:
        vp = H.ViTPoseModel(dev)  # keep the historical memory profile when asked
    focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(W, Hh)
    hb = H.HamerBatcher(model, cfg, dev, focal, a.batch_size)

    def add_safe(key, frame, box, side):
        # A degenerate crop must not kill the whole clip.
        try:
            hb.add(key, frame, box, side)
            return True
        except Exception as exc:
            skipped[0] += 1
            if skipped[0] <= 10:
                print(f'  [hamer] skip frame {key[0]} side {side}: {type(exc).__name__}: {str(exc)[:90]}', flush=True)
            return False

    by_frame = {}
    for x in obs:
        by_frame.setdefault(x['frame_idx'], []).append(x)
    cap = cv2.VideoCapture(a.video)
    for f in range(T):
        ok, frame = cap.read()
        if not ok:
            break
        for x in by_frame.get(f, ()):
            add_safe((f, x['is_right']), frame, x['bbox'], x['is_right'])
    hb.flush(); cap.release(); torch.cuda.synchronize()
    infer_s = time.perf_counter() - infer_start
    peak_hamer = torch.cuda.max_memory_allocated() / 2 ** 30
    if skipped[0]:
        print(f'  [hamer] skipped {skipped[0]} items', flush=True)

    for x in obs:
        x['results'] = {k: np.asarray(hb.out[(x['frame_idx'], x['is_right'])][k]).copy() for k in ('pose', 'orient', 'betas', 'cam_t')}
    with (out / 'sampled_raw.pkl').open('wb') as fp:
        pickle.dump({'observations': obs, 'T': T, 'sampled_frames': sampled}, fp, protocol=4)
    tasks, prov = interpolate_tasks(obs, T, a.max_interp_gap)
    window = T + 1 if a.no_smooth else a.window
    ex = H.export_like_original(tasks, T, window)
    stem = Path(a.video).stem
    with (out / f'{stem}_hamer_data.pkl').open('wb') as fp:
        pickle.dump(ex, fp, protocol=4)
    np.savez(out / f'{stem}_sampling.npz', source=prov, sampled_frames=sampled)
    info = {'frames': T, 'fps': fps, 'detect_interval': a.detect_interval, 'vitpose_calls': nvit,
            'hamer_hand_calls': len(obs), 'inference_seconds': infer_s,
            'seconds_with_load_export': time.perf_counter() - t0, 'window': 0 if a.no_smooth else a.window,
            'tta': 1, 'peak_alloc_GiB': max(peak_vp, peak_hamer), 'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2 ** 30,
            'peak_vitpose_GiB': peak_vp, 'peak_hamer_GiB': peak_hamer,
            'passes': 1 if a.single_pass else 2, 'skipped_crops': skipped[0],
            'provenance_legend': {'0': 'invalid', '1': 'model', '2': 'interpolated', '3': 'endpoint held'},
            'valid_LR': ex['global_orient_valid'].sum(axis=0).tolist()}
    (out / 'run_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('DONE', json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
