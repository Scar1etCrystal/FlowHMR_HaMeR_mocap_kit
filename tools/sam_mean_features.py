"""Production SAM hand features.

Body tokens are extracted for every frame. The hand tokens (left/right) are then
assembled into the 3072-d condition in one of three ways:

  hand_mode='sample_linear'  (DEFAULT) hand samples are taken one per `interval`
                        frames with a fixed seed, then each side is linearly
                        interpolated between neighbouring observations and held
                        constant outside the first/last observation. This keeps
                        the hand timing instead of collapsing it to one value.
                        Validated default: interval=4.
  hand_mode='mean'      hand samples are taken one per `interval` frames and
                        their per-side arithmetic mean is broadcast to every
                        frame. The earlier N60 production path; loses all hand
                        timing and measurably degrades wrist orientation.
  hand_mode='per_frame' hands are extracted for EVERY frame and used per frame.
                        Fully preserves hand timing at the highest SAM cost.

No cached full-hand result is used to fake any path.
"""
import contextlib
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

VERSION = 'sam-hand-v3'

class Quiet(contextlib.AbstractContextManager):
    def write(self, text):
        return len(text)
    def flush(self):
        pass
    def __enter__(self):
        self.redirect = contextlib.redirect_stdout(self)
        self.redirect.__enter__()
        return self
    def __exit__(self, *args):
        return self.redirect.__exit__(*args)

def sample_frames(total, interval, seed):
    if total < 1 or interval < 1:
        raise ValueError('Frame count and SAM interval must be positive')
    rng = np.random.RandomState(seed)
    return np.array([int(rng.randint(s, min(s + interval, total)))
                     for s in range(0, total, interval)], dtype=np.int64)

def linear_fill(observed, mask, total):
    """Linear blend between neighbouring observations; hold before the first and
    after the last one. Frame order and indices are never changed."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError('linear_fill needs at least one observation')
    out = np.empty_like(observed)
    for f in range(total):
        hi = int(np.searchsorted(idx, f))
        if hi == 0:
            out[f] = observed[idx[0]]
        elif hi >= idx.size:
            out[f] = observed[idx[-1]]
        elif int(idx[hi]) == f:
            out[f] = observed[f]
        else:
            lo, up = int(idx[hi - 1]), int(idx[hi])
            w = (f - lo) / (up - lo)
            out[f] = (1.0 - w) * observed[lo] + w * observed[up]
    return out

def extract(video, bbox_file, camera_file, output_dir, interval=4, seed=20261014,
            hand_mode='sample_linear'):
    """Write sam_mean_features.pt (T,3072) plus the components used to build it."""
    from flowhmr.utils.runtime_sam_features import build_sam3d_extractor
    if hand_mode not in ('sample_linear', 'mean', 'per_frame'):
        raise ValueError(
            f"hand_mode must be 'sample_linear', 'mean' or 'per_frame', got {hand_mode!r}")
    output_dir = Path(output_dir)
    with np.load(bbox_file, allow_pickle=False) as z:
        boxes = z['bbox'].astype(np.float32)
    with np.load(camera_file, allow_pickle=False) as z:
        intrinsics = z['K'].astype(np.float32)
    T = len(boxes)
    if intrinsics.shape != (T, 3, 3):
        raise ValueError(f'Camera/frame mismatch: {intrinsics.shape}, {T}')

    dense = hand_mode == 'per_frame'
    linear = hand_mode == 'sample_linear'
    selected = np.arange(T, dtype=np.int64) if dense else sample_frames(T, interval, seed)
    lookup = set(selected.tolist())

    ext = build_sam3d_extractor(torch.device('cuda'))
    counters = {'body': 0, 'hand': 0}
    original = ext.estimator.model.forward_step
    def counted(batch, decoder_type='body'):
        counters[decoder_type] = counters.get(decoder_type, 0) + 1
        return original(batch, decoder_type=decoder_type)
    ext.estimator.model.forward_step = counted

    body = torch.empty((T, 1024), dtype=torch.float32)
    hand_l = torch.zeros((T, 1024), dtype=torch.float32)
    hand_r = torch.zeros((T, 1024), dtype=torch.float32)
    got = np.zeros(T, dtype=bool)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {video}')
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        for i in range(T):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Video decode ended at {i}, expected {T}; no silent truncation')
            use_hand = i in lookup
            ext._body_tokens.clear()
            ext._hand_tokens.clear()
            with Quiet():
                outputs = ext.estimator.process_one_image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    bboxes=boxes[i].reshape(1, 4),
                    cam_int=torch.from_numpy(intrinsics[i:i+1]),
                    inference_type='full' if use_hand else 'body')
            if not outputs or not ext._body_tokens:
                raise RuntimeError(f'SAM body extraction failed at global frame {i}')
            token = ext._body_tokens[0][0, 0].detach().cpu().float()
            if token.shape != (1024,) or not torch.isfinite(token).all():
                raise RuntimeError(f'Invalid body token at frame {i}')
            body[i] = token
            if use_hand:
                if not ext._hand_tokens or ext._hand_tokens[0].shape[0] != 2:
                    raise RuntimeError(f'Expected Fast merged left/right tokens at frame {i}')
                h = ext._hand_tokens[0][:, 0].detach().cpu().float()
                if torch.isfinite(h).all():
                    hand_l[i] = h[0]
                    hand_r[i] = h[1]
                    got[i] = True
                else:
                    if dense:
                        raise RuntimeError(f'Non-finite hand token at frame {i} in per_frame mode')
                    print(f'[sam] invalid sampled hands at frame {i}; excluded from mean', flush=True)
            del outputs
            if i % 30 == 0 or i == T-1:
                elapsed = time.perf_counter() - start
                print(f'[sam {hand_mode} N={interval}] {i+1}/{T} frames | {elapsed:.1f}s | '
                      f'hands={counters["hand"]}', flush=True)
        if cap.read()[0]:
            raise RuntimeError('Video has more decoded frames than detection metadata')
        good = np.flatnonzero(got)
        if good.size == 0:
            raise RuntimeError('No finite hand conditions; cannot build the requested features')
        if dense and good.size != T:
            raise RuntimeError(f'per_frame mode requires all {T} frames, got {good.size}')
        mean_l = hand_l[good].mean(dim=0, keepdim=True)
        mean_r = hand_r[good].mean(dim=0, keepdim=True)

        if dense:
            features = torch.cat([body, hand_l, hand_r], dim=1).contiguous()
        elif linear:
            # Interpolate each side separately between the frames that were
            # actually observed, so hand timing survives between samples.
            fill_l = linear_fill(hand_l.numpy(), got, T)
            fill_r = linear_fill(hand_r.numpy(), got, T)
            if not (np.isfinite(fill_l).all() and np.isfinite(fill_r).all()):
                raise RuntimeError('Non-finite values produced by hand interpolation')
            features = torch.cat([body, torch.from_numpy(fill_l).float(),
                                  torch.from_numpy(fill_r).float()], dim=1).contiguous()
        else:
            features = torch.cat([body, mean_l.expand(T, -1), mean_r.expand(T, -1)],
                                 dim=1).contiguous()

        assert counters == {'body': T, 'hand': int(len(selected))}, counters
        assert features.shape == (T, 3072) and torch.isfinite(features).all()
        torch.save(features, output_dir / 'sam_mean_features.pt')
        components = dict(body=body.numpy(),
                          left_mean=mean_l.numpy(), right_mean=mean_r.numpy(),
                          selected_global_frames=selected.astype(np.int64),
                          used_global_frames=good.astype(np.int64))
        if not dense:
            # Keep the raw observations behind the mean/interpolation so the
            # assembled condition can be recomputed and checked afterwards.
            components['left_observed'] = hand_l[good].numpy()
            components['right_observed'] = hand_r[good].numpy()
        np.savez(output_dir / 'sam_mean_components.npz', **components)
        # Dense hands are kept so a later 'mean' variant can be built without
        # repeating the SAM pass; only written when every frame was observed.
        dense_path = output_dir / 'sam_hand_dense.npz'
        if dense:
            np.savez(dense_path, left=hand_l.numpy(), right=hand_r.numpy(),
                     observed=got, left_mean=mean_l.numpy(), right_mean=mean_r.numpy())
        elif dense_path.exists():
            dense_path.unlink()

        rule = ('every frame, used per frame' if dense else
                'one seeded random sample per interval, then linear interpolation '
                'between neighbours (ends held)' if linear else
                'one seeded random sample per interval, then per-side mean')
        info = {'version': VERSION, 'hand_mode': hand_mode, 'interval': interval,
                'sampling_seed': seed, 'sampling_rule': rule,
                'frames': T, 'dense_hands': bool(dense),
                'hand_feature_temporal_std_mean': float(features[:, 1024:].std(dim=0).mean()),
                'selected_frames': selected.tolist(), 'used_frames': good.tolist(),
                'body_forwards': counters['body'], 'merged_hand_forwards': counters['hand'],
                'seconds_with_model_load_excluded': time.perf_counter() - start,
                'peak_alloc_GiB': torch.cuda.max_memory_allocated() / 2**30}
        (output_dir / 'sam_run_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        print('SAM_DONE', json.dumps(info), flush=True)
        return info
    finally:
        cap.release()
        ext.estimator.model.forward_step = original
        for hook in ext._hooks:
            hook.remove()
        ext._body_tokens.clear(); ext._hand_tokens.clear()
        del ext
        gc.collect()
        torch.cuda.empty_cache()
