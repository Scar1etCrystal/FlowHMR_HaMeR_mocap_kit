"""Production SAM hand-mean features, using the accepted N60 experiment rule.
Body every frame; one seeded random hand observation per interval; separate L/R
arithmetic means broadcast to all frames. No full-hand cache is used to fake speed.
"""
import contextlib
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

VERSION = 'sam-mean-v1'

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

def extract(video, bbox_file, camera_file, output_dir, interval=60, seed=20261014):
    from flowhmr.utils.runtime_sam_features import build_sam3d_extractor
    output_dir = Path(output_dir)
    with np.load(bbox_file, allow_pickle=False) as z:
        boxes = z['bbox'].astype(np.float32)
    with np.load(camera_file, allow_pickle=False) as z:
        intrinsics = z['K'].astype(np.float32)
    T = len(boxes)
    if intrinsics.shape != (T, 3, 3):
        raise ValueError(f'Camera/frame mismatch: {intrinsics.shape}, {T}')
    selected = sample_frames(T, interval, seed)
    lookup = set(selected.tolist())
    ext = build_sam3d_extractor(torch.device('cuda'))
    counters = {'body': 0, 'hand': 0}
    original = ext.estimator.model.forward_step
    def counted(batch, decoder_type='body'):
        counters[decoder_type] = counters.get(decoder_type, 0) + 1
        return original(batch, decoder_type=decoder_type)
    ext.estimator.model.forward_step = counted
    body = torch.empty((T, 1024), dtype=torch.float32)
    left, right, used = [], [], []
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
                    left.append(h[0].clone());right.append(h[1].clone());used.append(i)
                else:
                    print(f'[sam] invalid sampled hands at frame {i}; excluded from mean', flush=True)
            del outputs
            if i % 30 == 0 or i == T-1:
                elapsed = time.perf_counter() - start
                print(f'[sam N={interval}] {i+1}/{T} frames | {elapsed:.1f}s | '
                      f'hands={counters["hand"]}', flush=True)
        if cap.read()[0]:
            raise RuntimeError('Video has more decoded frames than detection metadata')
        if not left:
            raise RuntimeError('No finite sampled hand conditions; cannot construct an accepted mean')
        lm, rm = torch.stack(left).mean(0), torch.stack(right).mean(0)
        features = torch.cat([body, lm.expand(T, -1), rm.expand(T, -1)], dim=1).contiguous()
        assert counters == {'body': T, 'hand': len(selected)}, counters
        assert features.shape == (T, 3072) and torch.isfinite(features).all()
        torch.save(features, output_dir/'sam_mean_features.pt')
        np.savez(output_dir/'sam_mean_components.npz', body=body.numpy(),
                 left_mean=lm.numpy(), right_mean=rm.numpy(),
                 selected_global_frames=selected, used_global_frames=np.array(used, dtype=np.int64))
        info = {'version': VERSION, 'mode': 'sampled_mean', 'interval': interval,
                'sampling_seed': seed, 'sampling_rule': 'one seeded random sample per interval',
                'frames': T, 'selected_frames': selected.tolist(), 'used_frames': used,
                'body_forwards': counters['body'], 'merged_hand_forwards': counters['hand'],
                'seconds_with_model_load_excluded': time.perf_counter()-start,
                'peak_alloc_GiB': torch.cuda.max_memory_allocated()/2**30}
        (output_dir/'sam_run_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
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
