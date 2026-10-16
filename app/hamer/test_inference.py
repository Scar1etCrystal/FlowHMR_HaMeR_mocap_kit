"""Headless HAMER inference - extracts hand pose data (MANO params, vertices, joints) without rendering."""
import os
# PATCHED: Only set PYOPENGL_PLATFORM on Linux
import sys as _sys
if _sys.platform != 'win32':
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
import subprocess as _subproc

# Pre-check GPU before importing torch: hide GPU if incompatible (CC<7.0 or VRAM<4GB)
# A subprocess crash (segfault) also means GPU is incompatible
try:
    _gpu = _subproc.run([sys.executable, '-c',
        'import torch; p=torch.cuda.get_device_properties(0); '
        'print(f"{p.total_memory//(1024*1024)} {p.major} {p.minor}")'],
        capture_output=True, text=True, timeout=20)
    if _gpu.returncode == 0:
        _vram, _maj, _min = _gpu.stdout.strip().split()
        _vram, _cc = int(_vram), int(_maj) + int(_min)*0.1
        if _cc < 7.0 or _vram < 4000:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print(f"GPU incompatible (VRAM={_vram}MB CC={_cc}), forcing CPU mode")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("GPU check crashed, forcing CPU mode")
except Exception:
    pass

import torch
import argparse
import cv2
import numpy as np
import pickle
from pathlib import Path

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full

from vitpose_model import ViTPoseModel

def main():
    parser = argparse.ArgumentParser(description='HaMeR headless inference')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--img_folder', type=str, default='example_data')
    parser.add_argument('--out_folder', type=str, default='test_output')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'])
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'])
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    # Load model
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    # Auto-select device: use CUDA only if GPU has enough VRAM (>=4GB)
    # detectron2 CUDA ops crash on older GPUs (e.g. GTX 1050 with 2GB)
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024*1024)
            # Compute capability check: detectron2 ops need >= 7.0 (Volta+)
            major, minor = torch.cuda.get_device_capability(0)
            cc = major + minor * 0.1
            if cc >= 7.0 and vram_mb >= 4000:
                device = torch.device('cuda')
                print(f"Using CUDA: {gpu_name} ({vram_mb}MB, CC {cc})")
            else:
                device = torch.device('cpu')
                print(f"Falling back to CPU: {gpu_name} ({vram_mb}MB, CC {cc}) - need >=4GB VRAM and CC >=7.0")
        except Exception as e:
            device = torch.device('cpu')
            print(f"Falling back to CPU (GPU check failed: {e})")
    else:
        device = torch.device('cpu')
        print("Using CPU (no CUDA available)")
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        _vitdet_local = Path(hamer.__file__).parent / '..' / 'models' / 'model_final_f05665.pkl'
        if _vitdet_local.exists():
            detectron2_cfg.train.init_checkpoint = str(_vitdet_local.resolve())
        else:
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        import hamer as _hamer_pkg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        _regnety_local = Path(_hamer_pkg.__file__).parent / '..' / 'models' / 'model_final_ef3a80.pkl'
        if _regnety_local.exists():
            detectron2_cfg.train.init_checkpoint = str(_regnety_local.resolve())
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Keypoint detector
    cpm = ViTPoseModel(device)

    # Get images
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    print(f'Found {len(img_paths)} images')

    all_results = []

    for img_path in img_paths:
        print(f'Processing: {img_path}')
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print(f'  Cannot read image, skipping')
            continue

        # Detect humans
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if len(pred_bboxes) == 0:
            print('  No person detected, skipping')
            continue

        # Detect keypoints
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # Extract hand bboxes from keypoints
        bboxes = []
        is_right = []
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)

            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            print('  No hands detected, skipping')
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        print(f'  Detected {len(boxes)} hand(s): {["left" if r==0 else "right" for r in right]}')

        # Run HAMER
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                person_id = int(batch['personid'][n])
                is_r = int(batch['right'][n].cpu().numpy())
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                verts[:, 0] = (2 * is_r - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]

                # MANO parameters (hand pose + shape)
                result = {
                    'image': str(img_path),
                    'person_id': person_id,
                    'is_right': is_r,
                    'pred_vertices': verts,
                    'pred_cam_t_full': cam_t,
                    'pred_cam': out['pred_cam'][n].detach().cpu().numpy(),
                    'pred_keypoints_3d': out.get('pred_keypoints_3d', [None])[n].detach().cpu().numpy() if 'pred_keypoints_3d' in out else None,
                    'bbox': boxes[n] if n < len(boxes) else None,
                    'mano_faces_right': model.mano.faces,
                }

                # Try to get MANO parameters
                if 'pred_mano_params' in out:
                    mp = out['pred_mano_params']
                    if isinstance(mp, dict):
                        result['mano_params'] = {k: v[n].detach().cpu().numpy() for k, v in mp.items()}
                    else:
                        result['mano_params'] = mp[n].detach().cpu().numpy()

                all_results.append(result)
                print(f'  Hand {n}: {"right" if is_r else "left"}, vertices={verts.shape}, cam_t={cam_t}')

    # Save
    out_path = os.path.join(args.out_folder, 'hamer_predictions.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nDone! Saved {len(all_results)} hand predictions to {out_path}')

if __name__ == '__main__':
    main()
