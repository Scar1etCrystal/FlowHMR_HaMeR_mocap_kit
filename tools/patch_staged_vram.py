"""Extend unloading to BEFORE VGGT, not just after it.
Keeps the same weights, precision, attention and per-video outputs.
Models on CPU can be restored for subsequent videos; cached features do not
trigger unnecessary SAM reloads. Every transition reports allocation evidence.
"""
from pathlib import Path
import shutil,py_compile
P=Path(__file__).resolve().parents[1];f=P/'app/flowhmr/flowhmr/pipeline/v2m_inference.py'
s=f.read_text(encoding='utf-8')
old_sam='            sam_extractor = build_sam3d_extractor(sam_dev)\n            print(f"  SAM3D extractor loaded (single GPU: {sam_dev})")'
new_sam='''            if os.environ.get("FLOWHMR_UNLOAD_BETWEEN_STAGES", "1") == "1":
                sam_extractor = None
                print(f"  SAM3D deferred until feature extraction (single GPU: {sam_dev})")
            else:
                sam_extractor = build_sam3d_extractor(sam_dev)
                print(f"  SAM3D extractor loaded (single GPU: {sam_dev})")'''
if 'SAM3D deferred until feature extraction' not in s:
    assert s.count(old_sam)==1
    s=s.replace(old_sam,new_sam)
api_anchor='        sam_tokens = self.sam_extractor.extract_video_tokens(\n'
api_reload='''        # Deferred SAM also supports the separate Web/staged feature API.
        if self.sam_extractor is None:
            from flowhmr.utils.runtime_sam_features import build_sam3d_extractor
            self.sam_extractor = build_sam3d_extractor(self.device)

'''
if '# Deferred SAM also supports' not in s:
    assert s.count(api_anchor)==1
    s=s.replace(api_anchor,api_reload+api_anchor)
if 'self._stage_memory_event("sam_ready")' not in s:
    anchor='            self.sam_extractor = build_sam3d_extractor(_dev)\n'
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+'            self._stage_memory_event("sam_ready")\n')
if 'def _stage_memory_event(' in s:
    f.write_text(s,encoding='utf-8');py_compile.compile(str(f),doraise=True)
    print('staged residency current; deferred SAM applied');raise SystemExit(0)
b=f.with_suffix('.py.before_staged_vram');shutil.copy2(f,b)
methods='''    def _stage_memory_event(self, label):
        import json, time
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            print("[stage_memory] " + json.dumps({
                "label": label, "time": time.time(),
                "allocated_GiB": torch.cuda.memory_allocated(self.device) / 2**30,
                "reserved_GiB": torch.cuda.memory_reserved(self.device) / 2**30,
            }), flush=True)

    def _park_for_camera(self):
        # Release hooks/cached SAM tensors as well as model references.
        if self.sam_extractor is not None:
            for h in getattr(self.sam_extractor, "_hooks", []):
                h.remove()
            self.sam_extractor = None
        if self.proc is not None and self.proc.detector is not None:
            self.proc.detector.model.to("cpu")
        self.pipeline.to("cpu")
        self.body_model.to("cpu")
        self.smpl_mesh.to("cpu")
        if self.postprocess is not None:
            self.postprocess.partial_body_model.to("cpu")
        import gc
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._stage_memory_event("before_camera_parked")

    def _restore_generation_models(self):
        self.pipeline.to(self.device)
        self.body_model.to(self.device)
        self.smpl_mesh.to(self.device)
        if self.postprocess is not None:
            self.postprocess.partial_body_model.to(self.device)
        self._stage_memory_event("generation_restored")

'''
anchor='    def process_video(self, video_path: str, output_dir: str, *,\n'
assert s.count(anchor)==1;s=s.replace(anchor,methods+anchor)
anchor='        # ==== Step 2: VGGT Camera ====\n'
assert s.count(anchor)==1
s=s.replace(anchor,'''        if os.environ.get("FLOWHMR_UNLOAD_BETWEEN_STAGES", "1") == "1":
            self._stage_memory_event("before_camera_unparked")
            self._park_for_camera()

'''+anchor)
anchor='        _pcb(StageName.CAMERA, 100, "VGGT camera ready")\n'
assert s.count(anchor)==1;s=s.replace(anchor,anchor+'        self._stage_memory_event("camera_finished_before_release")\n')
anchor='                release_vggt_model()\n'
assert s.count(anchor)==1;s=s.replace(anchor,anchor+'                self._stage_memory_event("after_camera_release")\n')
# Avoid rebuilding SAM for a cached feature file.
s=s.replace('        if self.sam_extractor is None:\n            from flowhmr.utils.runtime_sam_features import build_sam3d_extractor\n', '        if self.sam_extractor is None and not os.path.exists(feat_pt_path):\n            from flowhmr.utils.runtime_sam_features import build_sam3d_extractor\n')
anchor='        # ==== Step 4: Generation ====\n'
assert s.count(anchor)==1;s=s.replace(anchor,'''        if os.environ.get("FLOWHMR_UNLOAD_BETWEEN_STAGES", "1") == "1":
            self._restore_generation_models()

'''+anchor)
# A following video's detector may have been parked; restore only when uncached.
anchor='            result = self.proc.detect_and_track(working_video)\n'
assert s.count(anchor)==1;s=s.replace(anchor,'            self.proc.detector.model.to(self.device)\n'+anchor)
f.write_text(s,encoding='utf-8');py_compile.compile(str(f),doraise=True)
print('APPLIED staged VRAM; backup',b)
