"""Keep original JPEG bytes/quality, write Unicode paths safely on Windows."""
from pathlib import Path
import shutil,py_compile
f=Path(__file__).resolve().parents[1]/'app/flowhmr/flowhmr/utils/runtime_vggt_camera.py'
s=f.read_text(encoding='utf-8')
old='                cv2.imwrite(out_path, frame)'
new='''                # cv2.imwrite silently fails for Unicode paths on Windows.
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError(f"JPEG encoding failed for {out_path}")
                encoded.tofile(out_path)'''
if new in s:print('Unicode camera patch already applied')
else:
    assert s.count(old)==1
    backup=f.with_suffix('.py.before_unicode')
    if not backup.exists():shutil.copy2(f,backup)
    f.write_text(s.replace(old,new),encoding='utf-8');py_compile.compile(str(f),doraise=True)
    print('Applied Unicode-safe VGGT frame writing')
