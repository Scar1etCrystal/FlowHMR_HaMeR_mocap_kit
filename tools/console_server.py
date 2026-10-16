"""Local console server for the one-click video-to-FBX pipeline.

Serves the parameter page, starts a run, streams filtered progress over SSE and
reports GPU memory. Standard library only; binds to 127.0.0.1.

    <pack>\envs\python.exe tools\console_server.py [--port 8760] [--no-browser]

Endpoints
    GET  /                 the page (parameters.html from the package root)
    GET  /api/settings     current pipeline_settings.json
    GET  /api/videos       videos found in inputs\
    POST /api/upload?name= raw request body saved as a video
    POST /api/run          {"video": "...", ...16 settings keys...}
    POST /api/stop         terminate the running job
    POST /api/reveal       open the last output folder
    GET  /api/status       one-shot status snapshot
    GET  /api/events       SSE: 'status' and 'log' events
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PACK = Path(__file__).resolve().parents[1]
TOOLS = PACK / 'tools'
RUNNER = TOOLS / 'video_to_fbx.py'
SETTINGS = PACK / 'pipeline_settings.json'
PAGE = PACK / 'parameters.html'
UPLOAD_DIR = PACK / 'inputs' / '_console_upload'

STAGES = ['prepare', 'camera', 'sam', 'generate', 'hamer', 'merge']
STAGE_LABEL = {'prepare': 'PREPARE', 'camera': 'CAMERA', 'sam': 'SAM',
               'generate': 'GENERATE', 'hamer': 'HAMER', 'merge': 'MERGE'}
VIDEO_EXT = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v', '.mpg', '.mpeg', '.wmv', '.flv'}

# ---------------------------------------------------------------------------
# Model weights. The package ships code only; these files are downloaded
# separately (see WEIGHTS_AND_ENV.md and weights_manifest.json). Checking them
# up front turns a confusing mid-run crash into a clear message.
#   (relative path, human label, where to get it)
# ---------------------------------------------------------------------------
WEIGHTS = [
    ('app/flowhmr/checkpoints/flowhmr_latest/flowhmr_latest.ckpt', 'FlowHMR checkpoint',
     'app/flowhmr/docs/resources.md'),
    ('app/flowhmr/ckpts/yolox/yolox_l.pth', 'YOLOX person detector',
     'github.com/Megvii-BaseDetection/YOLOX releases (yolox_l.pth)'),
    ('app/flowhmr/ckpts/vggt-omega/vggt_omega_1b_512.pt', 'VGGT-Omega camera model',
     'huggingface.co/facebook/VGGT-Omega (gated, request access)'),
    ('app/flowhmr/ckpts/sam-3d-body-dinov3/model.ckpt', 'SAM 3D Body checkpoint',
     'huggingface.co/facebook/sam-3d-body-dinov3 (gated, request access)'),
    ('app/flowhmr/ckpts/sam-3d-body-dinov3/assets/mhr_model.pt', 'SAM 3D Body MHR asset',
     'huggingface.co/facebook/sam-3d-body-dinov3'),
    ('app/flowhmr/ckpts/sam-3d-body-dinov3/model_config.yaml', 'SAM 3D Body config',
     'huggingface.co/facebook/sam-3d-body-dinov3'),
    ('app/flowhmr/assets/body_models/smplh/neutral/model.npz', 'SMPL-H neutral body model',
     'mano.is.tue.mpg.de (AMASS smplh_amass_neutral; registration required)'),
    ('app/flowhmr/assets/body_models/smpl_neutral_J_regressor.pt', 'SMPL joint regressor',
     'GVHMR public asset'),
    ('app/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt', 'HaMeR checkpoint',
     'HaMeR official release'),
    ('app/hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth', 'ViTPose whole-body weights',
     'ViTPose official release'),
    ('app/hamer/_DATA/data/mano/MANO_RIGHT.pkl', 'MANO right hand model',
     'mano.is.tue.mpg.de (shipped with the HaMeR data)'),
    ('app/hamer/_DATA/data/mano/MANO_LEFT.pkl', 'MANO left hand model',
     'mano.is.tue.mpg.de (shipped with the HaMeR data)'),
    ('app/hamer/models/model_final_ef3a80.pkl', 'Detectron2 RegNet backbone',
     'Detectron2 model zoo'),
]


def preflight():
    """Report which required weight files are absent (size 0 counts as absent)."""
    missing, present = [], 0
    for rel, label, source in WEIGHTS:
        p = PACK / rel
        try:
            ok = p.is_file() and p.stat().st_size > 0
        except OSError:
            ok = False
        if ok:
            present += 1
        else:
            missing.append({'path': rel, 'label': label, 'source': source})
    return {'ready': not missing, 'present': present, 'total': len(WEIGHTS),
            'missing': missing}

# ---------------------------------------------------------------------------
# Log filtering. The pipeline prints a lot of harmless library noise: torch
# deprecation warnings, source-line echoes, and checkpoint key "mismatch"
# reports that are expected for this package. None of it is actionable, and it
# buries the real progress lines, so it is dropped here and counted instead.
# ---------------------------------------------------------------------------
WARN_HEADER = re.compile(r':\s*\w*Warning\s*:')
DROP_PATTERNS = [
    re.compile(r'^\s*warnings\.warn\b'),
    re.compile(r'^\s*@autocast\(enabled=False\)\s*$'),
    # checkpoint key reports: expected for this package, never actionable
    re.compile(r'missing keys in source state_dict'),
    re.compile(r'Missing keys\s*\('),
    re.compile(r'unexpected key'),
    re.compile(r'do not match exactly'),
    re.compile(r'^\s*The model and loaded state dict'),
    # plain-text library warnings without a "XxxWarning:" header
    re.compile(r'^\s*WARNING\b'),
    re.compile(r'^\s*Warning\b'),
    re.compile(r'^\s*apex is not installed'),
    re.compile(r'^\s*Ignored kwargs:'),
    # multi-line torch warnings whose header was already dropped
    re.compile(r'Please either pass the dim explicitly'),
    re.compile(r'The default value of dim will change'),
    re.compile(r'^\s*b3 = torch\.cross'),
    re.compile(r'You are using `torch\.'),
    re.compile(r'^\s*model_data = pickle\.load'),
]


class LogFilter:
    """Drops warning blocks and checkpoint-key noise; counts what it dropped."""

    def __init__(self):
        self.dropped = 0
        self._paren = 0        # inside a multi-line warnings.warn(...) call
        self._echo = 0         # source lines echoed under a warning header

    def feed(self, line: str):
        """Return the line to keep, or None when it should be suppressed."""
        text = line.rstrip('\n')
        if self._paren > 0:
            self._paren += text.count('(') - text.count(')')
            self.dropped += 1
            return None
        if self._echo > 0 and (text.startswith((' ', '\t')) or not text.strip()):
            self._echo -= 1
            self.dropped += 1
            return None
        self._echo = 0
        if WARN_HEADER.search(text):
            self.dropped += 1
            self._echo = 2
            return None
        if any(p.search(text) for p in DROP_PATTERNS):
            self.dropped += 1
            if 'warnings.warn(' in text:
                self._paren = max(0, text.count('(') - text.count(')'))
            return None
        return text


# ---------------------------------------------------------------------------
# Progress parsing
# ---------------------------------------------------------------------------
RE_STAGE = re.compile(r'^===\s*(\w+)\s*===$')
RE_OUTPUT = re.compile(r'^Output:\s*(.+)$')
RE_BBOX = re.compile(r'bbox detection:\s*(\d+)/(\d+)\s*frames')
RE_SAM = re.compile(r'\[sam\s+[^\]]*\]\s*(\d+)/(\d+)\s*frames')
RE_HAMER_ROW = re.compile(r'^\s*(\d+)/(\d+)\s')
RE_GEN_DONE = re.compile(r'^GENERATION_DONE\s+(\d+)')
RE_DONE = re.compile(r'^DONE:\s*(.+)$')
RE_FAIL = re.compile(r'^(FAILED:|RuntimeError|ValueError|FileNotFoundError|OSError)')
RE_TRACE = re.compile(r'^Traceback \(most recent call last\)')


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = 'idle'
        self.stage = None
        self.stage_index = None
        self.progress = {'current': None, 'total': None, 'label': None}
        self.started = None
        self.elapsed = 0.0
        self.exit_code = None
        self.filtered = 0
        self.last_error = None
        self.video = None
        self.output = None
        self.job = None
        self.gpu = {'used_mib': None, 'total_mib': None, 'util': None, 'torch_peak_gib': None}
        self.log_lines = 0
        self.proc = None
        self.subscribers = set()
        self.last_client = time.time()
        self.ever_connected = False
        # Replayed to a page that connects (or reloads) after the run started,
        # so the log panel is never mysteriously empty mid-run.
        self.log_buffer = deque(maxlen=500)

    def snapshot(self):
        with self.lock:
            return {
                'state': self.state, 'stage': self.stage, 'stage_index': self.stage_index,
                'stage_total': len(STAGES), 'progress': dict(self.progress),
                'elapsed_seconds': round(self.elapsed, 1), 'exit_code': self.exit_code,
                'filtered_lines': self.filtered, 'gpu': dict(self.gpu),
                'last_error': self.last_error, 'video': self.video,
                'output': self.output, 'job': self.job, 'log_lines': self.log_lines,
                'preflight': preflight(),
            }

    def publish(self, event, payload):
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait((event, payload))
            except queue.Full:
                pass


ST = State()

# Written by main(); used by the watchdog to shut the server down.
SRV = None
# Seconds with no page connected before the server exits by itself. 0 disables.
# A running job always keeps the server alive, so closing the tab mid-run does
# not kill the pipeline.
AUTO_STOP_GRACE = 25.0


def touch_client():
    with ST.lock:
        ST.last_client = time.time()
        ST.ever_connected = True


def watchdog():
    while True:
        time.sleep(2.0)
        if AUTO_STOP_GRACE <= 0:
            continue
        with ST.lock:
            if not ST.ever_connected:
                continue
            subscribers = len(ST.subscribers)
            running = ST.state == 'running'
            idle_for = time.time() - ST.last_client
        if subscribers == 0 and not running and idle_for > AUTO_STOP_GRACE:
            print(f'\nno page connected for {int(idle_for)}s and no job running'
                  '\nstopping the server (restart with Open_Console.bat)')
            if SRV is not None:
                threading.Thread(target=SRV.shutdown, daemon=True).start()
            return


def gpu_sample():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        used, total, util = [p.strip() for p in out.stdout.strip().splitlines()[0].split(',')]
        return {'used_mib': int(used), 'total_mib': int(total), 'util': int(util)}
    except Exception:
        return None


def torch_peak_from_job(job: Path):
    """Highest allocated peak the pipeline itself reported, in GiB."""
    best = None
    for name, keys in [('generation_info.json', ('peak_alloc_GiB',)),
                       ('sam_run_info.json', ('peak_alloc_GiB',)),
                       ('camera_info.json', ('peak_alloc_GiB',))]:
        p = job / name
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            for k in keys:
                v = data.get(k)
                if isinstance(v, (int, float)):
                    best = v if best is None else max(best, float(v))
        except Exception:
            pass
    return round(best, 2) if best is not None else None


def gpu_loop():
    while True:
        sample = gpu_sample()
        if sample:
            with ST.lock:
                ST.gpu.update(sample)
                if ST.job:
                    peak = torch_peak_from_job(Path(ST.job))
                    if peak is not None:
                        ST.gpu['torch_peak_gib'] = peak
            ST.publish('status', ST.snapshot())
        time.sleep(1.0)


def elapsed_loop():
    while True:
        with ST.lock:
            if ST.state == 'running' and ST.started:
                ST.elapsed = time.time() - ST.started
        time.sleep(0.5)


def handle_line(raw: str):
    line = raw.rstrip('\n')
    kind = 'info'

    m = RE_STAGE.match(line.strip())
    if m and m.group(1).lower() in STAGES:
        stage = m.group(1).lower()
        with ST.lock:
            ST.stage = stage
            ST.stage_index = STAGES.index(stage)
            ST.progress = {'current': None, 'total': None, 'label': STAGE_LABEL[stage]}
        kind = 'stage'
    elif RE_OUTPUT.match(line):
        with ST.lock:
            ST.job = RE_OUTPUT.match(line).group(1).strip()
    else:
        m = RE_BBOX.search(line)
        if m:
            with ST.lock:
                ST.progress = {'current': int(m.group(1)), 'total': int(m.group(2)), 'label': 'PREPARE'}
            kind = 'progress'
        else:
            m = RE_SAM.search(line)
            if m:
                with ST.lock:
                    ST.progress = {'current': int(m.group(1)), 'total': int(m.group(2)), 'label': 'SAM'}
                kind = 'progress'
            else:
                m = RE_GEN_DONE.match(line)
                if m:
                    with ST.lock:
                        ST.progress = {'current': int(m.group(1)), 'total': int(m.group(1)), 'label': 'GENERATE'}
                    kind = 'progress'
                else:
                    m = RE_DONE.match(line)
                    if m:
                        with ST.lock:
                            ST.output = m.group(1).strip()
                            ST.state = 'done'
                        # 'info', not 'stage': the stage style uppercases the
                        # text, which would misreport the path's real casing.
                        kind = 'info'
                    elif RE_TRACE.match(line) or RE_FAIL.match(line):
                        with ST.lock:
                            ST.last_error = line[:400]
                        kind = 'error'

    with ST.lock:
        ST.log_lines += 1
        ST.filtered = _filter.dropped
        ST.log_buffer.append({'line': line, 'kind': kind})
    ST.publish('log', {'line': line, 'kind': kind})
    ST.publish('status', ST.snapshot())


_filter = LogFilter()


def pump(proc):
    try:
        for raw in proc.stdout:
            keep = _filter.feed(raw)
            if keep is not None:
                handle_line(keep)
        proc.wait()
    except Exception as exc:                                  # pragma: no cover
        with ST.lock:
            ST.last_error = f'log reader failed: {exc}'
    finally:
        code = proc.returncode
        with ST.lock:
            ST.exit_code = code
            if ST.state == 'running':
                ST.state = 'done' if code == 0 else ('stopped' if ST.state == 'stopped' else 'failed')
            ST.proc = None
            if code not in (0, None) and not ST.last_error:
                ST.last_error = f'pipeline exited with code {code}'
            ST.filtered = _filter.dropped
        ST.publish('status', ST.snapshot())


def start_run(payload: dict):
    with ST.lock:
        if ST.state == 'running' and ST.proc is not None:
            return False, 'a run is already in progress'
    video = str(payload.get('video') or '').strip().strip('"')
    if not video:
        return False, 'no video path given'
    pf = preflight()
    if not pf['ready']:
        names = '; '.join(m['label'] for m in pf['missing'][:4])
        more = '' if len(pf['missing']) <= 4 else f" (+{len(pf['missing']) - 4} more)"
        return False, f"model weights missing: {names}{more}"
    vp = Path(video)
    if not vp.is_file():
        return False, f'video not found: {video}'
    if vp.suffix.lower() not in VIDEO_EXT:
        return False, f'unsupported video extension: {vp.suffix}'
    settings = {k: v for k, v in payload.items() if k != 'video'}
    if not settings:
        settings = json.loads(SETTINGS.read_text(encoding='utf-8'))
    run_dir = PACK / 'outputs' / '_console_run'
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / 'pipeline_settings.json'
    cfg_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')

    global _filter
    _filter = LogFilter()
    with ST.lock:
        ST.state = 'running'
        ST.stage = None
        ST.stage_index = None
        ST.progress = {'current': None, 'total': None, 'label': None}
        ST.started = time.time()
        ST.elapsed = 0.0
        ST.exit_code = None
        ST.last_error = None
        ST.video = str(vp)
        ST.output = None
        ST.job = None
        ST.log_lines = 0
        ST.filtered = 0
        ST.log_buffer.clear()
    ST.publish('status', ST.snapshot())

    env = dict(os.environ)
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    proc = subprocess.Popen(
        [sys.executable, '-u', str(RUNNER), str(vp), '--settings', str(cfg_path)],
        cwd=str(PACK), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', bufsize=1, env=env)
    with ST.lock:
        ST.proc = proc
    threading.Thread(target=pump, args=(proc,), daemon=True).start()
    return True, None


def stop_run():
    with ST.lock:
        proc = ST.proc
        if proc is None:
            return False, 'nothing is running'
        ST.state = 'stopped'
    try:
        proc.terminate()
    except Exception as exc:
        return False, str(exc)
    return True, None


def browse_for_video():
    """Open a native Windows file dialog and return the chosen path.

    Runs on Windows only; the dialog is modal, so the request that asked for it
    blocks until the user picks a file or cancels.
    """
    if os.name != 'nt':
        return None, 'the file picker is only available on Windows'
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms | Out-Null;"
        "$d = New-Object System.Windows.Forms.OpenFileDialog;"
        "$d.Title = 'Select a video';"
        "$d.Filter = 'Video files|*.mp4;*.mkv;*.mov;*.avi;*.webm;*.m4v;*.mpg;*.mpeg;*.wmv|All files|*.*';"
        "$d.CheckFileExists = $true;"
        "$d.Multiselect = $false;"
        "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
        "{ [Console]::Out.Write($d.FileName) }"
    )
    try:
        out = subprocess.run(['powershell.exe', '-NoProfile', '-STA', '-Command', ps],
                             capture_output=True, text=True, timeout=300)
    except Exception as exc:
        return None, f'file picker failed: {exc}'
    path = (out.stdout or '').strip()
    if not path:
        return None, None          # cancelled
    if Path(path).suffix.lower() not in VIDEO_EXT:
        return None, f'unsupported video extension: {Path(path).suffix}'
    return path, None


def list_videos():
    out = []
    roots = [PACK / 'inputs', PACK / 'inputs' / '_console_upload', PACK]
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXT and p not in seen:
                seen.add(p)
                out.append({'name': p.name, 'path': str(p),
                            'mb': round(p.stat().st_size / 2**20, 1)})
    return out[:200]


class ConsoleServer(ThreadingHTTPServer):
    """Bind exclusively.

    On Windows SO_REUSEADDR (the HTTPServer default) lets a second instance
    bind a port that is already in use, so the older process keeps receiving
    every request and the new code silently does nothing. Refuse to share and
    let the caller report a clear error instead.
    """
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    server_version = 'MocapKitConsole/1.0'
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):        # keep our own console readable
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'),
                   'application/json; charset=utf-8')

    def do_GET(self):
        touch_client()
        path = urlparse(self.path).path
        if path in ('/', '/index.html', '/parameters.html'):
            if not PAGE.is_file():
                self._json(500, {'ok': False, 'error': f'missing {PAGE.name}'})
                return
            self._send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
        elif path == '/api/settings':
            try:
                cfg = json.loads(SETTINGS.read_text(encoding='utf-8'))
            except Exception as exc:
                self._json(500, {'ok': False, 'error': str(exc)})
                return
            self._json(200, {'ok': True, 'settings': cfg})
        elif path == '/api/videos':
            self._json(200, {'ok': True, 'videos': list_videos()})
        elif path == '/api/preflight':
            self._json(200, {'ok': True, **preflight()})
        elif path == '/api/status':
            self._json(200, ST.snapshot())
        elif path == '/api/events':
            self._sse()
        else:
            self._json(404, {'ok': False, 'error': 'not found'})

    def do_POST(self):
        touch_client()
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/api/run':
            body = self._read_body()
            try:
                payload = json.loads(body or b'{}')
            except Exception as exc:
                self._json(400, {'ok': False, 'error': f'bad json: {exc}'})
                return
            ok, err = start_run(payload)
            self._json(200 if ok else 409, {'ok': ok, 'error': err})
        elif path == '/api/stop':
            ok, err = stop_run()
            self._json(200 if ok else 409, {'ok': ok, 'error': err})
        elif path == '/api/browse':
            chosen, err = browse_for_video()
            if err:
                self._json(500, {'ok': False, 'error': err})
            elif not chosen:
                self._json(200, {'ok': False, 'error': 'cancelled'})
            else:
                self._json(200, {'ok': True, 'path': chosen})
        elif path == '/api/reveal':
            with ST.lock:
                target = ST.output or ST.job
            if not target:
                self._json(409, {'ok': False, 'error': 'no output yet'})
                return
            p = Path(target)
            folder = p if p.is_dir() else p.parent
            try:
                if os.name == 'nt':
                    os.startfile(str(folder))          # noqa: S606
                else:
                    subprocess.Popen(['xdg-open', str(folder)])
            except Exception as exc:
                self._json(500, {'ok': False, 'error': str(exc)})
                return
            self._json(200, {'ok': True})
        elif path == '/api/upload':
            name = parse_qs(parsed.query).get('name', ['video.mp4'])[0]
            name = Path(unquote(name)).name or 'video.mp4'
            safe = re.sub(r'[^\w.\- ]+', '_', name, flags=re.UNICODE)[:120]
            if Path(safe).suffix.lower() not in VIDEO_EXT:
                self._json(400, {'ok': False, 'error': 'unsupported video extension'})
                return
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest = UPLOAD_DIR / safe
            remaining = int(self.headers.get('Content-Length') or 0)
            tmp = dest.with_suffix(dest.suffix + '.part')
            try:
                with open(tmp, 'wb') as fh:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
                os.replace(tmp, dest)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                self._json(500, {'ok': False, 'error': str(exc)})
                return
            self._json(200, {'ok': True, 'path': str(dest)})
        else:
            self._json(404, {'ok': False, 'error': 'not found'})

    def _read_body(self) -> bytes:
        n = int(self.headers.get('Content-Length') or 0)
        return self.rfile.read(n) if n else b''

    def _sse(self):
        q: queue.Queue = queue.Queue(maxsize=2000)
        with ST.lock:
            ST.subscribers.add(q)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        try:
            self._write_event('status', ST.snapshot())
            with ST.lock:
                backlog = list(ST.log_buffer)
            for item in backlog:
                self._write_event('log', item)
            while True:
                try:
                    event, payload = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b': keep-alive\n\n')
                    self.wfile.flush()
                    touch_client()
                    continue
                self._write_event(event, payload)
        except (BrokenPipeError, ConnectionAbortedError, OSError):
            pass
        finally:
            with ST.lock:
                ST.subscribers.discard(q)

    def _write_event(self, event, payload):
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f'event: {event}\ndata: {data}\n\n'.encode('utf-8'))
        self.wfile.flush()


def main():
    port = 8760
    args = sys.argv[1:]
    if '--port' in args:
        port = int(args[args.index('--port') + 1])
    url = f'http://127.0.0.1:{port}/'
    threading.Thread(target=gpu_loop, daemon=True).start()
    threading.Thread(target=elapsed_loop, daemon=True).start()
    try:
        srv = ConsoleServer(('127.0.0.1', port), Handler)
    except OSError as exc:
        print(f'cannot listen on 127.0.0.1:{port}: {exc}')
        print('Another console is probably already running. Close it, or pass --port <n>.')
        return 1
    global SRV, AUTO_STOP_GRACE
    SRV = srv
    if '--stay' in args:
        AUTO_STOP_GRACE = 0.0
    threading.Thread(target=watchdog, daemon=True).start()
    print(f'mocap kit console: {url}')
    print(f'  page      {PAGE}')
    print(f'  settings  {SETTINGS}')
    print('  Ctrl+C to stop the server (it does not stop a running job).')
    if AUTO_STOP_GRACE > 0:
        print(f'  stops by itself {int(AUTO_STOP_GRACE)}s after the last page closes'
              ' (a running job keeps it alive)')
    if '--no-browser' not in args:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nserver stopped')
    finally:
        srv.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
