"""Live end-to-end check of the console server: real run, real SSE stream.

Drives the same endpoints the page uses and reports what the page would show.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:8760'
PACK = Path(r'F:\flowhmr_pack')
VIDEO = sys.argv[1] if len(sys.argv) > 1 else str(PACK / 'inputs' / 'example.mp4')

statuses = []
logs = []
stages_seen = []
gpu_samples = []
errors = []
stop = threading.Event()


def sse():
    req = urllib.request.Request(BASE + '/api/events')
    with urllib.request.urlopen(req, timeout=600) as r:
        event = None
        while not stop.is_set():
            raw = r.readline()
            if not raw:
                break
            line = raw.decode('utf-8', errors='replace').rstrip('\n')
            if line.startswith('event: '):
                event = line[7:]
            elif line.startswith('data: '):
                payload = json.loads(line[6:])
                if event == 'status':
                    statuses.append(payload)
                    g = payload.get('gpu') or {}
                    if g.get('used_mib'):
                        gpu_samples.append((round(time.time() - t0, 1), g.get('used_mib'),
                                            g.get('util'), g.get('torch_peak_gib')))
                    if payload.get('stage') and (not stages_seen or stages_seen[-1] != payload['stage']):
                        stages_seen.append(payload['stage'])
                    if payload.get('last_error'):
                        errors.append(payload['last_error'])
                elif event == 'log':
                    logs.append((payload.get('kind'), payload.get('line')))
                event = None


def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b''
    req = urllib.request.Request(BASE + path, data=data, method='POST',
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


t0 = time.time()
settings = get('/api/settings')['settings']
pf = get('/api/preflight')
print('preflight:', pf['ready'], f"{pf['present']}/{pf['total']}")
threading.Thread(target=sse, daemon=True).start()
time.sleep(1.5)

settings['video'] = VIDEO
res = post('/api/run', settings)
print('run ->', res)
if not res.get('ok'):
    raise SystemExit('run rejected')

deadline = time.time() + 600
last = None
while time.time() < deadline:
    st = get('/api/status')
    if st['state'] != 'running':
        last = st
        break
    time.sleep(5)
stop.set()
time.sleep(0.5)

if last is None:
    last = get('/api/status')
print('\n=== final state ===')
print(json.dumps({k: last[k] for k in ('state', 'stage', 'exit_code', 'elapsed_seconds',
                                       'filtered_lines', 'output', 'last_error', 'log_lines')},
                 ensure_ascii=False, indent=2))
print('\n=== stages seen ===', stages_seen)
print('=== log lines delivered ===', len(logs))
kinds = {}
for k, _ in logs:
    kinds[k] = kinds.get(k, 0) + 1
print('=== log kinds ===', kinds)
noise = [l for k, l in logs if any(w in l for w in
         ('Warning', 'warnings.warn', 'missing keys', 'Missing keys', 'do not match',
          'unexpected key', 'apex is not', 'dim will change'))]
print('=== noise leaked to the page ===', len(noise))
for n in noise[:5]:
    print('   ', n[:120])
print('\n=== progress samples (first 12) ===')
for k, l in logs:
    if k in ('progress', 'stage'):
        print(f'   [{k}] {l[:100]}')
print('\n=== GPU samples (every ~10th) ===')
for i, (t, used, util, peak) in enumerate(gpu_samples):
    if i % 10 == 0:
        print(f'   t={t:6.1f}s used={used} MiB util={util}% torch_peak={peak} GiB')
if gpu_samples:
    print('   peak driver used =', max(s[1] for s in gpu_samples), 'MiB')
print('\n=== errors ===', errors or 'none')
print('\nRESULT:', 'PASS' if last['state'] == 'done' and last['exit_code'] == 0 and not noise
      else f"CHECK state={last['state']} exit={last['exit_code']} noise={len(noise)}")
