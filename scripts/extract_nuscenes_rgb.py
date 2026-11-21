#!/usr/bin/env python3
"""Index nuScenes RGB sweeps on a synchronized 10Hz grid, without GT geometry."""
from pathlib import Path
import argparse
import bisect
import json
import os
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--fps', type=float, default=10.)
    p.add_argument('--limit-scenes', type=int)
    a = p.parse_args()
    if a.fps <= 0:
        p.error('--fps must be positive')
    metadata = a.raw_root / a.version
    read = lambda name: json.loads((metadata / (name + '.json')).read_text())
    split_file = Path(__file__).resolve().parents[1] / 'freeforward/preprocessing/nuscenes_splits.json'
    selected = set(json.loads(split_file.read_text())[a.split])
    scenes = {s['token']: s for s in read('scene') if s['name'] in selected}
    if a.limit_scenes:
        scenes = dict(list(sorted(scenes.items(), key=lambda x: x[1]['name']))[:a.limit_scenes])
    samples = {s['token']: s for s in read('sample')}
    channels = {'CAM_FRONT_LEFT': '1', 'CAM_FRONT': '0', 'CAM_FRONT_RIGHT': '2'}
    rows = {token: {c: [] for c in channels.values()} for token in scenes}
    for row in read('sample_data'):
        token = samples[row['sample_token']]['scene_token']
        if token not in scenes:
            continue
        parts = Path(row['filename']).parts
        if len(parts) < 2 or parts[1] not in channels:
            continue
        rows[token][channels[parts[1]]].append((row['timestamp'] / 1e6, row['filename']))
    for token, scene in sorted(scenes.items(), key=lambda x: x[1]['name']):
        streams = {c: sorted(v) for c, v in rows[token].items()}
        if any(not v for v in streams.values()):
            raise ValueError('Missing nuScenes frontal camera sweep files')
        start = max(v[0][0] for v in streams.values())
        end = min(v[-1][0] for v in streams.values())
        total = int((end - start) * a.fps) + 1
        folder = a.output / scene['name']
        for i in range(total):
            t = start + i / a.fps
            for camera, stream in streams.items():
                j = bisect.bisect_left(stream, (t, ''))
                candidates = [k for k in [j - 1, j] if 0 <= k < len(stream)]
                k = min(candidates, key=lambda k: abs(stream[k][0] - t))
                if abs(stream[k][0] - t) > .08:
                    raise ValueError('Camera timestamps too far apart for synchronized RGB export')
                source = a.raw_root / stream[k][1]
                if not source.is_file():
                    raise FileNotFoundError(source)
                destination = folder / camera / f'{i:06d}.jpg'
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
        (folder / 'timestamps.json').write_text(json.dumps([i / a.fps for i in range(total)]))
        print('RGB_COMPLETE', scene['name'], total, flush=True)


if __name__ == '__main__':
    main()
