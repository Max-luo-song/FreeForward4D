#!/usr/bin/env python3
"""One command: optional authorized download, RGB extraction, GPU preprocessing."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['waymo', 'nuscenes', 'rgb'], required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--gpus', default='0')
    p.add_argument('--download-source')
    p.add_argument('--geometry-config', type=Path)
    p.add_argument('--limit-scenes', type=int)
    p.add_argument('--dry-run', action='store_true')
    a, extra = p.parse_known_args()
    commands = []
    if a.download_source:
        commands.append([sys.executable, str(ROOT / 'scripts/download_data.py'), '--source', a.download_source,
                         '--output', str(a.raw_root), '--extract'])
    rgb = a.raw_root if a.dataset == 'rgb' else a.output / 'rgb_frames' / a.split
    if a.dataset != 'rgb':
        python = os.environ.get('RAW_PYTHON', sys.executable) if a.dataset == 'waymo' else sys.executable
        command = [python, str(ROOT / f'scripts/extract_{a.dataset}_rgb.py'), '--raw-root', str(a.raw_root), '--output', str(rgb)]
        if a.dataset == 'nuscenes':
            command += ['--split', a.split]
        if a.limit_scenes:
            command += ['--limit-scenes', str(a.limit_scenes)]
        commands.append(command)
    command = [sys.executable, '-m', 'freeforward.preprocessing', '--input', str(rgb), '--output', str(a.output),
               '--split', a.split, '--gpus', a.gpus]
    if a.geometry_config:
        command += ['--geometry-config', str(a.geometry_config)]
    commands.append(command + extra)
    for command in commands:
        if a.dry_run:
            import shlex
            print(shlex.join(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
