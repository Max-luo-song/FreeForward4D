#!/usr/bin/env python3
"""Download external visual-prior/perceptual weights from their original hosts.

FreeForward4D reconstruction checkpoints are distributed separately. Model
access approvals and the original publishers' licenses remain applicable.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import urllib.request


def url_download(url, destination, md5=None):
    if destination.is_file():
        if md5 and hashlib.md5(destination.read_bytes()).hexdigest() != md5:
            raise ValueError(f'Existing checkpoint hash mismatch: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.download')
    try:
        with urllib.request.urlopen(url) as response, temporary.open('wb') as f:
            shutil.copyfileobj(response, f)
        if md5 and hashlib.md5(temporary.read_bytes()).hexdigest() != md5:
            raise ValueError(f'Downloaded checkpoint hash mismatch: {destination}')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'weights')
    p.add_argument('--only', choices=['all', 'geometry', 'tracks', 'perceptual'], default='all')
    p.add_argument('--omega-checkpoint', type=Path, help='Reuse an already authorized local checkpoint')
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = []
    if a.only in ['all', 'geometry', 'tracks']:
        from huggingface_hub import hf_hub_download, snapshot_download
    if a.only in ['all', 'geometry']:
        destination = root / 'vggt_omega_1b_512.pt'
        if a.omega_checkpoint:
            if a.omega_checkpoint.resolve() != destination:
                shutil.copy2(a.omega_checkpoint, destination)
        elif not destination.exists():
            try:
                downloaded = hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt')
            except Exception as exc:
                raise SystemExit('VGGT-Omega access is gated. Request access at https://huggingface.co/facebook/VGGT-Omega, authenticate with hf auth login, or supply --omega-checkpoint.') from exc
            shutil.copy2(downloaded, destination)
        files.append(destination)
        downloaded = hf_hub_download('Ruicheng/moge-2-vitl', 'model.pt', local_dir=root / 'moge-2-vitl')
        files.append(Path(downloaded))
    if a.only in ['all', 'tracks']:
        files.append(url_download('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
                                  root / 'sam2.1_hiera_large.pt'))
        snapshot_download('IDEA-Research/grounding-dino-tiny', local_dir=root / 'grounding-dino-tiny',
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
    if a.only in ['all', 'perceptual']:
        files.append(url_download('https://download.pytorch.org/models/vgg16-397923af.pth',
                                  root / 'vgg16-397923af.pth'))
        files.append(url_download('https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth',
                                  root / 'lpips_vgg.pth', md5='d507d7349b931f0638a25a48a722f98a'))
    inventory = {}
    for path in files:
        h = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(1024 * 1024):
                h.update(block)
        inventory[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
    name = root / 'download_manifest.json'
    current = json.loads(name.read_text()) if name.exists() else {}
    current.update(inventory)
    name.write_text(json.dumps(current, indent=2))
    print('PRIOR_DOWNLOADS_COMPLETE', root)


if __name__ == '__main__':
    main()
