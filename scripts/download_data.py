#!/usr/bin/env python3
"""Download a dataset from an authorized GCS prefix or HTTPS archive URL."""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile


def safe_destination(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    path.relative_to(root)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='Dataset URL/prefix supplied after official access approval')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sha256')
    p.add_argument('--extract', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.source.startswith('gs://'):
        if shutil.which('gcloud'):
            subprocess.run(['gcloud', 'storage', 'rsync', '--recursive', a.source, str(a.output)], check=True)
        elif shutil.which('gsutil'):
            subprocess.run(['gsutil', '-m', 'rsync', '-r', a.source, str(a.output)], check=True)
        else:
            p.error('Install Google Cloud CLI and authenticate with your authorized dataset account')
        return
    if not a.source.startswith('https://'):
        p.error('Only authorized HTTPS or gs:// sources are supported')
    path = a.output / 'dataset_download.archive'
    temporary = path.with_suffix('.partial')
    h = hashlib.sha256()
    with urllib.request.urlopen(a.source) as response, temporary.open('wb') as f:
        while chunk := response.read(1024 * 1024):
            h.update(chunk)
            f.write(chunk)
    if a.sha256 and h.hexdigest() != a.sha256:
        temporary.unlink()
        raise ValueError('Dataset archive SHA256 mismatch')
    temporary.replace(path)
    if a.extract:
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                for member in archive:
                    destination = safe_destination(a.output, member.name)
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as src, destination.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                    else:
                        raise ValueError('Links and special files are not accepted in dataset archives')
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    destination = safe_destination(a.output, member.filename)
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as src, destination.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
        else:
            p.error('Downloaded file is not a supported tar/zip archive')
    print('DOWNLOAD_COMPLETE', a.output, 'sha256=' + h.hexdigest())


if __name__ == '__main__':
    main()
