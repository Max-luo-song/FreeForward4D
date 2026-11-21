#!/usr/bin/env python3
"""Extract camera JPEGs and timestamps only from authorized Waymo TFRecords."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--downsample', type=int, default=4)
    p.add_argument('--limit-scenes', type=int)
    a = p.parse_args()
    if a.downsample < 1:
        p.error('--downsample must be positive')
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
        from PIL import Image
    except ImportError as exc:
        raise SystemExit('Use the separate freeforward-data environment from requirements-data.txt') from exc
    files = sorted(a.raw_root.rglob('*.tfrecord'))
    if a.limit_scenes:
        files = files[:a.limit_scenes]
    if not files:
        raise SystemExit('No .tfrecord files found')
    for file in files:
        timestamps = []
        folder = None
        count = 0
        source = dict(bytes=file.stat().st_size, downsample=a.downsample, name=file.name)
        for count, item in enumerate(tf.data.TFRecordDataset(str(file), compression_type=''), 1):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytes(item.numpy()))
            name = frame.context.name
            if '/' in name or '\\' in name or name in ('', '.', '..'):
                raise ValueError('Unsafe Waymo scene name')
            folder = a.output / name
            if count == 1:
                marker = folder / '.rgb_complete.json'
                if marker.exists() and json.loads(marker.read_text()).get('source') == source:
                    print('RGB_CACHED', name, flush=True)
                    break
            timestamps.append(frame.timestamp_micros / 1e6)
            present = set()
            for image in frame.images:
                camera = str(image.name - 1)
                if camera not in {'1', '0', '2'}:
                    continue
                present.add(camera)
                path = folder / camera / f'{count - 1:06d}.jpg'
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.stat().st_size:
                    continue
                temporary = path.with_suffix('.jpg.tmp')
                if a.downsample == 1:
                    temporary.write_bytes(image.image)
                else:
                    im = Image.open(io.BytesIO(image.image)).convert('RGB')
                    # Match the original image-only quarter-resolution conversion.
                    im = im.resize((im.width // a.downsample, im.height // a.downsample))
                    im.save(temporary, format='JPEG')
                os.replace(temporary, path)
            if present != {'1', '0', '2'}:
                raise ValueError('Frame missing a frontal camera')
        else:
            if folder is None or count < 2:
                raise ValueError('Empty or insufficient Waymo scene')
            times = [t - timestamps[0] for t in timestamps]
            (folder / 'timestamps.json').write_text(json.dumps(times))
            (folder / '.rgb_complete.json').write_text(json.dumps(dict(source=source, frames=count, gt_geometry_used=False)))
            print('RGB_COMPLETE', folder.name, count, flush=True)


if __name__ == '__main__':
    main()
