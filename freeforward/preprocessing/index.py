"""Build image-only scene records consumed by the reconstruction dataset."""
from pathlib import Path
import json
import os
import shutil
import numpy as np
from .pose.data import index_folders, load_manifest

BASIS = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]


def prepare_rgb(input_root, output_root, split, fps=10., cameras=('1', '0', '2')):
    """Import scene/camera/numeric.jpg folders without any geometric labels.

    Files are hard-linked when possible and copied across filesystems. No
    dataset cameras, object boxes, depth, or semantic masks are read.
    """
    output_root = Path(output_root).resolve()
    scenes = index_folders(input_root, fps, reference='0')['scenes']
    if not scenes:
        raise ValueError('No RGB scenes found')
    records = []
    indexed = []
    for sid, scene in enumerate(scenes):
        if set(scene['camera_ids']) != set(cameras):
            raise ValueError('Use three camera folders named 1 (left), 0 (front), 2 (right)')
        name = scene['scene_name']
        timestamp_file = Path(input_root).resolve() / name / 'timestamps.json'
        supplied_times = json.loads(timestamp_file.read_text()) if timestamp_file.exists() else None
        if supplied_times is not None and len(supplied_times) != len(scene['frames']):
            raise ValueError('Timestamp count differs from synchronized RGB frame count')
        relative = {c: [] for c in cameras}
        frames = []
        times = []
        for frame_id, frame in enumerate(scene['frames']):
            images = {}
            timestamp = supplied_times[frame_id] if supplied_times is not None else frame['timestamp']
            times.append(timestamp)
            for camera in cameras:
                source = Path(frame['images'][camera]).resolve()
                relative_path = f'{split}/{name}/images/{frame_id:03d}_{camera}.jpg'
                destination = output_root / 'datasets/waymo' / relative_path.replace('images/', 'images_4/')
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if destination.stat().st_size != source.stat().st_size:
                        raise ValueError(f'RGB output already exists with different size: {destination}')
                else:
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
                relative[camera].append(relative_path)
                images[camera] = str(destination)
            frames.append(dict(frame_id=frame_id, timestamp=timestamp, images=images))
        # "waymo" selects the established three-camera coordinate convention;
        # these JSON records contain image paths/times only, not GT annotations.
        record = dict(dataset='waymo', source_dataset=scene.get('source_dataset', 'rgb'),
                      scene_id=sid, scene_name=name, fps=float(fps), num_timesteps=len(frames),
                      normalized_time=times, relative_image_path=relative,
                      instances_info={}, frame_instances={})
        path = output_root / 'scene_metadata' / split / (name + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))
        records.append(str(path.relative_to(output_root)))
        indexed.append(dict(scene_name=name, fps=float(fps), camera_ids=list(cameras),
                            reference_camera='0', opencv_to_dataset=BASIS, frames=frames))
    path = output_root / 'scene_list' / f'waymo_{split}.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(records) + '\n')
    manifest = output_root / f'rgb_{split}.json'
    manifest.write_text(json.dumps(dict(schema_version=1, scenes=indexed), indent=2))
    load_manifest(manifest)
    return manifest
