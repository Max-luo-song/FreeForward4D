"""One persistent foundation-model worker on one selected CUDA device."""
from pathlib import Path
import argparse
import json
import tempfile
import time
from types import SimpleNamespace
from .pose.config import load_config, model_paths
from .pose.data import load_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    a = p.parse_args()
    job = json.loads(a.job.read_text())
    scenes = load_manifest(job['manifest'])
    scenes = [s for s in scenes if s['scene_name'] in set(job['scenes'])]
    config = load_config(job.get('geometry_config'))
    output = Path(job['output'])
    if job['stage'] == 'depth':
        from .depth import DepthExporter
        model = DepthExporter(config['moge_checkpoint'], model_paths()[1])
        for scene in scenes:
            model.export_scene(scene, output / 'depth', output, job['split'],
                               config['output_image_size'], job['depth_contract_id'])
    elif job['stage'] == 'tracks':
        from .tracking import (PersistentGroundedSAM2, stage_frames, completion_is_valid,
                               clear_partial, validate_masks, write_completion_record, remove_debug_artifacts)
        args = SimpleNamespace(gsam_root=Path(job['gsam_root']), checkpoint=Path(job['sam_checkpoint']),
                               model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml', grounding_model=job['grounding_model'],
                               text='car. truck. bus. motorcycle. bicycle. pedestrian.', step=10)
        model = PersistentGroundedSAM2(args)
        for scene in scenes:
            for camera in scene['camera_ids']:
                frames = [(f['frame_id'], Path(f['images'][camera])) for f in scene['frames']]
                path = output / 'tracks' / scene['scene_name'] / camera
                if completion_is_valid(frames, path):
                    continue
                clear_partial(path)
                path.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix='freeforward_frames_') as stage:
                    frames_dir = Path(stage) / 'frames'
                    stage_frames(frames, frames_dir)
                    model.process_video(frames, frames_dir, path)
                valid, samples = validate_masks(frames, path)
                if not valid:
                    raise ValueError(f'Incomplete SAM tracks: {path}')
                write_completion_record(path, scene['scene_name'], camera, frames, time.monotonic() - started, samples)
                remove_debug_artifacts(path)
    else:
        raise ValueError('Unknown preprocessing worker stage')
    print('WORKER_COMPLETE', job['stage'], len(scenes), flush=True)


if __name__ == '__main__':
    main()
