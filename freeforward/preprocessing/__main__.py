"""Run RGB-only geometry, depth and tracking preprocessing on selected GPUs."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from .index import prepare_rgb
from .pose.config import ROOT, load_config
from .pose.data import load_manifest
from .pose.processes import Processes


def selectors(gpus, count):
    if gpus is not None:
        result = [x.strip() for x in gpus.split(',')]
    else:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        result = visible.split(',')[:count] if visible else [str(i) for i in range(count)]
    if not result or any(not x or x == '-1' for x in result) or len(set(result)) != len(result):
        raise ValueError('GPU selectors must be nonempty and distinct')
    if count is not None and gpus is None and len(result) < count:
        raise ValueError('Requested more GPUs than CUDA_VISIBLE_DEVICES exposes')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description='FreeForward4D multi-GPU RGB preprocessing')
    p.add_argument('--input', type=Path, help='RGB folders: scene/{1,0,2}/000000.jpg')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--fps', type=float, default=10.)
    gpu = p.add_mutually_exclusive_group()
    gpu.add_argument('--gpus', help='Physical CUDA IDs/UUIDs, e.g. 0,2,3')
    gpu.add_argument('--num-gpus', type=int, default=1)
    p.add_argument('--stages', default='geometry,depth,tracks')
    p.add_argument('--geometry-config', type=Path)
    p.add_argument('--gsam-root', type=Path, default=ROOT / 'third_party/Grounded-SAM-2')
    p.add_argument('--sam-checkpoint', type=Path, default=ROOT / 'weights/sam2.1_hiera_large.pt')
    p.add_argument('--grounding-model', default=str(ROOT / 'weights/grounding-dino-tiny'))
    p.add_argument('--dry-run', action='store_true', help='Print stage/GPU plan; no model imports or GPU jobs')
    a = p.parse_args(argv)
    devices = selectors(a.gpus, a.num_gpus)
    stages = [s.strip() for s in a.stages.split(',')]
    if not stages or set(stages) - {'geometry', 'depth', 'tracks'} or len(stages) != len(set(stages)):
        p.error('--stages must contain distinct geometry,depth,tracks values')
    output = a.output.resolve()
    if a.dry_run:
        print(json.dumps(dict(gpus=devices, stages=stages, output=str(output), split=a.split,
                              geometry='shared affine metric calibration', resume=True), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f'rgb_{a.split}.json'
    if a.input:
        manifest = prepare_rgb(a.input, output, a.split, a.fps)
    if not manifest.is_file():
        p.error('Provide --input, or reuse an output containing rgb_<split>.json')
    scenes = load_manifest(manifest)
    config = load_config(a.geometry_config)
    binding = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   geometry_config=config, sam_checkpoint=str(a.sam_checkpoint.resolve()),
                   grounding_model=a.grounding_model)
    contract = output / f'.preprocessing_{a.split}.json'
    if contract.exists() and json.loads(contract.read_text()) != binding:
        raise ValueError('Output belongs to different input/model settings; select a new output directory')
    contract.write_text(json.dumps(binding, indent=2))
    if 'geometry' in stages:
        from .pose.runtime import prepare
        env_old = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
        try:
            prepare(scenes, config, output / 'geometry' / a.split, len(devices))
        finally:
            if env_old is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = env_old
    depth_id = None
    if 'depth' in stages:
        from .depth import depth_contract
        depth_id = depth_contract(output / 'depth', config['output_image_size'], config['moge_checkpoint'])
    for stage in [s for s in stages if s != 'geometry']:
        processes = Processes()
        jobs = []
        directory = output / '.jobs' / a.split
        directory.mkdir(parents=True, exist_ok=True)
        for rank, device in enumerate(devices):
            selected = [s['scene_name'] for s in scenes[rank::len(devices)]]
            if not selected:
                continue
            job = dict(stage=stage, manifest=str(manifest), scenes=selected, output=str(output),
                       split=a.split, depth_contract_id=depth_id,
                       geometry_config=str(a.geometry_config.resolve()) if a.geometry_config else None,
                       gsam_root=str(a.gsam_root.resolve()), sam_checkpoint=str(a.sam_checkpoint.resolve()),
                       grounding_model=a.grounding_model)
            path = directory / f'{stage}_{rank}.json'
            path.write_text(json.dumps(job, indent=2))
            jobs.append((device, path))
        def worker(job):
            device, path = job
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED='1')
            with path.with_suffix('.log').open('a', buffering=1) as log:
                processes.run([sys.executable, '-u', '-m', 'freeforward.preprocessing.worker',
                               '--job', str(path)], env, log)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(worker, j) for j in jobs]
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            except BaseException:
                processes.cancel()
                for future in futures:
                    future.cancel()
                raise
            finally:
                processes.cancel()
        print('STAGE_COMPLETE', stage, flush=True)
    print('PREPROCESSING_COMPLETE', output)


if __name__ == '__main__':
    main()
