"""Export frozen MoGe-2 depth predictions to the model's supervision store."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import numpy as np


def depth_contract(output, size, checkpoint):
    specification = dict(model='MoGe-2', units='metres', depth_type='camera_z',
                         gt_geometry_used=False, additional_gca_scale=False,
                         height=size[0], width=size[1], resolution_level=9,
                         checkpoint=Path(checkpoint).name,
                         resize='bilinear depth; nearest mask; original metric scale')
    ident = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    value = dict(id=ident, specification=specification)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'contract.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Depth output belongs to a different model/grid configuration')
    path.write_text(json.dumps(value, indent=2))
    return ident


class DepthExporter:
    def __init__(self, checkpoint, repo):
        import torch
        from .pose.config import model_paths
        model_paths()
        sys.path.insert(0, str(repo))
        from moge.model.v2 import MoGeModel
        self.torch = torch
        self.model = MoGeModel.from_pretrained(str(checkpoint)).eval().cuda()

    def export_scene(self, scene, output, data_root, split, size, ident):
        from PIL import Image
        torch = self.torch
        data_root = Path(data_root).resolve()
        for frame in scene['frames']:
            for camera in scene['camera_ids']:
                rgb_path = Path(frame['images'][camera])
                path = Path(output) / split / scene['scene_name'] / f'camera_{camera}' / f"{frame['frame_id']:06d}.npz"
                digest = hashlib.sha256(rgb_path.read_bytes()).hexdigest()
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        old = json.loads(str(z['metadata']))
                        if old.get('rgb_sha256') == digest and old.get('contract_id') == ident:
                            continue
                    raise ValueError(f'Depth cache does not match RGB/contract: {path}')
                image = np.array(Image.open(rgb_path).convert('RGB'))
                x = torch.from_numpy(image).permute(2, 0, 1).float().cuda() / 255.
                with torch.inference_mode():
                    pred = self.model.infer(x, resolution_level=9, use_fp16=True, apply_mask=False)
                    d = pred['depth'].float()
                    m = pred['mask'].bool() & torch.isfinite(d) & (d > .01)
                    d = torch.where(m, d, torch.zeros_like(d))
                    d = torch.nn.functional.interpolate(d[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]
                    m = torch.nn.functional.interpolate(m[None, None].float(), size=size, mode='nearest')[0, 0].bool()
                    m &= torch.isfinite(d) & (d > .01)
                    d = torch.where(m, d, torch.zeros_like(d))
                if not m.any():
                    raise ValueError(f'No valid MoGe-2 depth: {rgb_path}')
                metadata = dict(scene_name=scene['scene_name'], camera_id=camera, frame_id=frame['frame_id'],
                                split=split, contract_id=ident, rgb_relative_path=str(rgb_path.relative_to(data_root)),
                                rgb_sha256=digest)
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
                    temporary = Path(f.name)
                try:
                    np.savez_compressed(temporary, depth=d.cpu().numpy().astype(np.float32),
                                        valid=m.cpu().numpy().astype(np.uint8), metadata=np.asarray(json.dumps(metadata)))
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
