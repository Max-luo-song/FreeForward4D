"""Separate, fail-closed MoGe supervision store; never reads GT depth."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


class PseudoDepthStore:
    def __init__(self, root, split, contract_id, target_size, cache_size=1024):
        self.root = Path(root)
        self.split = split
        self.target_size = tuple(target_size)
        self.cache_size = max(0, int(cache_size))
        self.cache = OrderedDict()
        contract = json.loads((self.root / "contract.json").read_text())
        if not contract_id or contract["id"] != contract_id:
            raise ValueError("Pseudo-depth contract does not match the pinned training config")
        self.contract_id = contract_id
        spec = contract["specification"]
        if (spec["model"] != "MoGe-2" or spec["units"] != "metres" or spec["depth_type"] != "camera_z"
                or spec["gt_geometry_used"] or spec["additional_gca_scale"]
                or (spec["height"], spec["width"]) != self.target_size):
            raise ValueError("Unsupported pseudo-depth units/source/grid contract")

    def load(self, scene, camera, frame, rgb_path, data_root):
        key = (scene, str(camera), int(frame))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.root / self.split / scene / f"camera_{camera}" / f"{frame:06d}.npz"
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            depth = data["depth"].copy()
            valid = data["valid"].copy()
        expected = dict(scene_name=scene, camera_id=str(camera), frame_id=int(frame),
                        split=self.split, contract_id=self.contract_id,
                        rgb_relative_path=str(Path(rgb_path).relative_to(data_root)))
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Pseudo-depth scene/frame/camera/source mismatch: {path}")
        if hashlib.sha256(Path(rgb_path).read_bytes()).hexdigest() != metadata["rgb_sha256"]:
            raise ValueError(f"RGB content no longer matches teacher input: {rgb_path}")
        if depth.shape != self.target_size or valid.shape != self.target_size:
            raise ValueError(f"Pseudo-depth pixel grid mismatch; do not use GT max-pooling: {path}")
        if depth.dtype != np.float32 or not np.isin(valid, [0, 1]).all():
            raise ValueError(f"Invalid pseudo-depth dtype/mask: {path}")
        valid = valid.astype(bool)
        if not np.isfinite(depth).all() or not valid.any() or (depth[valid] <= .01).any() or (depth[~valid] != 0).any():
            raise ValueError(f"Malformed or empty pseudo-depth label: {path}")
        result = torch.from_numpy(depth), torch.from_numpy(valid)
        if self.cache_size:
            self.cache[key] = result
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return result
