"""Shared image offsets and bounded window scales, fitted from RGB predictions."""
from pathlib import Path
import json
import os
import tempfile
import numpy as np
from .affine_scale import fit


def write_anchors(path, depth, confidence, selected, metric, entries, scene, raw_path):
    arrays = {}
    rows = []
    support = []
    for idx in selected:
        d = depth[idx].cpu().numpy()[::4, ::4].astype(float)
        c = confidence[idx].cpu().numpy()[::4, ::4].astype(float)
        m = metric[idx][0].cpu().numpy()[::4, ::4].astype(float)
        mask = metric[idx][1].cpu().numpy()[::4, ::4].astype(bool)
        metric_valid = mask & np.isfinite(m) & (m > 1e-4)
        if not metric_valid.any():
            continue
        q = float(np.median(m[metric_valid]))
        valid = metric_valid & np.isfinite(d) & (d > 1e-4) & (c > np.quantile(c, .5))
        ix = np.flatnonzero(valid)
        if len(ix) < 100:
            continue
        support.extend((m.ravel()[ix] / d.ravel()[ix]).tolist())
        ix = ix[np.linspace(0, len(ix) - 1, min(2048, len(ix)), dtype=int)]
        k = len(rows)
        arrays[f"relative_{k}"] = d.ravel()[ix]
        arrays[f"metric_{k}"] = m.ravel()[ix]
        e = entries[idx]
        rows.append(dict(image=f"{scene}/{int(e['frame_id']):06d}_{e['camera_id']}", metric_reference=q))
    if len(support) < 100:
        raise ValueError("Insufficient valid depth for shared metric calibration")
    arrays['metadata'] = np.asarray(json.dumps(rows))
    arrays['support_band'] = np.quantile(support, [.4, .6])
    with np.load(raw_path, allow_pickle=False) as raw:
        arrays['raw_c2w'] = raw['omega_c2w_raw'].copy()
        arrays['frame_ids'] = raw['frame_ids'].copy()
        arrays['camera_ids'] = raw['camera_ids'].copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def calibrate_scene(work, scene, starts):
    """Solve all window/image factors before overlap alignment; resume safely."""
    work = Path(work)
    name = scene['scene_name']
    observations, bands = [], []
    references = {}
    for wi, start in enumerate(starts):
        path = work / 'scale' / f'start_{start:03d}' / (name + '.anchors.npz')
        with np.load(path, allow_pickle=False) as z:
            bands.append(z['support_band'].astype(float))
            for k, row in enumerate(json.loads(str(z['metadata']))):
                image, q = row['image'], row['metric_reference']
                if image in references and not np.isclose(q, references[image], rtol=1e-5, atol=1e-6):
                    raise ValueError('Shared image depth reference changed across windows')
                references.setdefault(image, q)
                observations.append(dict(window=wi, image=image, relative=z[f'relative_{k}'].copy(),
                                         metric=z[f'metric_{k}'].copy(), metric_reference=references[image]))
    result = fit(len(starts), observations, bounded=True, shared_offset=True,
                 tau=.05, support_bands=np.asarray(bands))
    if not result['converged']:
        raise RuntimeError('Shared affine calibration did not converge')
    basis = np.asarray(scene['opencv_to_dataset'], float)
    for start, scale in zip(starts, result['scales']):
        path = work / 'windows' / f'start_{start:03d}' / name / 'omega_pose_override.npz'
        with np.load(path, allow_pickle=False) as z:
            payload = {k: z[k].copy() for k in z.files}
        anchor_path = work / 'scale' / f'start_{start:03d}' / (name + '.anchors.npz')
        with np.load(anchor_path, allow_pickle=False) as anchors:
            raw = anchors['raw_c2w'].astype(float)
            frame_ids, camera_ids = anchors['frame_ids'], anchors['camera_ids'].astype(str)
        reference = np.flatnonzero((frame_ids == frame_ids.min()) &
                                   (camera_ids == scene['reference_camera']))
        if len(reference) != 1:
            raise ValueError('Missing unique window reference camera')
        poses = np.linalg.inv(raw[reference[0]])[None] @ raw
        poses[:, :3, 3] *= scale
        payload.update(omega_c2w_rig_local=poses.astype(np.float32),
                       omega_w2c_rig_local=np.linalg.inv(poses).astype(np.float32),
                       omega_camera_to_world_rig_local=(poses @ np.linalg.inv(basis)).astype(np.float32),
                       rgb_metric_scale=np.asarray(scale),
                       metric_scale_source=np.asarray('shared_affine_metric_calibration'))
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
            temporary = Path(f.name)
        try:
            np.savez_compressed(temporary, **payload)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return dict(method='shared_affine_metric_calibration', fit=result,
                scene=name, offsets_applied_to_depth=False, GT_used=False)
