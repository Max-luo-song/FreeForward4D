"""Source-preserving dynamic Gaussians, local motion, and cross-time supervision.

Inspired by SLARM's per-Gaussian trajectories/time support and Envision4D's
source-exclusive rendering. This is a FreeForward adaptation, not either official model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from freeforward.models.sam_object_motion_r5 import predict_sam_canonical_motion


class LocalSourceMotionHead(nn.Module):
    def __init__(self, embed_dim, hidden=96, max_velocity=4., max_acceleration=6., max_jerk=12.):
        super().__init__()
        self.feature = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(64 + 3 + 3 + 3 + 2, hidden), nn.GELU(), nn.Linear(hidden, 9))
        self.limits = (float(max_velocity), float(max_acceleration), float(max_jerk))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, rgb, positions, coarse_velocity, subpixel):
        x = torch.cat([self.feature(features), rgb.to(features.dtype),
                       torch.tanh(positions.float() / 50).to(features.dtype),
                       (coarse_velocity / 20).to(features.dtype), subpixel.to(features.dtype)], -1)
        raw = self.head(x)
        return tuple(limit * torch.tanh(piece / limit)
                     for piece, limit in zip(raw.split(3, -1), self.limits))


def source_preserving_scene(gs_state, gs_params, image, track_ids, gs_time, timespan,
                            motion_head, detail_head, local_head, local_motion=False,
                            patch_size=8, min_observations=2, min_track_pixels=16,
                            association_max_distance=4., association_min_overlap=1):
    velocity, ids, _, diagnostics = predict_sam_canonical_motion(
        gs_state, gs_params['means'], track_ids, gs_time, timespan, motion_head,
        patch_size, min_observations, min_track_pixels, association_max_distance,
        association_min_overlap)
    b, t, v, h, w = ids.shape
    n = t * v * h * w
    flat_ids = ids.reshape(b, n)
    outputs = {key: gs_params[key].clone() for key in ['means', 'scales', 'quats', 'opacities', 'colors']}
    residual_v = torch.zeros_like(velocity)
    acceleration = torch.zeros_like(velocity)
    jerk = torch.zeros_like(velocity)
    rgb = image.permute(0, 1, 2, 4, 5, 3).reshape(b, n, 3)
    ph, pw = h // patch_size, w // patch_size
    assert gs_state.shape[1] == t * v * ph * pw
    active_count = 0
    for batch in range(b):
        idx = torch.where(flat_ids[batch] > 0)[0]
        active_count += len(idx)
        if len(idx) == 0:
            continue
        source_view = idx // (h * w)
        y, x = (idx % (h * w)) // w, idx % w
        patch = source_view * ph * pw + (y // patch_size) * pw + x // patch_size
        feature = gs_state[batch, patch]
        attrs = {key: outputs[key].reshape(b, n, -1)[batch, idx] for key in outputs}
        evidence = detail_head.encode_evidence(
            feature, rgb[batch, idx], attrs['colors'], attrs['opacities'].squeeze(-1),
            torch.zeros_like(attrs['means']), attrs['means'].new_zeros(len(idx)), .12)
        refined = detail_head.refine(evidence, attrs['means'], attrs['scales'], attrs['quats'],
                                     attrs['opacities'].squeeze(-1), attrs['colors'])
        for key, value in zip(['means', 'scales', 'quats', 'opacities', 'colors'], refined[:5]):
            if key == 'opacities':
                outputs[key].reshape(b, n)[batch, idx] = value.to(outputs[key].dtype)
            else:
                outputs[key].reshape(b, n, -1)[batch, idx] = value.to(outputs[key].dtype)
        if local_motion:
            subpixel = torch.stack([x % patch_size, y % patch_size], -1).float()
            subpixel = 2 * subpixel / max(patch_size - 1, 1) - 1
            dv, a, j = local_head(feature, rgb[batch, idx], refined[0],
                                   velocity.reshape(b, n, 3)[batch, idx], subpixel)
            residual_v.reshape(b, n, 3)[batch, idx] = dv.to(residual_v.dtype)
            acceleration.reshape(b, n, 3)[batch, idx] = a.to(acceleration.dtype)
            jerk.reshape(b, n, 3)[batch, idx] = j.to(jerk.dtype)
    outputs.update(
        source_time_seconds=(gs_time.float() * float(timespan))[..., None, None].expand(b, t, v, h, w),
        active_mask=torch.ones_like(ids, dtype=torch.bool),
        motion_velocity=velocity + residual_v,
        motion_acceleration=acceleration,
        source_jerk=jerk,
        source_residual_velocity=residual_v,
        source_preserving=True,
    )
    diagnostics.update(r9_source_dynamic_points=active_count,
                       r9_active_gaussians=b * n, r9_allocated_gaussians=b * n)
    return outputs['motion_velocity'], ids, outputs, diagnostics


@torch.no_grad()
def temporal_source_weights(ids, source_times, target_times, sigma=.35, exclude_self=False):
    """Normalize temporal support per object across observed source times.

    Background retains its existing lifespan path. Source-exclusive queries also
    remove background from the query's source time. No query mutates stored state.
    """
    if sigma <= 0:
        raise ValueError('source time sigma must be positive')
    b, t, v, h, w = ids.shape
    assert source_times.shape == (b, t, v)
    if not torch.allclose(source_times, source_times[:, :, :1].expand_as(source_times), atol=1e-5):
        raise ValueError('source temporal gate requires synchronized cameras')
    if target_times.ndim == 3:
        target_times = target_times[..., 0]
    q = target_times.shape[1]
    source = source_times[..., 0]
    dt = target_times[:, :, None] - source[:, None, :]
    same_time = dt.abs() < 1e-5
    weights = torch.ones(b, q, t * v * h * w, device=ids.device, dtype=torch.float32)
    time_index = torch.arange(t, device=ids.device).repeat_interleave(v * h * w)
    for batch in range(b):
        flat = ids[batch].reshape(-1)
        if exclude_self:
            weights[batch] *= (~same_time[batch, :, time_index]).float()
        for gid in torch.unique(flat):
            if int(gid) == 0:
                continue
            mask = flat == gid
            present = mask.reshape(t, -1).any(-1)
            allowed = present[None].expand(q, -1)
            if exclude_self:
                allowed = allowed & ~same_time[batch]
            logits = -.5 * (dt[batch] / sigma).square()
            scores = logits.masked_fill(~allowed, -1e9).softmax(-1)
            scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            weights[batch, :, mask] = scores[:, time_index[mask]]
    return weights


@torch.no_grad()
def cross_time_foreground_mask(ids, source_index):
    """Only supervise objects also observed at another source time."""
    b, t, v, h, w = ids.shape
    result = torch.zeros(b, 1, v, h, w, dtype=torch.bool, device=ids.device)
    for batch in range(b):
        other = torch.cat([ids[batch, :source_index], ids[batch, source_index + 1:]], 0)
        supported = torch.unique(other)
        supported = supported[supported > 0]
        result[batch, 0] = torch.isin(ids[batch, source_index], supported) & (ids[batch, source_index] > 0)
    return result


def cross_time_rgb_loss(model, stage2, context_inputs, step):
    """Re-render one allowed input time using other sources; no target labels."""
    from freeforward.dataset.constants import MEAN, STD
    index = step % len(context_inputs)
    context = context_inputs[index][0]
    mask = cross_time_foreground_mask(stage2['gs_params']['r6_global_track_ids'], index)
    if not mask.any():
        return stage2['gs_params']['means'].sum() * 0., mask.float().mean()
    query = stage2.copy()
    for suffix in ('camtoworlds', 'camtoworlds_global', 'intrinsics', 'time', 'instances_pose', 'instances_id'):
        query['target_' + suffix] = context['context_' + suffix]
    query['source_exclusive_query'] = True
    result = model(query, stage=3)['render_results']
    pred = result[result['rgb_key']].float()
    truth = context['context_image'].permute(0, 1, 2, 4, 5, 3).float()
    std = torch.tensor(STD, device=pred.device)
    pixel_mse = ((pred - truth) * std).square().mean(-1)
    loss = (pixel_mse * mask).sum() / mask.sum().clamp_min(1)
    return loss, mask.float().mean()
