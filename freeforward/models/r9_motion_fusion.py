"""Experimental R9 object acceleration and conservative spatial association."""
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectAccelerationHead(nn.Module):
    """Encode the observed sequence, retaining its second temporal moment."""
    def __init__(self, embed_dim, hidden_dim=128, max_acceleration=6.0):
        super().__init__()
        self.max_acceleration = float(max_acceleration)
        self.encoder = nn.Sequential(nn.LayerNorm(embed_dim + 3),
                                     nn.Linear(embed_dim + 3, hidden_dim), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(3 * hidden_dim),
                                  nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, times, support):
        centered = times.float() - times.float().mean()
        span = (times.max() - times.min()).float().clamp_min(0.5)
        tau = centered / span
        extra = torch.stack([tau, tau.square(), torch.log1p(support.float()) / 5], -1)
        h = self.encoder(torch.cat([features, extra.to(features.dtype)], -1))
        moment2 = tau.square() - tau.square().mean()
        descriptor = torch.cat([h.mean(0), (h * tau[:, None]).mean(0),
                                (h * moment2[:, None]).mean(0)], -1)
        return self.max_acceleration * torch.tanh(self.head(descriptor) / self.max_acceleration)


def predict_object_acceleration(gs_state, global_ids, gs_time, timespan, head, patch_size=8):
    b, t, v, h, w = global_ids.shape
    features = gs_state.reshape(b, t, v, h // patch_size, w // patch_size, -1)
    seconds = gs_time.float() * float(timespan)
    acceleration = gs_state.new_zeros(b, t, v, h, w, 3)
    for batch in range(b):
        for gid in torch.unique(global_ids[batch]):
            if int(gid) == 0:
                continue
            mask = global_ids[batch] == gid
            cov = F.avg_pool2d(mask.float().reshape(t * v, 1, h, w), patch_size, patch_size)
            cov = cov.reshape(t, v, h // patch_size, w // patch_size)
            support = cov.sum((1, 2, 3))
            valid = support > 0
            # Two observations cannot identify acceleration: retain constant motion.
            if int(valid.sum()) < 3:
                continue
            pooled = (features[batch] * cov[..., None]).sum((1, 2, 3)) / support.clamp_min(1e-6)[:, None]
            view_support = cov.sum((2, 3))
            times = (seconds[batch] * view_support).sum(-1) / support.clamp_min(1e-6)
            coeff = head(pooled[valid], times[valid], support[valid])
            acceleration[batch][mask] = coeff.to(acceleration.dtype)
    return acceleration


@torch.no_grad()
def conservative_groups(xyz, features, rgb, times, voxel_size, radius_factor=1.5,
                        min_cosine=0.2, block_size=4096):
    """Match to reference-frame anchors in 27 neighboring cells, or retain self.

    Discrete association is deliberately detached. The original R9 learned
    confidence/refinement remains responsible for differentiable within-group fusion.
    Rejected points form singletons; no new Gaussian slots are allocated.
    """
    xyz = xyz.float()
    cells = torch.floor(xyz / float(voxel_size)).long()
    ref_time = torch.unique(times)[(torch.unique(times) - times.mean()).abs().argmin()]
    ref_indices = torch.where(times == ref_time)[0]
    ref_cells, ref_inverse = torch.unique(cells[ref_indices], dim=0, return_inverse=True)
    count = ref_cells.shape[0]
    representatives = torch.full((count,), xyz.shape[0], device=xyz.device, dtype=torch.long)
    representatives.scatter_reduce_(0, ref_inverse, ref_indices, reduce="amin", include_self=True)
    lo = cells.min(0).values - 1
    extent = cells.max(0).values - lo + 2
    # Guard key encoding against silent overflow for pathological geometries.
    if float(extent.double().prod()) >= 2 ** 62:
        raise RuntimeError("candidate grid exceeds int64 key capacity")
    def key(c):
        c = c - lo
        return (c[..., 0] * extent[1] + c[..., 1]) * extent[2] + c[..., 2]
    sorted_keys, permutation = torch.sort(key(ref_cells))
    offsets = torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), device=xyz.device)
    norm_features = F.normalize(features.float(), dim=-1)
    assignments = torch.full((len(xyz),), -1, device=xyz.device, dtype=torch.long)
    radius = float(voxel_size) * float(radius_factor)
    for start in range(0, len(xyz), block_size):
        end = min(start + block_size, len(xyz))
        query_keys = key(cells[start:end, None] + offsets[None])
        locations = torch.searchsorted(sorted_keys, query_keys)
        safe = locations.clamp(max=count - 1)
        exists = (locations < count) & (sorted_keys[safe] == query_keys)
        candidate_ids = permutation[safe]
        point_ids = representatives[candidate_ids]
        distance2 = (xyz[start:end, None] - xyz[point_ids]).square().sum(-1)
        similarity = (norm_features[start:end, None] * norm_features[point_ids]).sum(-1)
        color_error = (rgb[start:end, None].float() - rgb[point_ids].float()).square().mean(-1)
        valid = exists & (distance2 <= radius ** 2) & (similarity >= min_cosine)
        cost = distance2 / (radius ** 2) + 0.5 * (1 - similarity) + 0.25 * color_error
        cost = cost.masked_fill(~valid, float("inf"))
        best_cost, best = cost.min(-1)
        selected = candidate_ids.gather(1, best[:, None]).squeeze(1)
        assignments[start:end] = torch.where(torch.isfinite(best_cost), selected, -1)
    # Anchors remain anchored, preventing cyclic associations.
    assignments[ref_indices] = ref_inverse
    rejected = assignments < 0
    assignments[rejected] = count + torch.arange(int(rejected.sum()), device=xyz.device)
    _, inverse = torch.unique(assignments, sorted=True, return_inverse=True)
    return inverse, {"rejected": rejected.sum(), "candidates": len(xyz) - len(ref_indices),
                     "anchors": count}
