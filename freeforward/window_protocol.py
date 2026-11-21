"""Shared short/long-window contracts; no change to model parameters."""
import torch


def window_splits(total, stride=5):
    if total < 2 * stride or total % stride:
        raise ValueError(f"window must contain complete context intervals: {total=}, {stride=}")
    context = list(range(0, total, stride))
    return {
        "context": context,
        "interpolation": [i for i in range(context[-1]) if i % stride],
        "extrapolation": list(range(context[-1] + 1, total)),
    }


def validate_window_start(start, total, scene_frames):
    if start < 0 or start + total > scene_frames:
        raise ValueError(f"requested window [{start}, {start + total}) exceeds scene length {scene_frames}")


def slice_render_targets(data, start, end):
    """Slice every target-time tensor, preserving all source state by reference."""
    total = data["target_camtoworlds"].shape[1]
    if not 0 <= start < end <= total:
        raise ValueError(f"invalid render slice [{start}, {end}) for {total}")
    result = data.copy()
    for key, value in data.items():
        if key.startswith("target_") and torch.is_tensor(value) and value.ndim >= 2:
            if value.shape[1] != total:
                raise ValueError(f"unaligned target tensor {key}: {value.shape[1]} != {total}")
            result[key] = value[:, start:end]
    return result
