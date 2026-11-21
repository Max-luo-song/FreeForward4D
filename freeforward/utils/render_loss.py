"""Recompute render/loss activations without truncating the recurrent graph."""
import torch
from torch.utils.checkpoint import checkpoint
from freeforward.utils.losses import compute_loss


def final_scene_render_loss(model, data, targets, args, rgb_loss):
    def run(render_data, render_targets):
        # Rendering adds affine/sky metadata; never mutate checkpoint inputs.
        local = render_data.copy()
        local['gs_params'] = render_data['gs_params'].copy()
        prediction = model(local, stage=3)
        losses = compute_loss(prediction, render_targets, args, rgb_loss)
        preview = {k: v.detach() if torch.is_tensor(v) else v
                   for k, v in prediction['render_results'].items()}
        return losses, preview

    if getattr(args, 'checkpoint_final_scene_render', False) and torch.is_grad_enabled():
        # Non-reentrant checkpointing preserves nested tensor inputs and does not
        # run the first render under no_grad (important for AMP weight caches).
        losses, preview = checkpoint(run, data, targets, use_reentrant=False)
    else:
        losses, preview = run(data, targets)
    prediction = data.copy()
    prediction['render_results'] = preview
    return prediction, losses
