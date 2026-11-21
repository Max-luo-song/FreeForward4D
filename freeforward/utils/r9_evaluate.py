"""Fixed training-scene diagnostic, with context-only model inputs and separate GT loading.

The inherited SAM cache is all-video assisted. Results are explicitly labeled as
fixed-scene mechanism diagnostics, not held-out-scene or strict sparse-input NVS.
"""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from freeforward.dataset.constants import MEAN, STD
from freeforward.dataset.data_utils import to_batch_tensor, prepare_inputs_and_targets
from freeforward.utils.misc import capture_rng_state, restore_rng_state
from freeforward.window_protocol import window_splits, validate_window_start


def slice_prediction(pred, indices, total):
    result = dict(pred)
    result["render_results"] = {k: (v[:, indices] if torch.is_tensor(v) and v.ndim >= 2
                                   and v.shape[1] == total else v)
                                 for k, v in pred["render_results"].items()}
    return result


def slice_targets(target, indices, total):
    scored = {}
    for key, value in target.items():
        if key.startswith("target_"):
            if value.shape[1] != total:
                raise ValueError(f"{key} has {value.shape[1]} times, expected {total}")
            scored[key] = value[:, indices]
    return scored


@torch.no_grad()
def evaluate_r9(model, dataset_train, args, step):
    from inference import run_inference, concatenate_chunk_targets, compute_metrics
    import lpips
    rng = capture_rng_state()
    was_training = model.training
    device = next(model.parameters()).device
    output = Path(args.log_dir) / "fixed_eval" / f"step_{step:06d}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    try:
        model.eval()
        torch.cuda.empty_cache()
        perceptual = lpips.LPIPS(net="alex").to(device).eval()
        ds = copy.copy(dataset_train)
        ds.args = copy.copy(args)
        ds.args.full_window_targets = True
        # No depth, flow, dynamic GT or sky labels in any model input.
        ds.load_depth = ds.load_flow = ds.load_dynamic_mask = ds.load_ground_label = False
        ds.load_pseudo_depth = False
        ds.skip_sky_mask = True
        labels = copy.copy(ds)
        labels.load_depth = labels.load_dynamic_mask = True
        labels.skip_sky_mask = False
        eval_args = copy.copy(ds.args)
        eval_args.annotation_file = "fixed_train_scene_subset"
        eval_args.scene_id = 0
        eval_args.debug = False
        mean = torch.tensor(MEAN, device=device)
        std = torch.tensor(STD, device=device)
        fps = float(ds.annotations[0]["fps"])
        expected_frames = args.num_target_chunks * int(args.timespan * fps)
        starts = [int(x) for x in args.r9_eval_starts.split(",")]
        for start in starts:
            validate_window_start(start, expected_frames, ds.annotations[0]["num_timesteps"])
        for start in starts:
            eval_args.start_idx = start
            pred, target, render_input, _ = run_inference(model, ds, eval_args, device)
            # Evaluation labels are loaded only after the model prediction.
            raw_gt = to_batch_tensor(labels.__getitem__(0, start, return_all=True))
            pairs = prepare_inputs_and_targets(raw_gt, device, timespan=args.timespan,
                                              from_list=True, args=labels.args)
            _, scored_target = concatenate_chunk_targets(pairs)
            torch.testing.assert_close(target["target_image"], scored_target["target_image"])
            total = scored_target["target_image"].shape[1]
            assert total == expected_frames, f"expected {expected_frames} frames, got {total}"
            splits = window_splits(total)
            r = pred["render_results"]
            rgb = (r[r["rgb_key"]][0].float() * std + mean).clamp(0, 1)
            gt = (scored_target["target_image"][0].permute(0, 1, 3, 4, 2).float() * std + mean).clamp(0, 1)
            for name, indices in splits.items():
                # Context labels have four times and must not be indexed as 20 target times.
                sliced = slice_targets(scored_target, indices, total)
                assert all(v.shape[1] == len(indices) for v in sliced.values())
                metrics = compute_metrics(slice_prediction(pred, indices, total), sliced, render_input, device)
                rendered = rearrange(rgb[indices], "t v h w c -> (t v) c h w")
                truth = rearrange(gt[indices], "t v h w c -> (t v) c h w")
                distances = [perceptual(rendered[i:i+3] * 2 - 1, truth[i:i+3] * 2 - 1).flatten()
                             for i in range(0, len(rendered), 3)]
                metrics["lpips"] = float(torch.cat(distances).mean())
                records.append({"start": start, "split": name, "images": len(rendered), "metrics": metrics})
            # Identical frame/view crops each evaluation, PNG and no video compression.
            preview_frames = (2, 12, 18) if total == 20 else (2, total // 2, total - 2)
            front_slot = 0 if rgb.shape[1] == 1 else 1
            for frame in preview_frames:
                comparison = torch.cat([gt[frame, front_slot], rgb[frame, front_slot],
                                        (rgb[frame, front_slot] - gt[frame, front_slot]).abs()], dim=1)
                Image.fromarray((comparison.cpu().numpy() * 255).round().astype(np.uint8)).save(
                    output / f"start{start:03d}_frame{frame:02d}_front.png")
            # Paired time sequences for visual motion QA; metrics still use tensors.
            import imageio.v2 as imageio
            frames = []
            for frame in range(total):
                top = torch.cat([gt[frame, camera] for camera in range(gt.shape[1])], dim=1)
                bottom = torch.cat([rgb[frame, camera] for camera in range(rgb.shape[1])], dim=1)
                frames.append((torch.cat([top, bottom], dim=0).cpu().numpy() * 255).round().astype(np.uint8))
            imageio.mimsave(output / f"start{start:03d}_comparison.mp4", frames, fps=fps, macro_block_size=1)
            del pred, target, render_input, raw_gt, pairs, scored_target, r, rgb, gt
        summary = {}
        for split in ("context", "interpolation", "extrapolation"):
            rows = [r for r in records if r["split"] == split]
            summary[split] = {key: float(np.average([r["metrics"][key] for r in rows],
                                                  weights=[r["images"] for r in rows]))
                              for key in rows[0]["metrics"]}
        pseudo = bool(getattr(args, "enable_pseudo_depth_loss", False))
        supervision = "RGB_MoGe_pseudodepth_training" if pseudo else ("RGBDtraining" if args.enable_depth_loss else "RGBtraining")
        camera_source = getattr(args, "camera_pose_source", "GT")
        result = {"step": step, "protocol": f"fixed_scene621_allvideo_SAM_assisted_{camera_source}camera_" + supervision,
                  "camera_pose_source": camera_source,
                  "input_size": list(args.input_size), "num_cameras": args.num_max_cameras,
                  "rgb_directory": getattr(args, "waymo_rgb_directory", "images_4"),
                  "window_frames": expected_frames, "window_seconds": expected_frames / fps,
                  "context_frames": splits["context"], "single_persistent_scene": True,
                  "depth_loss_coeff": args.depth_loss_coeff if args.enable_depth_loss or pseudo else 0.0,
                  "depth_supervision_source": "MoGe" if pseudo else ("GT" if args.enable_depth_loss else "none"),
                  "evaluation_depth_source": "GT",
                  "pseudo_depth_contract_id": getattr(args, "pseudo_depth_contract_id", None),
                  "depth_loss_normalization": getattr(args, "depth_loss_normalization", "target_max"),
                  "held_out_scene": False, "sam_context_isolated": False,
                  "gt_instance_inputs": not getattr(args, "pose_free_camera_only", False),
                  "depth_supervision_only": getattr(args, "depth_supervision_only", False),
                  "source_variant": getattr(args, "r9_source_variant", "r9"),
                  "training_perceptual_weight": args.perceptual_weight if args.enable_perceptual_loss else 0.,
                  "seconds": time.perf_counter() - started, "summary": summary, "windows": records}
        (output / "metrics.json").write_text(json.dumps(result, indent=2))
        with (Path(args.log_dir) / "evaluation_history.jsonl").open("a") as stream:
            stream.write(json.dumps({k: v for k, v in result.items() if k != "windows"}) + "\n")
        return summary
    finally:
        model.train(was_training)
        restore_rng_state(rng)
        torch.cuda.empty_cache()
