# Training on one 8-GPU H200 node

## Recipes

| Stage | Context observations | Window | Global batch | Updates | Learning rate |
| --- | --- | --- | --- | --- | --- |
| Waymo 2s | 4, spaced approximately 0.5s apart | 2s | 64 | 100,000 | 1e-4 |
| Waymo 8s | 16, spaced approximately 0.5s apart | 8s | 32 | 50,000 | 3e-5 |

The JSON files preserve the model architecture, motion heads, predicted-camera settings, and depth supervision. The current model uses `r9_source_variant=local`; historical config fields remain for checkpoint compatibility. The public network name is `FreeForward-B/8`.

The code processes one observation per update (`num_context_timesteps=1`) and uses 4/16 context chunks per 2s/8s window (`num_target_chunks`). A 2s clip therefore has four context observations.

## Launch

Activate the reconstruction environment, prepare the RGB-derived conditions, and export the path roots described in the README. Check the rasterizer before training:

```bash
python scripts/check_environment.py --rasterizer-smoke
bash scripts/train.sh --stage 2s --nproc-per-node 8 --dry-run
bash scripts/train.sh --stage 2s --nproc-per-node 8
```

The 2s recipe is `4 sequences/GPU × 2 accumulation × 8 GPUs = 64`. The 8s recipe is `2 × 2 × 8 = 32`. The launcher prints the resolved recipe and uses the learning rate in the config without automatic rescaling.

```bash
# Initialize 8s training from the trained 2s network.
bash scripts/train.sh --stage 8s --nproc-per-node 8 \
  --init-from "$FREEFORWARD_RUN_ROOT/FreeForward4D/waymo_2s/checkpoints/ckpt_099999.pth"
```

## Start and resume

- Fresh 2s training does not require a reconstruction checkpoint. Download the frozen visual-prior and perceptual weights first.
- `--init-from FILE` loads strictly matching model weights and starts a new run.
- `--resume-from FILE` restores optimizer, scheduler, RNG, sampler progress, and update count. The launcher checks GPU count and per-GPU batch compatibility.
- `--auto-resume` searches only this experiment's checkpoint directory.

```bash
bash scripts/train.sh --stage 2s --auto-resume
```

## Environment checks

`--dry-run` prints the command without loading CUDA models or requiring prepared data. A real launch validates the depth contract and H200 devices. The public launcher uses a single node; smaller GPU counts and other GPU types are available through explicit smoke-test overrides.

Ensure `ninja`, a C++ compiler, `nvcc`, and the active environment's `bin/` directory are on `PATH`. Build rasterizer extensions for the local GPU architecture. The launcher raises the file-descriptor limit, constrains CPU thread pools, and starts workers using `torch.distributed.run --standalone`.
