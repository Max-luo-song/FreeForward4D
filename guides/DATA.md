# Data and prediction caches

The reconstruction loader uses an image-path/timestamp JSON schema for both Waymo and imported RGB videos. A scene record is metadata, not a 3D annotation file. With `pose_free_camera_only=true`, camera and intrinsics values are read exclusively from the predicted-camera store.

```text
DATA_ROOT/
  rgb_train.json
  rgb_val.json
  scene_list/waymo_train.txt
  scene_list/waymo_val.txt
  scene_metadata/{train,val}/SCENE.json
  datasets/waymo/{train,val}/SCENE/images_4/FRAME_CAMERA.jpg
  geometry/{train,val}/global_aligned/SCENE/omega_pose_override.npz
  geometry/{train,val}/scenes/SCENE/cameras.npz
  geometry/{train,val}/calibration/SCENE.json
  depth/contract.json
  depth/{train,val}/SCENE/camera_CAMERA/FRAME.npz
  tracks/SCENE/CAMERA/mask_data/mask_FRAME.npy
```

`datasets/waymo` is the established three-frontal-camera storage convention. Imported nuScenes RGB uses the same storage schema and camera aliases (`1,0,2`), with the scene's actual sampled timestamps.

## Download

Use authorized URLs from the original dataset publishers. Google Cloud prefixes require Google Cloud CLI and your own account:

```bash
gcloud auth login
python scripts/download_data.py --source gs://YOUR_APPROVED_PREFIX \
  --output /datasets/waymo_raw/training
```

For HTTPS archives, supply the actual download URL. A known checksum can be enforced:

```bash
python scripts/download_data.py --source 'https://YOUR_AUTHORIZED_ARCHIVE_URL' \
  --output /datasets/nuscenes --extract --sha256 EXPECTED_SHA256
```

Access approvals and license agreements happen at the official dataset portal. The downloader propagates errors and rejects archive path traversal and special files.

## Cameras

`cameras.npz` contains `frame_ids`, `camera_ids`, OpenCV camera-to-world matrices `c2w`, corresponding intrinsics `K`, and image size `[height,width]`. Cameras are in a per-scene metric frame fixed by the first reference observation. The model-facing file additionally stores the dataset-axis camera transforms and `predicted_intrinsics_freeforward`. Intrinsics refer to the reconstruction image grid.

The frontend estimates relative geometry over overlapping windows. Confidence-selected RGB images provide MoGe-2 depth observations; a joint robust fit estimates one positive scale per window and one shared additive fitting offset per physical image. Only scales affect camera translations. Offsets are not added to the MoGe-2 depth supervision. Overlapping camera observations align the calibrated windows using SE(3).

Partial batches retain compact calibration observations and raw camera matrices. The shared fit is completed before final scene export. Completion markers are written after both camera formats pass validation. A resumed fit starts again from the raw cameras, preventing repeated scale multiplication.

## Depth

Each file contains float32 `depth`, a `valid` mask, and JSON `metadata`. Values are camera-frame z-depth in meters. The metadata binds RGB content, frame/camera identity, split, and the depth contract. Training fails if the source, units, grid, or contract does not match. Invalid pixels are zero.

MoGe-2 prediction is independent of GT depth. Predictions are resized to the reconstruction grid; rendered and target depths share the same maximum-valid-target normalization in the training loss.

## Tracks

Masks contain camera-local integer track IDs, with zero as background. They are resized with nearest-neighbor sampling, then camera-specific offsets prevent accidental cross-view identity collisions. The reconstruction network associates tracks using predicted 3D centers. A tracked foreground object can be stationary.

The default text categories are car, truck, bus, motorcycle, bicycle, and pedestrian. The persistent tracking worker uses Grounding DINO and SAM 2, including forward propagation and reverse refinement. Complete masks are retained; temporary visualization outputs are removed.

## Resource planning

Choose GPU selectors explicitly with `--gpus 0,2,4` or a count with `--num-gpus 3`. Scene jobs are distributed across workers; each worker loads a model once per stage. Stages run sequentially to avoid keeping all foundation models on a GPU simultaneously.

Raw geometry outputs are stored for a bounded batch and discarded after compact calibration observations and temporary cameras are saved. `window_batch_size`, `raw_compression`, and disk reserve are configurable in `configs/preprocessing.json`. Larger batches trade storage and host RAM for startup overhead; do not increase them without checking free space.
