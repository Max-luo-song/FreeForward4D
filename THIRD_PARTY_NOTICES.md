# Third-party sources and attribution

FreeForward4D adds RGB-derived geometric conditions, cross-view track association,
hierarchical object/Gaussian motion, temporal weighting, and release tooling to an
existing feed-forward reconstruction framework. Renaming the public package does
not remove the original authors' rights or contributions.

## Reconstruction framework

- [UFO](https://github.com/xiaomi-research/ufo), Xiaomi Corporation and the UFO authors.
  The backbone and portions of the training, dataset, and rendering framework are
  adapted from this source. The repository retains its CC BY-NC 4.0 license and
  existing per-file Xiaomi/Apache notices. Modifications include RGB-only predicted
  geometry/tracks, local motion heads, temporal rendering, training support, and
  public namespace/path cleanup.
- [STORM / GaussianSTORM](https://github.com/NVlabs/GaussianSTORM), for the
  feed-forward dynamic-Gaussian formulation and inherited framework components.
- [timm](https://github.com/huggingface/pytorch-image-models), for Vision Transformer
  layer designs identified in source comments.
- [gsplat](https://github.com/nerfstudio-project/gsplat), for differentiable Gaussian
  rasterization. It is installed as a separate dependency.
- [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), for
  LPIPS; the adapted implementation identifies its source in the file header.

## Visual priors and geometric processing

- [VGGT-Omega](https://github.com/facebookresearch/vggt-omega), for RGB camera and
  relative-geometry predictions. Its source and gated checkpoints retain the
  original publisher's license and access requirements.
- [MoGe / MoGe-2](https://github.com/microsoft/MoGe), for monocular metric geometry.
- The frozen `utils3d` compatibility source under `freeforward/vendor/utils3d`
  retains its original MIT license in `freeforward/vendor/UTILS3D_LICENSE`.
  It supplies the geometry API used by the pinned MoGe-2 frontend, without an
  installation dependency on the original, now unavailable Git URL.
- [Geometrically-Constrained Agent for Spatial Reasoning](https://arxiv.org/abs/2511.22659),
  for prior cross-model metric anchoring. FreeForward4D's shared affine fit uses
  window scales and per-image offsets with regularization and bounded support.
- [SLARM](https://github.com/kevinchiu19/SLARM), for related higher-order Gaussian
  motion representations.
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2),
  [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), and
  [SAM 2](https://github.com/facebookresearch/sam2), for RGB detection, segmentation,
  and persistent tracking. The tracking worker adapts their published tracking
  routine. Their code and checkpoints are separate dependencies under their own
  licenses.

## Data and media

- [Waymo Open Dataset](https://waymo.com/open/) and
  [nuScenes](https://www.nuscenes.org/) remain subject to their original data terms.
  Users obtain their own access. The repository does not redistribute raw datasets.
  Official nuScenes split names are copied from [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit),
  Copyright 2018–2020 Motional, Inc., under Apache License 2.0.
- Project-page examples are rendered model outputs derived from the project's
  existing experiments. Media preparation changes presentation encoding/crops only.
  RGB reference rows are not presented as model predictions.

Source headers are preserved. External code downloaded into `third_party/` keeps
its original license files; these directories and all downloaded weights are
excluded from this repository's Git history.
