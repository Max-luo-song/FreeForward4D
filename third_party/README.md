# External model code

Run `bash scripts/setup_priors.sh` to obtain the pinned VGGT-Omega, MoGe, and
Grounded-SAM-2 sources. The external repositories retain their original licenses.
Downloaded source trees are ignored by Git, as are the checkpoints under `weights/`.

For an existing installation, `FREEFORWARD_OMEGA_REPO` and
`FREEFORWARD_MOGE_REPO` can select compatible source trees. Tracking accepts
`--gsam-root` and `--sam-checkpoint` through the preprocessing CLI.
