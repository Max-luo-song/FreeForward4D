#!/usr/bin/env python3
"""FreeForward4D environment check: Python/torch/torchvision/gsplat versions, CUDA
toolkit and device readiness, optional gsplat rasterizer forward/backward smoke test.

Exit code 0 = environment usable. ``--rasterizer-smoke`` additionally compiles the
gsplat CUDA kernels (JIT, needs nvcc) and verifies a tiny finite forward/backward on
the selected GPU. Heavy packages are imported lazily so the plain check stays fast.
"""
from pathlib import Path
import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "gsplat": "1.5.3",
}
# Runtime imports exercised by training; (distribution name, import name).
RUNTIME_DEPS = [
    ("numpy", "numpy"),
    ("open3d", "open3d"),
    ("jaxtyping", "jaxtyping"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("einops", "einops"),
    ("timm", "timm"),
    ("scikit-image", "skimage"),
    ("imageio", "imageio"),
    ("imageio-ffmpeg", "imageio_ffmpeg"),
    ("opencv-python-headless", "cv2"),
    ("Pillow", "PIL"),
    ("wandb", "wandb"),
    ("tensorboard", "tensorboard"),
    ("plyfile", "plyfile"),
    ("requests", "requests"),
    ("tqdm", "tqdm"),
    ("lpips", "lpips"),
]

problems = []


def report(ok, label, detail=""):
    print("  [%s] %s%s" % ("OK " if ok else "FAIL", label, (" — %s" % detail) if detail else ""))
    if not ok:
        problems.append(label)


def base_version(version):
    """'2.10.0+cu128' -> '2.10.0' for comparison against pinned base versions."""
    return version.split("+", 1)[0]


def check_python():
    print("Python:")
    print("  executable : %s" % sys.executable)
    print("  version    : %s" % sys.version.replace("\n", " "))
    v = sys.version_info
    report(v >= (3, 10), "python>=3.10", "%d.%d.%d" % (v.major, v.minor, v.micro))
    if (v.major, v.minor) != (3, 10):
        print("  [note] Python 3.10 is the recommended/reference version")


def check_packages():
    print("Packages (installed distributions):")
    for dist_name, import_name in [("torch", "torch"), ("torchvision", "torchvision"),
                                   ("gsplat", "gsplat"), *RUNTIME_DEPS]:
        try:
            version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            report(False, dist_name, "not installed (pip install -r requirements.txt)")
            continue
        detail = version
        ok = True
        expected = EXPECTED.get(dist_name)
        if expected and base_version(version) != expected:
            ok = False
            detail = "%s (expected %s)" % (version, expected)
        elif dist_name == "numpy" and not version.startswith("1."):
            ok = False
            detail = "%s (training requires NumPy >=1.26,<2)" % version
        report(ok, dist_name, detail)


def check_cuda():
    print("CUDA / GPU:")
    nvcc = shutil.which("nvcc")
    if nvcc:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        line = [ln for ln in out.splitlines() if "release" in ln]
        print("  nvcc       : %s (%s)" % (nvcc, line[0].strip() if line else "unknown version"))
    else:
        print("  nvcc       : not found (required to JIT-compile gsplat CUDA kernels)")
        problems.append("nvcc missing")
    try:
        import torch
    except ImportError:
        report(False, "torch import", "install torch first (scripts/setup.sh)")
        return None
    print("  torch      : %s (cuda %s, cudnn %s)"
          % (torch.__version__, torch.version.cuda, torch.backends.cudnn.version()))
    if not torch.cuda.is_available():
        print("  [note] torch.cuda.is_available() = False (CPU-only node or driver issue)")
        return torch
    print("  devices    : %d" % torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        mem = torch.cuda.get_device_properties(index).total_memory / 2**30
        hopper = (major, minor) == (9, 0)
        print("    [%d] %s sm%d%d %.0f GiB%s"
              % (index, name, major, minor, mem, " (Hopper/H200 target)" if hopper else ""))
    try:
        print("  nccl       : %s" % (".".join(map(str, torch.cuda.nccl.version()))))
    except Exception as exc:
        print("  nccl       : unavailable (%s)" % exc)
    return torch


def rasterizer_smoke(gpu):
    print("gsplat rasterizer smoke (GPU %d):" % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MAX_JOBS", "4")
    try:
        import torch
    except ImportError:
        report(False, "rasterizer smoke", "torch not installed; run scripts/setup.sh first")
        return

    if not torch.cuda.is_available():
        report(False, "rasterizer smoke", "CUDA not available on the selected GPU")
        return
    try:
        from gsplat import rasterization
    except ImportError as exc:
        report(False, "gsplat import", str(exc))
        return
    try:
        device = torch.device("cuda:0")
        torch.manual_seed(0)
        n = 64
        means = torch.randn(n, 3, device=device, requires_grad=True)
        quats = torch.randn(n, 4, device=device, requires_grad=True)
        scales = torch.rand(n, 3, device=device, requires_grad=True) * 0.1
        opacities = torch.rand(n, device=device, requires_grad=True)
        colors = torch.rand(n, 3, device=device, requires_grad=True)
        viewmats = torch.eye(4, device=device).unsqueeze(0)
        viewmats[0, 2, 3] = 4.0
        intrinsics = torch.tensor(
            [[150.0, 0.0, 80.0], [0.0, 150.0, 60.0], [0.0, 0.0, 1.0]], device=device
        ).unsqueeze(0)
        render, _, _ = rasterization(
            means3d=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=intrinsics, width=160, height=120,
        )
        loss = render.square().mean()
        loss.backward()
        finite = bool(torch.isfinite(render).all())
        grads_finite = all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in (means, quats, scales, opacities, colors)
        )
        report(finite and grads_finite, "tiny forward/backward finite",
               "render=%s loss=%.6f" % (tuple(render.shape), float(loss)))
    except Exception as exc:
        report(False, "rasterizer smoke", "%s: %s" % (type(exc).__name__, exc))


def main():
    parser = argparse.ArgumentParser(prog="FreeForward4D environment check", description=__doc__)
    parser.add_argument("--rasterizer-smoke", action="store_true",
                        help="Compile gsplat kernels and run a tiny finite forward/backward")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index for --rasterizer-smoke (default 0)")
    args = parser.parse_args()

    print("FreeForward4D environment check (repo: %s)" % ROOT)
    print("=" * 60)
    check_python()
    check_packages()
    check_cuda()
    if args.rasterizer_smoke:
        rasterizer_smoke(args.gpu)
    print("=" * 60)
    if problems:
        print("PROBLEMS: %s" % "; ".join(problems))
        return 1
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
