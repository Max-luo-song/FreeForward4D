#!/usr/bin/env python3
"""FreeForward4D training launcher for one 8-GPU H200 node.

Resolves per-GPU batch size and gradient accumulation, checks devices and the
depth contract, and starts torch.distributed.run in standalone mode.

    bash scripts/train.sh --stage 2s --dry-run
    bash scripts/train.sh --stage 2s

Arguments after ``--`` are forwarded to train.py. Small-device smoke runs use
explicit overrides; the public training recipes target eight H200 GPUs.
"""
from pathlib import Path
import argparse
import glob
import io
import json
import os
import pickle
import re
import resource
import shlex
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]

# World size the shipped configs are authored against (8x H200 reference run).
REFERENCE_WORLD_SIZE = 8
# Formal training requires at least this many ranks; smaller worlds are smoke runs.
MIN_FORMAL_WORLD_SIZE = 8
# Hopper (H200) compute capability.
HOPPER_CAPABILITY = (9, 0)
TORCH_CUDA_ARCH = "9.0"

ENV_DATA_ROOT = "FREEFORWARD_DATA_ROOT"
ENV_WEIGHTS_ROOT = "FREEFORWARD_WEIGHTS_ROOT"
ENV_RUN_ROOT = "FREEFORWARD_RUN_ROOT"

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


class LaunchError(RuntimeError):
    """Fatal, user-actionable launch problem."""


# ----------------------------------------------------------------------------
# Config reading (mirrors freeforward.utils.config.load_config macro expansion;
# stdlib-only copy so --dry-run never imports the package or torch).
# ----------------------------------------------------------------------------
def read_config(path, *, allow_unresolved):
    with open(path, "r") as handle:
        config = json.load(handle)
    unresolved = []

    def expand(value):
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, str):
            result = os.path.expanduser(os.path.expandvars(value))
            missing = re.findall(r"\$\{([^}]+)\}", result)
            for name in missing:
                if allow_unresolved:
                    result = result.replace("${%s}" % name, "<%s>" % name)
                    unresolved.append(name)
                else:
                    raise LaunchError(
                        "Config %s needs environment variable %s. Export it or pass "
                        "--data-root/--weights-root/--run-root." % (path, name)
                    )
            return result
        return value

    return expand(config), sorted(set(unresolved))


# ----------------------------------------------------------------------------
# Parallel recipe: resolve batches for the selected local GPUs.
# ----------------------------------------------------------------------------
def solve_parallel_recipe(global_batch, world_size, base_batch):
    """Return (per_rank_batch, accumulation). Prefer base_batch, halve it for
    larger worlds before touching accumulation below 1."""
    if global_batch < 1 or world_size < 1 or base_batch < 1:
        raise LaunchError(
            "Invalid recipe numbers: global_batch=%d world_size=%d base_batch=%d"
            % (global_batch, world_size, base_batch)
        )
    if global_batch % world_size != 0:
        raise LaunchError(
            "Global batch %d is not divisible by world size %d; adjust "
            "--global-batch-size or the GPU count." % (global_batch, world_size)
        )
    batch = base_batch
    while batch >= 1:
        if global_batch % (world_size * batch) == 0:
            return batch, global_batch // (world_size * batch)
        batch //= 2
    raise LaunchError(
        "Global batch %d cannot be factored over world size %d with per-rank "
        "batch <= %d (world size too large for this global batch)."
        % (global_batch, world_size, base_batch)
    )


# ----------------------------------------------------------------------------
# Checkpoint header introspection without torch (torch.save zip container:
# tensor bytes live in separate archive members; data.pkl carries the pickled
# dict of argparse.Namespace + rng_states + persistent tensor references).
# ----------------------------------------------------------------------------
class _PickleStub:
    """Stand-in for torch/numpy globals; tolerates REDUCE/BUILD/APPEND."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def append(self, item):
        pass

    def extend(self, items):
        pass

    def __setitem__(self, key, value):
        pass


class _HeaderUnpickler(pickle.Unpickler):
    # Only plain-data stdlib globals are resolved for real; every other global
    # (torch rebuild helpers, numpy reconstructors, anything unexpected) becomes
    # an inert stub so reading a header can never execute unexpected callables.
    _ALLOWED_GLOBALS = {
        ('argparse', 'Namespace'), ('types', 'SimpleNamespace'),
        ('collections', 'OrderedDict'),
        *{('builtins', name) for name in ('set', 'frozenset', 'slice', 'complex',
                                         'bytearray', 'bytes', 'object')},
    }

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED_GLOBALS:
            return super().find_class(module, name)
        return _PickleStub

    def persistent_load(self, pid):
        return None  # tensor storages: metadata only, never loaded


def read_checkpoint_header(path, *, allow_torch_fallback):
    """Return dict(args=Namespace|None, world_size|None, latest_step|None).

    Raises LaunchError when the header cannot be read at all."""
    error = None
    try:
        if not zipfile.is_zipfile(path):
            raise LaunchError("not a torch zip checkpoint (legacy pickle format)")
        with zipfile.ZipFile(path) as archive:
            member = next(n for n in archive.namelist() if n.endswith("data.pkl"))
            blob = archive.read(member)
        payload = _HeaderUnpickler(io.BytesIO(blob)).load()
    except Exception as exc:  # unreadable header -> optional torch fallback
        error = exc
        payload = None
    if payload is None:
        if not allow_torch_fallback:
            raise LaunchError(
                "Cannot read checkpoint header of %s without torch (%s)." % (path, error)
            )
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc2:
            raise LaunchError("Cannot read checkpoint %s: %s" % (path, exc2))
    if not isinstance(payload, dict) or "model" not in payload:
        raise LaunchError(
            "%s is not a full training checkpoint (no 'model'/'optimizer' state); "
            "it cannot be used with --resume-from. Use --init-from for weights-only init."
            % path
        )
    saved_args = payload.get("args")
    rng_states = payload.get("rng_states")
    world = len(rng_states) if isinstance(rng_states, list) else None
    return {
        "args": saved_args,
        "world_size": world,
        "latest_step": payload.get("latest_step"),
    }


def verify_resume_compatible(path, recipe, *, allow_torch_fallback=True):
    """Full resume restores per-rank RNG/sampler state: world size and per-rank
    batch/accumulation must match the checkpoint or the data layout changes."""
    header = read_checkpoint_header(path, allow_torch_fallback=allow_torch_fallback)
    saved_args = header["args"]
    saved_world = header["world_size"]
    saved_batch = getattr(saved_args, "batch_size", None)
    saved_accum = getattr(saved_args, "gradient_accumulation_steps", None)
    problems = []
    if saved_world is not None and saved_world != recipe["world_size"]:
        problems.append("world_size %d (checkpoint) != %d (this launch)" % (saved_world, recipe["world_size"]))
    if saved_batch is not None and saved_batch != recipe["batch_size"]:
        problems.append("per-rank batch_size %d != %d" % (saved_batch, recipe["batch_size"]))
    if saved_accum is not None and saved_accum != recipe["gradient_accumulation_steps"]:
        problems.append("gradient_accumulation_steps %d != %d" % (saved_accum, recipe["gradient_accumulation_steps"]))
    if problems:
        raise LaunchError(
            "Full resume from %s is incompatible with the resolved recipe: %s.\n"
            "A full checkpoint also stores per-rank RNG/sampler/optimizer state, so "
            "resuming at a different world size or per-rank batch would silently "
            "change the data order. To reuse the weights at this scale, restart with "
            "weights-only init instead:\n"
            "    --init-from %s\n"
            "(optimizer, loss scaler, sampler and RNG start fresh; load is strict)."
            % (path, "; ".join(problems), path)
        )
    return header


# ----------------------------------------------------------------------------
# Pseudo-depth contract id
# ----------------------------------------------------------------------------
def resolve_pseudo_depth_contract(config, *, dry_run):
    """Return contract id to pin via CLI, or None when the config pins it already."""
    if not config.get("enable_pseudo_depth_loss"):
        return None
    root = config.get("pseudo_depth_root")
    pinned = config.get("pseudo_depth_contract_id")
    contract_path = os.path.join(root, "contract.json") if root else None
    contract = None
    if contract_path and os.path.isfile(contract_path):
        with open(contract_path, "r") as handle:
            contract = json.load(handle)
    if contract is None:
        if dry_run:
            print("NOTE: pseudo-depth contract %s not found; id unresolved (dry-run only)."
                  % (contract_path or "<unset pseudo_depth_root>"))
            return pinned
        raise LaunchError(
            "Pseudo-depth supervision is enabled but %s is missing. Generate the "
            "pseudo-depth cache under ${%s}/depth first (freeforward.preprocessing), "
            "or disable enable_pseudo_depth_loss."
            % (contract_path or "<unset pseudo_depth_root>", ENV_DATA_ROOT)
        )
    contract_id = contract.get("id")
    if not contract_id:
        raise LaunchError("%s has no 'id' field." % contract_path)
    if pinned and pinned != contract_id:
        raise LaunchError(
            "Config pins pseudo_depth_contract_id=%r but %s declares id=%r."
            % (pinned, contract_path, contract_id)
        )
    return contract_id


# ----------------------------------------------------------------------------
# GPU validation (torch imported lazily; never on --dry-run)
# ----------------------------------------------------------------------------
def validate_gpus(nproc_per_node, *, allow_other_gpu):
    import torch

    if not torch.cuda.is_available():
        raise LaunchError(
            "CUDA is not available on this node. Formal FreeForward4D training needs "
            "NVIDIA Hopper GPUs; use --dry-run for CPU-side plan checks."
        )
    count = torch.cuda.device_count()
    if nproc_per_node > count:
        raise LaunchError(
            "--nproc-per-node=%d exceeds the %d visible CUDA devices." % (nproc_per_node, count)
        )
    bad = []
    names = []
    for index in range(count):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        names.append("%s (sm%d%d)" % (name, capability[0], capability[1]))
        if "H200" not in name and capability != HOPPER_CAPABILITY:
            bad.append("device %d: %s capability %s" % (index, name, capability))
    if bad and not allow_other_gpu:
        raise LaunchError(
            "Non-Hopper GPUs detected:\n  %s\nFormal training targets NVIDIA H200 "
            "(sm90). Pass --allow-other-gpu only for smoke/debug runs."
            % "\n  ".join(bad)
        )
    return count, names


def raise_nofile_limit():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else max(soft, 1048576)
        target = min(target, 1048576) if target != resource.RLIM_INFINITY else 1048576
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError) as exc:
        print("WARNING: could not raise RLIMIT_NOFILE: %s" % exc, file=sys.stderr)


def atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def own_run_latest_checkpoint(ckpt_dir):
    """Same selection rule as freeforward.utils.misc.load_model auto-resume:
    latest mtime *.pth in this run's own checkpoint dir, excluding 'latest'."""
    checkpoints = [c for c in glob.glob(os.path.join(ckpt_dir, "*.pth")) if "latest" not in c]
    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1] if checkpoints else None


def get_args_parser():
    parser = argparse.ArgumentParser(
        prog="FreeForward4D train launcher",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stage", choices=["2s", "8s"],
                        help="Use configs/waymo_<stage>.json from this repository")
    source.add_argument("--config", type=str, help="Path to a FreeForward4D JSON config")

    cluster = parser.add_argument_group("local GPUs")
    cluster.add_argument("--nproc-per-node", type=int, default=None,
                         help="Default: number of visible CUDA devices")

    roots = parser.add_argument_group("path roots (exported before the config is read)")
    roots.add_argument("--data-root", type=str, default=os.environ.get(ENV_DATA_ROOT),
                       help="Sets $%s" % ENV_DATA_ROOT)
    roots.add_argument("--weights-root", type=str, default=os.environ.get(ENV_WEIGHTS_ROOT),
                       help="Sets $%s" % ENV_WEIGHTS_ROOT)
    roots.add_argument("--run-root", type=str, default=os.environ.get(ENV_RUN_ROOT),
                       help="Sets $%s" % ENV_RUN_ROOT)

    recipe = parser.add_argument_group("recipe overrides")
    recipe.add_argument("--global-batch-size", type=int, default=None,
                        help="Sequences per optimizer step across all ranks "
                             "(default: config batch_size x gradient_accumulation_steps x %d)"
                        % REFERENCE_WORLD_SIZE)
    recipe.add_argument("--num-iterations", type=int, default=None,
                        help="Optimizer updates (default: config num_iterations)")
    recipe.add_argument("--init-from", type=str, default=None,
                        help="Weights-only initialization checkpoint (train.py --load_from)")
    recipe.add_argument("--resume-from", type=str, default=None,
                        help="Full-state resume checkpoint (train.py --resume_from)")
    recipe.add_argument("--exp-name", type=str, default=None)

    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--auto-resume", dest="auto_resume", action="store_true",
                        default=None, help="Force resume from this run's own latest checkpoint")
    resume.add_argument("--no-auto-resume", dest="auto_resume", action="store_false",
                        default=None, help="Disable auto-resume even if the config enables it")

    smoke = parser.add_argument_group("smoke-run escape hatches (not for formal training)")
    smoke.add_argument("--allow-other-gpu", action="store_true",
                       help="Skip the H200/Hopper (sm90) device check")
    smoke.add_argument("--allow-small-world", action="store_true",
                       help="Allow world sizes below %d" % MIN_FORMAL_WORLD_SIZE)

    parser.add_argument("--dry-run", action="store_true",
                        help="Print the full command and resolved recipe; no GPU/data/torch needed")
    return parser


def split_passthrough(argv):
    if "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1:]
    return argv, []


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    own_argv, passthrough = split_passthrough(argv)
    parser = get_args_parser()
    args = parser.parse_args(own_argv)

    # -- Path roots: export BEFORE the config is read (${...} macros) ----------
    roots = {
        ENV_DATA_ROOT: args.data_root,
        ENV_WEIGHTS_ROOT: args.weights_root,
        ENV_RUN_ROOT: args.run_root,
    }
    for name, value in roots.items():
        if value:
            os.environ[name] = os.path.abspath(os.path.expanduser(value))
    missing_roots = [name for name, value in roots.items() if not os.environ.get(name)]
    if missing_roots and not args.dry_run:
        raise LaunchError(
            "Set these environment variables (or pass the flags) before launching: %s"
            % ", ".join("%s (--%s)" % (n, n.lower().replace("_", "-").replace("freeforward-", ""))
                        for n in missing_roots)
        )

    # -- Local GPU count ---------------------------------------------------------
    if args.nproc_per_node is not None:
        nproc_per_node = args.nproc_per_node
    elif os.environ.get("NPROC_PER_NODE"):
        nproc_per_node = _env_int("NPROC_PER_NODE", 1)
    elif args.dry_run:
        nproc_per_node = 8  # assume a full H200 node for planning purposes
    else:
        try:
            import torch
        except ImportError:
            raise LaunchError(
                "torch is not installed in %s; run scripts/setup.sh first "
                "(or pass --nproc-per-node and use --dry-run for CPU-side checks)."
                % sys.executable
            )
        nproc_per_node = torch.cuda.device_count()
        if nproc_per_node == 0:
            raise LaunchError(
                "No visible CUDA devices; pass --nproc-per-node explicitly or set "
                "CUDA_VISIBLE_DEVICES (formal training requires H200 GPUs)."
            )
    if not 1 <= nproc_per_node <= REFERENCE_WORLD_SIZE:
        raise LaunchError("The public launcher supports 1–8 local GPUs; formal training uses 8 H200 GPUs.")
    world_size = nproc_per_node

    # -- Config ------------------------------------------------------------------
    config_path = Path(args.config) if args.config else ROOT / "configs" / ("waymo_%s.json" % args.stage)
    config_path = config_path.expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise LaunchError("Config not found: %s" % config_path)
    config, unresolved_vars = read_config(str(config_path), allow_unresolved=args.dry_run)
    stage = config.get("training_stage") or args.stage or "custom"
    if args.stage and stage and stage != args.stage:
        print("WARNING: config training_stage=%r differs from --stage %r" % (stage, args.stage),
              file=sys.stderr)

    # -- Numeric recipe ----------------------------------------------------------
    for required_key in ("batch_size", "num_iterations"):
        if required_key not in config:
            raise LaunchError("Config %s is missing required key %r." % (config_path, required_key))
    base_batch = int(config["batch_size"])
    config_accum = int(config.get("gradient_accumulation_steps", 1))
    global_batch = args.global_batch_size or base_batch * config_accum * REFERENCE_WORLD_SIZE
    num_iterations = args.num_iterations or int(config["num_iterations"])
    batch_size, accumulation = solve_parallel_recipe(global_batch, world_size, base_batch)

    if world_size < MIN_FORMAL_WORLD_SIZE and not args.allow_small_world:
        raise LaunchError(
            "World size %d < %d. Formal FreeForward4D training uses 8 H200 GPUs; "
            "pass --allow-small-world only for smoke runs." % (world_size, MIN_FORMAL_WORLD_SIZE)
        )

    # -- Run directories (mirror train.py main()) --------------------------------
    exp_name = args.exp_name or config.get("exp_name") or str(config.get("model", "FreeForward-B/8")).replace("/", "-")
    project = config.get("project", "debug")
    output_dir = config.get("output_dir", "./output")
    log_dir = os.path.join(output_dir, project, exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")

    # -- Pseudo-depth contract -----------------------------------------------------
    contract_id = resolve_pseudo_depth_contract(config, dry_run=args.dry_run)

    # -- Init / resume -------------------------------------------------------------
    init_from = args.init_from or config.get("load_from")
    resume_from = args.resume_from or config.get("resume_from")
    auto_resume = args.auto_resume if args.auto_resume is not None else bool(config.get("auto_resume", False))
    resume_target = None
    if resume_from:
        resume_target = os.path.expanduser(os.path.expandvars(resume_from))
    elif auto_resume:
        own = own_run_latest_checkpoint(ckpt_dir) if "<" not in ckpt_dir else None
        if own:
            resume_target = own
    elif args.auto_resume is False and config.get("auto_resume"):
        raise LaunchError(
            "The config enables auto_resume and this launcher cannot unset store_true "
            "flags from the CLI. Point --exp-name at a fresh run or edit the config."
        )

    recipe = {
        "stage": stage,
        "world_size": world_size,
        "nnodes": 1,
        "nproc_per_node": nproc_per_node,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": global_batch,
        "num_iterations": num_iterations,
    }
    resume_header = None
    if resume_target and os.path.isfile(resume_target):
        try:
            # Same compatibility check in dry-run (stdlib-only) and real launches.
            resume_header = verify_resume_compatible(
                resume_target, recipe, allow_torch_fallback=not args.dry_run)
        except LaunchError as exc:
            if not args.dry_run:
                raise
            print("WARNING: resume compatibility not enforced in dry-run: %s" % exc,
                  file=sys.stderr)
        if resume_header and resume_header["latest_step"] is not None and \
                resume_header["latest_step"] + 1 >= num_iterations:
            print("WARNING: %s is already at step %d >= num_iterations=%d; training would end immediately."
                  % (resume_target, resume_header["latest_step"], num_iterations), file=sys.stderr)
    elif resume_target and not args.dry_run:
        raise LaunchError("Resume checkpoint not found: %s" % resume_target)

    # -- GPU validation (real CUDA launches only; never on dry-run) ----------------
    device = str(config.get("device", "cuda"))
    gpu_names = None
    if not args.dry_run and device.startswith("cuda"):
        try:
            _, gpu_names = validate_gpus(nproc_per_node, allow_other_gpu=args.allow_other_gpu)
        except ImportError:
            raise LaunchError("torch is not installed in %s; run scripts/setup.sh first."
                              % sys.executable)

    # -- Child environment ---------------------------------------------------------
    child_env = dict(os.environ)
    for var in THREAD_ENV_VARS:
        child_env.setdefault(var, "1")
    child_env.setdefault("MAX_JOBS", "4")  # gsplat JIT compilation fan-out
    arch_list = child_env.get("TORCH_CUDA_ARCH_LIST")
    if not arch_list:
        child_env["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH
    elif TORCH_CUDA_ARCH not in arch_list:
        print("WARNING: TORCH_CUDA_ARCH_LIST=%r lacks %s (H200)." % (arch_list, TORCH_CUDA_ARCH),
              file=sys.stderr)

    # -- Command -------------------------------------------------------------------
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc-per-node", str(nproc_per_node)]
    command = torchrun + [
        str(ROOT / "train.py"),
        "--config", str(config_path),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(accumulation),
        "--num_iterations", str(num_iterations),
        "--strict_checkpoint_load",
    ]
    if resume_target:
        # Full resume wins; clear any config-provided weights-only init so the
        # strict existence check in train.py does not trip on a stale path.
        command += ["--resume_from", resume_target, "--load_from", ""]
    elif init_from:
        command += ["--load_from", os.path.expanduser(os.path.expandvars(init_from))]
    if args.auto_resume:
        command.append("--auto_resume")
    if contract_id and not config.get("pseudo_depth_contract_id"):
        command += ["--pseudo_depth_contract_id", contract_id]
    if args.exp_name:
        command += ["--exp_name", args.exp_name]
    command += passthrough

    # -- Report ----------------------------------------------------------------------
    env_prefix = " ".join("%s=%s" % (k, shlex.quote(child_env[k]))
                          for k in [*THREAD_ENV_VARS, "MAX_JOBS", "TORCH_CUDA_ARCH_LIST",
                                    ENV_DATA_ROOT, ENV_WEIGHTS_ROOT, ENV_RUN_ROOT]
                          if child_env.get(k))
    print("=" * 78)
    print("FreeForward4D launch plan%s" % (" (DRY RUN — nothing is executed)" if args.dry_run else ""))
    print("=" * 78)
    print("config            : %s" % config_path)
    if unresolved_vars:
        print("unresolved macros : %s (placeholder values shown; export them for a real run)"
              % ", ".join(unresolved_vars))
    print("stage             : %s" % stage)
    print("local GPUs        : %d (single node)" % nproc_per_node)
    print("global batch      : %d = %d ranks x %d/rank x %d accumulation"
          % (global_batch, world_size, batch_size, accumulation))
    print("optimizer updates : %d (no learning-rate rescaling; config lr is used as-is)"
          % num_iterations)
    print("experiment        : %s" % log_dir)
    if resume_target:
        print("init (weights)    : %s" % (("%s (superseded by full resume)" % init_from)
                                          if init_from else "none"))
        print("resume (full)     : %s" % resume_target)
    else:
        print("init (weights)    : %s" % (init_from or "none — training from scratch"))
        print("resume (full)     : none")
    if resume_target and resume_header:
        print("resume header     : saved world=%s latest_step=%s (compatible)"
              % (resume_header["world_size"], resume_header["latest_step"]))
    print("auto-resume       : %s (own run directory only)" % ("on" if auto_resume else "off"))
    if config.get("enable_pseudo_depth_loss"):
        print("pseudo-depth id   : %s"
              % (contract_id or config.get("pseudo_depth_contract_id") or "<unresolved>"))
    if gpu_names:
        print("GPUs              : %s" % "; ".join(gpu_names))
    elif device.startswith("cuda"):
        print("GPUs              : validation skipped (dry-run)")
    else:
        print("device            : %s (CPU smoke)" % device)
    print("command           :")
    print("  %s %s" % (env_prefix, shlex.join(command)))
    if args.dry_run:
        return

    # -- Real launch -----------------------------------------------------------------
    raise_nofile_limit()
    os.makedirs(ckpt_dir, exist_ok=True)
    recipe_path = os.path.join(log_dir, "launch_recipe.json")
    atomic_write_json(recipe_path, {
        **recipe,
        "recipe_format": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "config": str(config_path),
        "exp_name": exp_name,
        "project": project,
        "init_from": init_from,
        "resume_from": resume_target,
        "auto_resume": auto_resume,
        "pseudo_depth_contract_id": contract_id or config.get("pseudo_depth_contract_id"),
        "env_roots": {k: child_env.get(k) for k in roots},
        "passthrough_args": passthrough,
        "command": command,
    })
    print("Wrote %s" % recipe_path)
    # exec replaces this process: SIGTERM/SIGINT reach torchrun (and train.py's
    # emergency-checkpoint handler) without a wrapper in between.
    os.execvpe(command[0], command, child_env)


if __name__ == "__main__":
    try:
        main()
    except LaunchError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)
