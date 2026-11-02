#!/usr/bin/env python3
"""Train/evaluate LVSM with explicit paths; --dry-run only prints the resolved plan."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def build_config(args, output):
    from nvs.recipes import model_config
    from nvs.trainval import LVSMLauncherConfig
    config = LVSMLauncherConfig()
    config.model_config = model_config(args.method, args.dataset)
    if args.model_config:
        from nvs.config_io import read_recipe, model_from_recipe
        saved = read_recipe(args.model_config)
        expected_dataset = "realestate10k" if args.dataset == "re10k" else "objaverse"
        if saved.get("dataset_name", expected_dataset) != expected_dataset:
            raise ValueError("The saved model configuration belongs to a different dataset")
        config.model_config = model_from_recipe(saved)
        expected_method = {"worldrope": "urope_uvd_axial", "rayrope": "rayrope", "urope": "urope", "plucker": "none", "prope": "prope", "xyrope": "rope2d"}[args.method]
        if config.model_config.pos_enc != expected_method:
            raise ValueError("--method does not match the saved model configuration")
    config.dataset_name = "realestate10k" if args.dataset == "re10k" else "objaverse"
    config.dataset_batch_scenes = args.batch
    config.dataset_supervise_views = 1
    config.seed = config.data_seed = 42
    config.amp = True
    config.amp_dtype = "fp16"
    config.acc = 1
    config.max_steps = config.test_every = args.steps
    config.use_torch_compile = False
    config.skip_nonfinite_updates = True
    config.output_dir = str(output)
    config.test_only = args.test
    config.resume = str(args.checkpoint.resolve()) if args.checkpoint else None
    config.test_index_fp = str(args.test_index.resolve()) if args.test_index else None
    config.test_n = args.test_n
    config.test_input_views = 2
    config.test_supervise_views = 3
    if args.dataset == "re10k":
        config.re10k_fast_train_dir = str(args.fast_dir.resolve()) if args.fast_dir else None
        config.re10k_legacy_schedule_file = str(args.schedule.resolve()) if args.schedule else None
        config.re10k_image_cache_dir = str(args.image_cache.resolve()) if args.image_cache else None
    else:
        config.objaverse_fast_dir = str(args.fast_dir.resolve()) if args.fast_dir else None
        config.objaverse_schedule_file = str(args.schedule.resolve()) if args.schedule else None
    return config

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["re10k", "objaverse"], default="re10k")
    p.add_argument("--method", choices=["worldrope", "rayrope", "urope", "plucker", "prope", "xyrope"], default="worldrope")
    p.add_argument("--model-config", type=Path, help="Saved recipe supplying model architecture only; other launcher arguments remain explicit")
    p.add_argument("--data-root", type=Path, help="Required for RE10K scene/view data; optional for packed Objaverse")
    p.add_argument("--output-dir", type=Path, required=True, help="Fresh output directory")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--fast-dir", type=Path)
    p.add_argument("--schedule", type=Path)
    p.add_argument("--image-cache", type=Path)
    p.add_argument("--batch", type=int, default=8, help="Scenes per rank")
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--test", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--test-index", type=Path)
    p.add_argument("--test-n", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    gpus = args.gpus.split(",")
    if not all(x.isdigit() for x in gpus) or len(set(gpus)) != len(gpus):
        p.error("--gpus must contain distinct numeric GPU indices")
    if args.batch < 1 or args.steps < 1 or (args.test_n is not None and args.test_n < 1):
        p.error("batch, steps and test-n must be positive")
    if args.test and args.checkpoint is None:
        p.error("--test requires --checkpoint")
    if args.schedule and (not args.fast_dir or len(gpus) != 2 or args.batch != 8):
        p.error("The bundled legacy schedule adapters require --fast-dir, two ranks and batch 8 per rank")
    if args.image_cache and (args.dataset != "re10k" or not args.fast_dir):
        p.error("--image-cache requires RE10K with --fast-dir")
    if args.dataset == "re10k" and args.data_root is None:
        p.error("RE10K requires --data-root")
    if args.dataset == "objaverse" and not args.test and args.schedule is None:
        p.error("Packed Objaverse training requires its matching --schedule")
    if args.dataset == "objaverse" and not args.fast_dir:
        p.error("Objaverse paper data requires the packed radial-0 --fast-dir backend")
    if args.dataset == "objaverse" and args.test_index is not None:
        p.error("The bundled Objaverse evaluator supports its default gap-4 protocol only")
    if args.checkpoint is not None and not args.test:
        p.error("--checkpoint is for evaluation; this launcher starts fresh training")
    plan = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan.update(code_root=str(ROOT), world_size=len(gpus), fixed_schedule=bool(args.schedule), paper_sampling_reproduced=bool(args.schedule))
    if args.dry_run:
        plan["config"] = asdict(build_config(args, args.output_dir.resolve()))
        print(json.dumps(plan, indent=2, default=str))
        return
    if args.data_root is not None and not args.data_root.is_dir():
        p.error("--data-root does not exist")
    for path in [args.fast_dir, args.schedule, args.image_cache, args.checkpoint, args.test_index]:
        if path is not None and not path.exists():
            p.error(f"Input does not exist: {path}")
    output = args.output_dir.resolve()
    if "LOCAL_RANK" not in os.environ and output.exists():
        p.error("--output-dir must be fresh")
    protected = Path("/131_data/limkyujin")
    if output == protected or protected in output.parents:
        p.error("The reference source tree cannot be used for outputs")
    os.chdir(ROOT)
    if args.data_root is not None:
        link = ROOT / "data_processed" / ("realestate10k" if args.dataset == "re10k" else "objaverse/views_release")
        if link.exists() or link.is_symlink():
            if link.resolve() != args.data_root.resolve():
                p.error(f"Existing data link points elsewhere: {link}")
        else:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(args.data_root.resolve(), target_is_directory=True)
    if "LOCAL_RANK" not in os.environ:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS="1", NCCL_P2P_DISABLE="1")
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={len(gpus)}", str(Path(__file__).resolve()), *sys.argv[1:]]
        subprocess.run(cmd, env=env, check=True)
        return
    from nvs.trainval import LVSMLauncher
    LVSMLauncher(build_config(args, output)).run()

if __name__ == "__main__":
    main()
