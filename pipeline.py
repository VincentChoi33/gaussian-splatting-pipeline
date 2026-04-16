#!/usr/bin/env python3
"""Gaussian Splatting Pipeline — end-to-end video/photos to compressed PLY."""
import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def cmd_run(args):
    """Run full pipeline: preprocess → sfm → train → export → compress."""
    from pipeline.preprocess import run_preprocess
    from pipeline.sfm import run_sfm
    from pipeline.train import run_train
    from pipeline.export import run_export
    from pipeline.compress import run_compress

    cfg = load_config(args.config)
    inp = Path(args.input)
    out = Path(args.output) / args.name
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Pipeline start: {inp} → {out}")

    images_dir = run_preprocess(inp, out, cfg["preprocess"])
    run_sfm(images_dir, out, cfg["sfm"])
    run_train(out, cfg["train"], name=args.name)
    full_ply = run_export(out, cfg["export"])
    run_compress(full_ply, out, cfg["compress"])

    log.info(f"Pipeline complete: {out}")


def cmd_preprocess(args):
    from pipeline.preprocess import run_preprocess
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    run_preprocess(Path(args.input), out, cfg["preprocess"])


def cmd_sfm(args):
    from pipeline.sfm import run_sfm
    cfg = load_config(args.config)
    out = Path(args.output)
    run_sfm(Path(args.input), out, cfg["sfm"])


def cmd_train(args):
    from pipeline.train import run_train
    cfg = load_config(args.config)
    out = Path(args.output)
    run_train(out, cfg["train"], name=args.name or out.name)


def cmd_export(args):
    from pipeline.export import run_export
    cfg = load_config(args.config)
    out = Path(args.output)
    run_export(out, cfg["export"])


def cmd_compress(args):
    from pipeline.compress import run_compress
    cfg = load_config(args.config)
    inp = Path(args.input)
    out = Path(args.output)
    run_compress(inp, out, cfg["compress"])


def main():
    parser = argparse.ArgumentParser(description="Gaussian Splatting Pipeline")
    parser.add_argument("--config", type=Path, default=None, help="Custom config YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--input", "-i", required=True, help="Video file or image directory")
    p_run.add_argument("--output", "-o", required=True, help="Output base directory")
    p_run.add_argument("--name", "-n", required=True, help="Scene name")
    p_run.set_defaults(func=cmd_run)

    p_pre = sub.add_parser("preprocess", help="Extract frames + blur filter")
    p_pre.add_argument("--input", "-i", required=True)
    p_pre.add_argument("--output", "-o", required=True)
    p_pre.set_defaults(func=cmd_preprocess)

    p_sfm = sub.add_parser("sfm", help="Run SfM (LightGlue + COLMAP)")
    p_sfm.add_argument("--input", "-i", required=True, help="Images directory")
    p_sfm.add_argument("--output", "-o", required=True)
    p_sfm.set_defaults(func=cmd_sfm)

    p_train = sub.add_parser("train", help="Train 3DGS")
    p_train.add_argument("--output", "-o", required=True, help="Scene directory (with sparse/0)")
    p_train.add_argument("--name", "-n", default=None)
    p_train.set_defaults(func=cmd_train)

    p_export = sub.add_parser("export", help="Export ckpt → PLY")
    p_export.add_argument("--output", "-o", required=True, help="Scene directory")
    p_export.set_defaults(func=cmd_export)

    p_compress = sub.add_parser("compress", help="Compress PLY")
    p_compress.add_argument("--input", "-i", required=True, help="Full PLY path")
    p_compress.add_argument("--output", "-o", required=True, help="Output directory")
    p_compress.set_defaults(func=cmd_compress)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
