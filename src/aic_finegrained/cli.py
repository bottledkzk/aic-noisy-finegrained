"""Public entry points for raw images and score-only examples."""

import argparse
import json
from pathlib import Path

import numpy as np

from .scores import calibrate_scores, synthetic_comparison


def calibration_options(parser):
    parser.add_argument("--method", choices=("sot", "balanced", "none"), default="sot")
    parser.add_argument("--rho", type=float, default=9.0, help="Class-marginal KL penalty for SOT")
    parser.add_argument("--temperature", type=float, default=1.0)


def image_options(parser, batch):
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True, help="Local Transformers CLIP directory")
    parser.add_argument("--size", type=int, default=544)
    parser.add_argument("--batch", type=int, default=batch)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Robust fine-grained recognition with CLIP and SOT")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Compare three postprocessors on generated CPU data")
    demo.add_argument("--seed", type=int, default=43)
    image_demo = sub.add_parser("demo-images", help="Run the five training stages on synthetic images on CPU")
    image_demo.add_argument("--out", type=Path, required=True, help="New directory for the demonstration")
    image_demo.add_argument("--seed", type=int, default=43)
    scores = sub.add_parser("scores", help="Reprocess an existing N by C canonical score matrix")
    scores.add_argument("--input", type=Path, required=True)
    scores.add_argument("--out", type=Path, required=True, help="Output NumPy probability array")
    calibration_options(scores)

    fitting = sub.add_parser("train", help="Fit the five image-training stages")
    image_options(fitting, batch=256)
    fitting.add_argument("--work", type=Path, required=True)
    fitting.add_argument("--stage", choices=("all", "ce", "groupcache", "sp2", "teacher", "student"), default="all")
    fitting.add_argument("--epochs", type=int, default=80)
    fitting.add_argument("--seed", type=int, default=43)
    fitting.add_argument("--cache-size", type=int, default=608)
    fitting.add_argument("--feature-batch", type=int, default=32)
    fitting.add_argument("--target-k", type=int, default=60)
    fitting.add_argument("--group-ids", type=Path, help="Optional integer group IDs in train CSV order")

    prediction = sub.add_parser("infer", help="Generate predictions from one visual classifier")
    image_options(prediction, batch=32)
    prediction.add_argument("--query-csv", type=Path, required=True)
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--bank-indices", type=Path, help="Optional zero-based training-row indices")
    prediction.add_argument("--neighbor-k", type=int, default=60)
    prediction.add_argument("--graph-k", type=int, default=10, help="Use 0 to disable query-graph smoothing")
    prediction.add_argument("--out", type=Path, required=True)
    prediction.add_argument("--canonical-out", type=Path, help="Optional canonical score output for replay")
    calibration_options(prediction)

    args = parser.parse_args(argv)
    if args.command == "demo":
        print(json.dumps(synthetic_comparison(args.seed), indent=2))
    elif args.command == "demo-images":
        from .demo_images import run_demo

        run_demo(args.out, args.seed)
    elif args.command == "scores":
        with args.input.open("rb") as stream:
            matrix = np.load(stream, allow_pickle=False)
        q, info = calibrate_scores(matrix, args.method, args.rho, args.temperature)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            np.save(stream, q)
        print(json.dumps(info, indent=2))
    elif args.command == "train":
        from .train import train

        train(args)
    else:
        from .infer import infer

        infer(args)
