import argparse
import json
import os
from typing import List

import numpy as np

from .descriptors import compute_delta
from .model import EvaluatorModel, EvaluatorModelConfig


def load_embeddings(path: str) -> np.ndarray:
	# Expect .npy of shape (N, D) or JSON list of lists
	if path.lower().endswith(".npy"):
		return np.load(path)
	with open(path, "r", encoding="utf-8") as f:
		obj = json.load(f)
		return np.array(obj, dtype=np.float32)


def cmd_train(args: argparse.Namespace) -> None:
    # Support multi-workload training via multiple pairs
    deltas = []
    ys = []
    if args.multi is not None:
        # args.multi is a JSON file: [{"source": path, "target": path, "metric": float}, ...]
        with open(args.multi, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        for item in pairs:
            s_src = load_embeddings(item["source"])
            t_tgt = load_embeddings(item["target"])
            deltas.append(compute_delta(s_src, t_tgt, L=args.slices))
            ys.append(float(item["metric"]))
    else:
        s_src = load_embeddings(args.source_embeddings)
        t_tgt = load_embeddings(args.target_embeddings)
        deltas.append(compute_delta(s_src, t_tgt, L=args.slices))
        ys.append(float(args.observed_metric))

    X = np.stack(deltas, axis=0)
    y = np.array(ys, dtype=np.float32)
    model = EvaluatorModel(EvaluatorModelConfig())
    model.fit(X, y, calibrate=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"Saved evaluator model to {args.out}")


def cmd_infer(args: argparse.Namespace) -> None:
	model = EvaluatorModel.load(args.model)
	s_src = load_embeddings(args.source_embeddings)
	t_tgt = load_embeddings(args.target_embeddings)
	delta = compute_delta(s_src, t_tgt, L=args.slices)[None, :]
	pred = float(model.predict(delta)[0])
	print(json.dumps({"predicted_metric": pred}, indent=2))


def main() -> None:
	parser = argparse.ArgumentParser(description="Train/Infer FusionSQL evaluator")
	sub = parser.add_subparsers(dest="cmd", required=True)

	tr = sub.add_parser("train", help="Train evaluator model on a workload pair")
    tr.add_argument("--source_embeddings", required=False, help="Path to source embeddings (.npy or JSON)")
    tr.add_argument("--target_embeddings", required=False, help="Path to target embeddings (.npy or JSON)")
    tr.add_argument("--observed_metric", type=float, required=False, help="Observed execution accuracy for target workload")
    tr.add_argument("--multi", required=False, help="JSON file of training pairs for multi-workload training")
	tr.add_argument("--slices", type=int, default=64)
	tr.add_argument("--out", required=True, help="Path to save trained model (.joblib)")
	tr.set_defaults(func=cmd_train)

	inf = sub.add_parser("infer", help="Predict metric for new workload")
	inf.add_argument("--model", required=True, help="Path to trained model (.joblib)")
	inf.add_argument("--source_embeddings", required=True)
	inf.add_argument("--target_embeddings", required=True)
	inf.add_argument("--slices", type=int, default=64)
	inf.set_defaults(func=cmd_infer)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
