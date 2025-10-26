import argparse
import json
import os
from typing import List

import numpy as np

from .descriptors import compute_delta
from .embeddings import embed_texts
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
            deltas.append(compute_delta(
                s_src,
                t_tgt,
                L=args.slices,
                use_hybrid_swd=args.hybrid_swd,
                pca_k=args.pca_k,
                rand_r=args.rand_r,
                subsample=args.pca_subsample,
            ))
            ys.append(float(item["metric"]))
    else:
        s_src = load_embeddings(args.source_embeddings)
        t_tgt = load_embeddings(args.target_embeddings)
        deltas.append(compute_delta(
            s_src,
            t_tgt,
            L=args.slices,
            use_hybrid_swd=args.hybrid_swd,
            pca_k=args.pca_k,
            rand_r=args.rand_r,
            subsample=args.pca_subsample,
        ))
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
    delta = compute_delta(
        s_src,
        t_tgt,
        L=args.slices,
        use_hybrid_swd=args.hybrid_swd,
        pca_k=args.pca_k,
        rand_r=args.rand_r,
        subsample=args.pca_subsample,
    )[None, :]
	pred = float(model.predict(delta)[0])
	print(json.dumps({"predicted_metric": pred}, indent=2))


def cmd_embed(args: argparse.Namespace) -> None:
    # Build texts list: by default we embed SQLs; optionally include questions
    with open(args.input, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    texts: List[str] = []
    for r in items:
        if args.field == "sql":
            if r.get("gold_sql"):
                texts.append(str(r["gold_sql"]))
            elif r.get("query"):
                texts.append(str(r["query"]))
        else:
            if r.get("question"):
                texts.append(str(r["question"]))
    vecs = embed_texts(
        texts,
        model_name_or_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, vecs)
    print(json.dumps({"num": len(texts), "dim": int(vecs.shape[1]) if vecs.size else 0, "output": args.output}))


def main() -> None:
	parser = argparse.ArgumentParser(description="Train/Infer FusionSQL evaluator")
	sub = parser.add_subparsers(dest="cmd", required=True)

	tr = sub.add_parser("train", help="Train evaluator model on a workload pair")
	tr.add_argument("--source_embeddings", required=False, help="Path to source embeddings (.npy or JSON)")
	tr.add_argument("--target_embeddings", required=False, help="Path to target embeddings (.npy or JSON)")
	tr.add_argument("--observed_metric", type=float, required=False, help="Observed execution accuracy for target workload")
	tr.add_argument("--multi", required=False, help="JSON file of training pairs for multi-workload training")
    tr.add_argument("--slices", type=int, default=64)
    tr.add_argument("--hybrid_swd", action="store_true")
    tr.add_argument("--pca_k", type=int, default=8)
    tr.add_argument("--rand_r", type=int, default=16)
    tr.add_argument("--pca_subsample", type=int, default=4096)
	tr.add_argument("--out", required=True, help="Path to save trained model (.joblib)")
	tr.set_defaults(func=cmd_train)

	inf = sub.add_parser("infer", help="Predict metric for new workload")
	inf.add_argument("--model", required=True, help="Path to trained model (.joblib)")
	inf.add_argument("--source_embeddings", required=True)
	inf.add_argument("--target_embeddings", required=True)
    inf.add_argument("--slices", type=int, default=64)
    inf.add_argument("--hybrid_swd", action="store_true")
    inf.add_argument("--pca_k", type=int, default=8)
    inf.add_argument("--rand_r", type=int, default=16)
    inf.add_argument("--pca_subsample", type=int, default=4096)
	inf.set_defaults(func=cmd_infer)

    emb = sub.add_parser("embed", help="Compute embeddings for a dataset JSONL")
    emb.add_argument("--input", required=True, help="Aligned JSONL with gold_sql/question fields")
    emb.add_argument("--output", required=True, help="Path to save .npy embeddings")
    emb.add_argument("--model", required=True, help="HF model name, e.g., Qwen/Qwen2.5-7B-Instruct")
    emb.add_argument("--field", choices=["sql", "question"], default="sql", help="Which field to embed")
    emb.add_argument("--device", default=None)
    emb.add_argument("--batch_size", type=int, default=16)
    emb.add_argument("--max_length", type=int, default=256)
    emb.add_argument("--torch_dtype", default=None, help="fp16|bf16|fp32")
    emb.set_defaults(func=cmd_embed)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
