import argparse
import json
import os
from typing import List

import numpy as np

from ..data.datasets import (
    load_spider,
    load_spider20,
    load_bird,
    load_wikisql,
    load_sparc,
    load_cosql,
)
from ..data.loaders import read_jsonl
from ..cli import main as eval_cli_main  # for alignment path semantics if needed
from .embeddings import embed_texts
from .descriptors import compute_delta
from .model import EvaluatorModel, EvaluatorModelConfig


def build_aligned_records(dataset: str, gold: str, pred: str, db_root: str, tables_json: str | None, wikisql_tables: str | None, wikisql_db_out: str | None) -> tuple[list[dict], str]:
    if dataset == "spider":
        records = load_spider(gold, pred)
        root = db_root
    elif dataset == "spider2":
        records = load_spider20(gold, pred)
        root = db_root
    elif dataset == "bird":
        records = load_bird(gold, pred)
        root = db_root
    elif dataset == "wikisql":
        assert wikisql_tables and wikisql_db_out
        records = load_wikisql(gold, pred, wikisql_tables, wikisql_db_out)
        root = wikisql_db_out
    elif dataset == "sparc":
        records = load_sparc(gold, pred)
        root = db_root
    else:
        records = load_cosql(gold, pred)
        root = db_root
    return records, root


def cmd_train_pipeline(args: argparse.Namespace) -> None:
    # Build aligned records for source dataset and FusionDataset
    src_records, _ = build_aligned_records(
        args.dataset,
        args.gold,
        args.pred,
        args.db_root,
        args.tables_json,
        args.wikisql_tables,
        args.wikisql_db_out,
    )
    with open(args.fusion_jsonl, "r", encoding="utf-8") as f:
        fusion_records = [json.loads(line) for line in f if line.strip()]

    # Compute embeddings
    src_texts = [r["gold_sql"] for r in src_records if r.get("gold_sql")]
    fusion_texts = [r["gold_sql"] for r in fusion_records if r.get("gold_sql")]

    src_emb = embed_texts(
        src_texts,
        model_name_or_path=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )
    fusion_emb = embed_texts(
        fusion_texts,
        model_name_or_path=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )

    # Compute delta features
    X = compute_delta(
        src_emb,
        fusion_emb,
        L=args.slices,
        use_hybrid_swd=args.hybrid_swd,
        pca_k=args.pca_k,
        rand_r=args.rand_r,
        subsample=args.pca_subsample,
    )[None, :]
    y = np.array([args.exec_accuracy], dtype=np.float32)

    model = EvaluatorModel(EvaluatorModelConfig())
    model.fit(X, y, calibrate=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(json.dumps({"saved": args.out, "feature_dim": int(X.shape[1])}))


def cmd_infer_pipeline(args: argparse.Namespace) -> None:
    # Build aligned records for target evaluation dataset and FusionDataset
    tgt_records, _ = build_aligned_records(
        args.dataset,
        args.gold,
        args.pred,
        args.db_root,
        args.tables_json,
        args.wikisql_tables,
        args.wikisql_db_out,
    )
    with open(args.fusion_jsonl, "r", encoding="utf-8") as f:
        fusion_records = [json.loads(line) for line in f if line.strip()]

    tgt_texts = [r["gold_sql"] for r in tgt_records if r.get("gold_sql")]
    fusion_texts = [r["gold_sql"] for r in fusion_records if r.get("gold_sql")]

    tgt_emb = embed_texts(
        tgt_texts,
        model_name_or_path=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )
    fusion_emb = embed_texts(
        fusion_texts,
        model_name_or_path=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )

    X = compute_delta(
        tgt_emb,
        fusion_emb,
        L=args.slices,
        use_hybrid_swd=args.hybrid_swd,
        pca_k=args.pca_k,
        rand_r=args.rand_r,
        subsample=args.pca_subsample,
    )[None, :]

    model = EvaluatorModel.load(args.model)
    pred = float(model.predict(X)[0])
    print(json.dumps({"predicted_execution_accuracy": pred}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline: embed + delta + MLP for execution_accuracy")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train regressor with FusionDataset delta features and observed metric")
    tr.add_argument("--dataset", choices=["spider", "spider2", "bird", "wikisql", "sparc", "cosql"], required=True)
    tr.add_argument("--gold", required=True)
    tr.add_argument("--pred", required=True)
    tr.add_argument("--db_root", required=False, default=None)
    tr.add_argument("--tables_json", required=False, default=None)
    tr.add_argument("--wikisql_tables", required=False, default=None)
    tr.add_argument("--wikisql_db_out", required=False, default=None)
    tr.add_argument("--fusion_jsonl", required=True, help="Path to fusion_dataset.jsonl")
    tr.add_argument("--exec_accuracy", type=float, required=True, help="Observed execution_accuracy of target dataset for this base model")
    tr.add_argument("--model_name", required=True, help="HF model (e.g., Qwen/Qwen2.5-72B-Instruct)")
    tr.add_argument("--device", default=None)
    tr.add_argument("--batch_size", type=int, default=16)
    tr.add_argument("--max_length", type=int, default=256)
    tr.add_argument("--torch_dtype", default=None)
    tr.add_argument("--slices", type=int, default=64)
    tr.add_argument("--hybrid_swd", action="store_true")
    tr.add_argument("--pca_k", type=int, default=8)
    tr.add_argument("--rand_r", type=int, default=16)
    tr.add_argument("--pca_subsample", type=int, default=4096)
    tr.add_argument("--out", required=True)
    tr.set_defaults(func=cmd_train_pipeline)

    inf = sub.add_parser("infer", help="Predict execution_accuracy on a target dataset using trained model")
    inf.add_argument("--dataset", choices=["spider", "spider2", "bird", "wikisql", "sparc", "cosql"], required=True)
    inf.add_argument("--gold", required=True)
    inf.add_argument("--pred", required=True)
    inf.add_argument("--db_root", required=False, default=None)
    inf.add_argument("--tables_json", required=False, default=None)
    inf.add_argument("--wikisql_tables", required=False, default=None)
    inf.add_argument("--wikisql_db_out", required=False, default=None)
    inf.add_argument("--fusion_jsonl", required=True)
    inf.add_argument("--model_name", required=True)
    inf.add_argument("--device", default=None)
    inf.add_argument("--batch_size", type=int, default=16)
    inf.add_argument("--max_length", type=int, default=256)
    inf.add_argument("--torch_dtype", default=None)
    inf.add_argument("--slices", type=int, default=64)
    inf.add_argument("--hybrid_swd", action="store_true")
    inf.add_argument("--pca_k", type=int, default=8)
    inf.add_argument("--rand_r", type=int, default=16)
    inf.add_argument("--pca_subsample", type=int, default=4096)
    inf.add_argument("--model", required=True, help="Path to trained model .joblib")
    inf.set_defaults(func=cmd_infer_pipeline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


