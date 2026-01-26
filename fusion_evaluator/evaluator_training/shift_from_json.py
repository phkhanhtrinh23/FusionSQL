import argparse
import json
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from .descriptors import compute_delta, frechet_like, mahalanobis_diag, sliced_wasserstein, sliced_wasserstein_hybrid
from .embeddings import embed_texts


def iter_json_array(path: Path, limit: Optional[int] = None) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        ch = f.read(1)
        while ch and ch.isspace():
            ch = f.read(1)
        if ch != "[":
            raise ValueError(f"Expected JSON array in {path}")

        seen = 0
        while True:
            ch = f.read(1)
            while ch and ch.isspace():
                ch = f.read(1)
            if not ch or ch == "]":
                break
            if ch == ",":
                continue
            if ch != "{":
                raise ValueError(f"Expected object in array in {path}")

            buf = ["{"]
            depth = 1
            in_str = False
            escape = False
            while True:
                ch = f.read(1)
                if not ch:
                    raise ValueError(f"Unexpected EOF while reading {path}")
                buf.append(ch)
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == "\"":
                        in_str = False
                else:
                    if ch == "\"":
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            break

            obj = json.loads("".join(buf))
            yield obj
            seen += 1
            if limit is not None and seen >= limit:
                break


def _format_column(name: str, col_type: Optional[str], comment: Optional[str], is_pk: bool) -> str:
    pieces = [name]
    meta = []
    if col_type:
        meta.append(f"type: {col_type}")
    if comment:
        meta.append(f"comment: {comment}")
    if is_pk:
        meta.append("primary key")
    if meta:
        pieces.append(f"({'; '.join(meta)})")
    return " ".join(pieces)


def _schema_to_prompt(schema: dict) -> str:
    items = schema.get("schema_items") or []
    lines = []
    for item in items:
        table_name = item.get("table_name") or "unknown_table"
        table_comment = item.get("table_comment") or ""
        header = f"Table {table_name}"
        if table_comment:
            header = f"{header} ({table_comment})"
        columns = item.get("column_names") or []
        col_types = item.get("column_types") or []
        col_comments = item.get("column_comments") or []
        pk_flags = item.get("pk_indicators") or []
        formatted_cols = []
        for i, name in enumerate(columns):
            col_type = col_types[i] if i < len(col_types) else None
            comment = col_comments[i] if i < len(col_comments) else None
            is_pk = bool(pk_flags[i]) if i < len(pk_flags) else False
            formatted_cols.append(_format_column(str(name), col_type, comment, is_pk))
        if formatted_cols:
            lines.append(f"{header} has columns: {', '.join(formatted_cols)}.")
        else:
            lines.append(f"{header} has no columns listed.")

    fk_items = schema.get("foreign_keys") or []
    if fk_items:
        fk_parts = []
        for fk in fk_items:
            if not isinstance(fk, list) or len(fk) != 4:
                continue
            src_table, src_col, tgt_table, tgt_col = fk
            fk_parts.append(f"{src_table}.{src_col} -> {tgt_table}.{tgt_col}")
        if fk_parts:
            lines.append(f"Foreign keys: {', '.join(fk_parts)}.")
    return " ".join(lines).strip()


def _record_to_prompt(obj: dict) -> Optional[str]:
    question = obj.get("question") or obj.get("text") or ""
    schema = obj.get("schema")
    if not question or not schema:
        return None
    schema_text = _schema_to_prompt(schema)
    if not schema_text:
        return None
    return f"Question: {question} Schema: {schema_text}"


def collect_field_texts(
    path: Path,
    field: str,
    limit: Optional[int] = None,
) -> Tuple[list[str], int]:
    texts = []
    skipped = 0
    for obj in iter_json_array(path, limit=limit):
        if field == "prompt":
            prompt = _record_to_prompt(obj)
            if prompt:
                texts.append(prompt)
            else:
                skipped += 1
            continue
        val = obj.get(field)
        if isinstance(val, str) and val.strip():
            texts.append(val)
        else:
            skipped += 1
    return texts, skipped


def compute_descriptors(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    *,
    slices: int = 64,
    use_hybrid_swd: bool = False,
    pca_k: int = 8,
    rand_r: int = 16,
    pca_subsample: Optional[int] = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = frechet_like(source_emb, target_emb)
    m = mahalanobis_diag(source_emb, target_emb)
    if use_hybrid_swd:
        s = sliced_wasserstein_hybrid(
            source_emb,
            target_emb,
            k=pca_k,
            R=rand_r,
            seed=seed,
            subsample=pca_subsample,
        )
    else:
        s = sliced_wasserstein(source_emb, target_emb, L=slices, seed=seed)
    delta = compute_delta(
        source_emb,
        target_emb,
        L=slices,
        use_hybrid_swd=use_hybrid_swd,
        pca_k=pca_k,
        rand_r=rand_r,
        subsample=pca_subsample,
        seed=seed,
    )
    return g, m, s, delta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pooled embeddings and shift descriptors for two JSON datasets.",
    )
    parser.add_argument(
        "--source",
        default="fusion_evaluator/data/spider/sft_spider_dev_text2sql.json",
        help="Path to source JSON array.",
    )
    parser.add_argument(
        "--target",
        default="fusion_evaluator/data/bird/sft_bird_dev_text2sql.json",
        help="Path to target JSON array.",
    )
    parser.add_argument(
        "--source_field",
        default="prompt",
        help="Field to embed from source records. Use 'prompt' to combine question + schema.",
    )
    parser.add_argument(
        "--target_field",
        default="prompt",
        help="Field to embed from target records. Use 'prompt' to combine question + schema.",
    )
    parser.add_argument(
        "--target_limit",
        type=int,
        default=500,
        help="Number of target samples to embed.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model for embeddings.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--torch_dtype", default=None, help="fp16|bf16|fp32")
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--hybrid_swd", action="store_true", default=True)
    parser.add_argument("--no_hybrid_swd", action="store_false", dest="hybrid_swd")
    parser.add_argument("--pca_k", type=int, default=8)
    parser.add_argument("--rand_r", type=int, default=16)
    parser.add_argument("--pca_subsample", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)

    source_texts, source_skipped = collect_field_texts(source_path, args.source_field)
    target_texts, target_skipped = collect_field_texts(
        target_path,
        args.target_field,
        limit=args.target_limit,
    )

    print("Target texts collected:", target_texts[:1])

    if not source_texts:
        raise ValueError(f"No texts extracted from {source_path} using field '{args.source_field}'.")
    if not target_texts:
        raise ValueError(f"No texts extracted from {target_path} using field '{args.target_field}'.")

    source_emb = embed_texts(
        source_texts,
        model_name_or_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )
    target_emb = embed_texts(
        target_texts,
        model_name_or_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )

    g, m, s, delta = compute_descriptors(
        source_emb,
        target_emb,
        slices=args.slices,
        use_hybrid_swd=args.hybrid_swd,
        pca_k=args.pca_k,
        rand_r=args.rand_r,
        pca_subsample=args.pca_subsample,
        seed=args.seed,
    )

    payload = {
        "source_count": len(source_texts),
        "target_count": len(target_texts),
        "source_skipped": source_skipped,
        "target_skipped": target_skipped,
        "frechet_like": g.tolist(),
        "mahalanobis_diag": m.tolist(),
        "sliced_wasserstein": s.tolist(),
        "delta_vector": delta.tolist(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
