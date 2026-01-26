import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .descriptors import compute_delta
from .embeddings import embed_texts
from .model import EvaluatorModel, EvaluatorModelConfig
from .shift_from_json import iter_json_array, _record_to_prompt
from ..exec.engine import SQLiteExecutor
from ..metrics.basic import execution_accuracy_metric


def load_prompt_records(path: Path) -> list[dict]:
    records = []
    for obj in iter_json_array(path):
        prompt = _record_to_prompt(obj)
        if not prompt:
            continue
        db_id = obj.get("db_id")
        gold_sql = obj.get("sql")
        if not isinstance(db_id, str) or not isinstance(gold_sql, str):
            continue
        records.append(
            {
                "db_id": db_id,
                "prompt": prompt,
                "gold_sql": gold_sql,
            }
        )
    return records


def build_chat_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    system = "You are a text-to-SQL assistant. Return only the SQL query, no explanation."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\nUser: {prompt}\nAssistant:"


def extract_sql(text: str) -> str:
    if not text:
        return ""
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            if not block:
                continue
            if block.lower().startswith("sql"):
                block = block[3:].strip()
            if block:
                return block
    upper = text.upper()
    for kw in ("WITH", "SELECT", "INSERT", "UPDATE", "DELETE"):
        idx = upper.find(kw)
        if idx >= 0:
            return text[idx:].strip()
    return text.strip().splitlines()[0].strip()


def generate_sqls(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: list[str],
    *,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[str]:
    outputs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = [build_chat_prompt(tokenizer, p) for p in prompts[i : i + batch_size]]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row, in_len in zip(out, input_lens):
            gen_ids = row[int(in_len) :]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            outputs.append(extract_sql(gen_text))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample target subsets, compute shift vectors + execution accuracy, train MLP regressor.",
    )
    parser.add_argument(
        "--source",
        default="fusion_evaluator/data/spider/sft_spider_train_text2sql.json",
        help="Source JSON array (training workload).",
    )
    parser.add_argument(
        "--target",
        default="fusion_evaluator/data/bird/sft_bird_dev_text2sql.json",
        help="Target JSON array (evaluation workload).",
    )
    parser.add_argument(
        "--db_root",
        required=True,
        help="Root directory containing db_id folders with .sqlite files.",
    )
    parser.add_argument(
        "--target_limit",
        type=int,
        default=500,
        help="Number of samples per target subset.",
    )
    parser.add_argument(
        "--num_sets",
        type=int,
        default=100,
        help="Number of random subsets to sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model for SQL generation.",
    )
    parser.add_argument(
        "--embed_model",
        default=None,
        help="HF model for embeddings (defaults to --model).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embed_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--torch_dtype", default=None, help="fp16|bf16|fp32")
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--hybrid_swd", action="store_true", default=True)
    parser.add_argument("--no_hybrid_swd", action="store_false", dest="hybrid_swd")
    parser.add_argument("--pca_k", type=int, default=8)
    parser.add_argument("--rand_r", type=int, default=16)
    parser.add_argument("--pca_subsample", type=int, default=4096)
    parser.add_argument("--gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--out_npz", default="outputs/shift_samples/shift_samples.npz")
    parser.add_argument("--out_model", default="outputs/shift_samples/shift_mlp.joblib")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    source_records = load_prompt_records(Path(args.source))
    target_records = load_prompt_records(Path(args.target))
    if not source_records:
        raise ValueError("No source prompts extracted.")
    if not target_records:
        raise ValueError("No target prompts extracted.")
    if args.target_limit > len(target_records):
        raise ValueError("target_limit exceeds number of target records.")

    source_prompts = [r["prompt"] for r in source_records]
    target_prompts = [r["prompt"] for r in target_records]

    embed_model = args.embed_model or args.model

    source_emb = embed_texts(
        source_prompts,
        model_name_or_path=embed_model,
        device=args.device,
        batch_size=args.embed_batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )
    target_emb = embed_texts(
        target_prompts,
        model_name_or_path=embed_model,
        device=args.device,
        batch_size=args.embed_batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = None
    if args.torch_dtype:
        low = args.torch_dtype.lower()
        if low in ("fp16", "float16"):
            dtype = torch.float16
        elif low in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif low in ("fp32", "float32"):
            dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    executor = SQLiteExecutor(args.db_root)

    all_sql = generate_sqls(
        tokenizer,
        model,
        target_prompts,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.gen_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    accuracies = np.zeros(len(target_records), dtype=np.float32)
    for i, (rec, pred_sql) in enumerate(zip(target_records, all_sql)):
        accuracies[i] = execution_accuracy_metric(
            executor,
            rec["db_id"],
            pred_sql,
            rec["gold_sql"],
        )

    deltas = []
    accs = []
    sample_indices = []
    for _ in range(args.num_sets):
        idx = rng.choice(len(target_records), size=args.target_limit, replace=False)
        sample_indices.append(idx)
        tgt_subset = target_emb[idx]
        delta = compute_delta(
            source_emb,
            tgt_subset,
            L=args.slices,
            use_hybrid_swd=args.hybrid_swd,
            pca_k=args.pca_k,
            rand_r=args.rand_r,
            subsample=args.pca_subsample,
            seed=args.seed,
        )
        deltas.append(delta)
        accs.append(float(accuracies[idx].mean()))

    deltas_arr = np.stack(deltas, axis=0)
    accs_arr = np.array(accs, dtype=np.float32)
    sample_indices_arr = np.stack(sample_indices, axis=0)

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez(
        args.out_npz,
        deltas=deltas_arr,
        accuracies=accs_arr,
        sample_indices=sample_indices_arr,
        seed=args.seed,
        target_limit=args.target_limit,
        num_sets=args.num_sets,
    )

    model_cfg = EvaluatorModelConfig(hidden_sizes=(256, 128, 64), seed=args.seed)
    reg = EvaluatorModel(model_cfg)
    reg.fit(deltas_arr, accs_arr, calibrate=False)
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    reg.save(args.out_model)

    print(json.dumps({"npz": args.out_npz, "model": args.out_model, "num_sets": args.num_sets}, indent=2))


if __name__ == "__main__":
    main()
