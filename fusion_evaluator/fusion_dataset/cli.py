import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .acquire import discover_csvs
from .llm import PromptLoader, HuggingFaceLLM, GenerationConfig, render_template
from .refine import is_english_text, structural_ok, header_hash, semantic_richness_score
from .synthesize import cluster_tables, materialize_sqlite
from .sqlgen import simple_select, count_rows, group_by_agg, complex_nested
from .questions import synthesize_question


def main() -> None:
	parser = argparse.ArgumentParser(description="Build FusionDataset")
	parser.add_argument("--sources", nargs="+", help="Root folders containing CSVs (TabLib/Kaggle-like)")
	parser.add_argument("--out_root", required=True, help="Output root for DBs and dataset JSONL")
	parser.add_argument("--max_tables", type=int, default=500)
	parser.add_argument("--min_semantic", type=float, default=0.02)
	# LLM prompting
	parser.add_argument("--prompts", default=None, help="Path to prompts.yaml")
	parser.add_argument("--hf_model", default=None, help="HuggingFace model name for prompting")
	parser.add_argument("--device", default=None)
	parser.add_argument("--torch_dtype", default=None)
	parser.add_argument("--q_styles", default="formal,colloquial,imperative,interrogative,descriptive,vague,metaphorical,conversational", help="Comma-separated question styles")
	parser.add_argument("--q_per_sql", type=int, default=4)
	parser.add_argument("--enable_rewrites", action="store_true")
	parser.add_argument("--rw_per_cat", type=int, default=2, help="Number of Q/A pairs per rewrite category")
	args = parser.parse_args()

	os.makedirs(args.out_root, exist_ok=True)
	csvs = discover_csvs(args.sources)
	rows: List[Dict] = []

	# Load and filter CSVs
	dfs: Dict[str, pd.DataFrame] = {}
	for p in tqdm(csvs[: args.max_tables], desc="load_csv"):
		try:
			df = pd.read_csv(p)
		except Exception:
			continue
		if not structural_ok(df):
			continue
		if not any(is_english_text(str(h)) for h in df.columns.tolist()[: min(5, len(df.columns))]):
			continue
		if semantic_richness_score(df) < args.min_semantic:
			continue
		dfs[p] = df

	# Cluster and build DBs
	clusters = cluster_tables(dfs)
	db_index = 0
	for cluster in tqdm(clusters, desc="synthesize_dbs"):
		if not cluster:
			continue
		db_id = f"fusion_{db_index:05d}"
		db_index += 1
		tables = {Path(p).stem: dfs[p] for p in cluster}
		db_path = materialize_sqlite(os.path.join(args.out_root, "databases"), db_id, tables)

	# Optional: load prompts and model
	pl = None
	llm = None
	if args.prompts and args.hf_model:
		pl = PromptLoader(args.prompts)
		llm = HuggingFaceLLM(args.hf_model, device=args.device, torch_dtype=args.torch_dtype)
		gen_cfg = GenerationConfig(max_new_tokens=128)

	# Generate a few SQLs per DB
		for tname, df in tables.items():
			cols = list(map(str, df.columns))
			cands = [
				simple_select(tname, cols),
				count_rows(tname),
			]
			if len(cols) >= 2:
				cands.append(group_by_agg(tname, cols[0], cols[1]))
			cands.append(complex_nested(tname, cols[0]))

			for sql in cands:
				# Use template-based LLM question generation if available; fall back to local synthesize
				if pl and llm:
					styles = args.q_styles.split(",")
					prompt = render_template(
						pl.get("question_generation")["user"],
						db_id=db_id,
						table=tname,
						columns=", ".join(cols),
						sql=sql,
						num=args.q_per_sql,
						styles=", ".join(styles),
					)
					qs = llm.generate([prompt], gen_cfg)[0].splitlines()
					qs = [q.strip() for q in qs if q.strip()]
					for idx, q in enumerate(qs[: args.q_per_sql]):
						rows.append({
							"id": f"{db_id}:{tname}:llm:{abs(hash(sql))%10**9}:{idx}",
							"db_id": db_id,
							"question": q,
							"gold_sql": sql,
						})

					# Optional SQL/Q rewrites per categories
					if args.enable_rewrites:
						def _emit_pairs(tkey: str, q_base: str, sql_base: str) -> None:
							pt = pl.get(tkey)
							if not pt:
								return
							p = render_template(
								pt["user"],
								db_id=db_id,
								table=tname,
								columns=", ".join(cols),
								question=q_base,
								sql=sql_base,
								num=args.rw_per_cat,
							)
							out = llm.generate([p], gen_cfg)[0]
							lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
							cur_q = None
							for ln in lines:
								if ln.startswith("Q:"):
									cur_q = ln[2:].strip()
								elif ln.startswith("A:") and cur_q is not None:
									sql_new = ln[2:].strip()
									rows.append({
										"id": f"{db_id}:{tname}:rw:{abs(hash(sql_new))%10**9}",
										"db_id": db_id,
										"question": cur_q,
										"gold_sql": sql_new,
									})
									cur_q = None

						for q0 in qs[: min(len(qs), args.q_per_sql)]:
							_emit_pairs("semantic_rewrite", q0, sql)
							_emit_pairs("numeric_condition_transform", q0, sql)
							_emit_pairs("query_logic_adjustment", q0, sql)
				else:
					for style in [
						"formal",
						"colloquial",
						"imperative",
						"interrogative",
						"descriptive",
						"vague",
						"metaphorical",
						"conversational",
					]:
						q = synthesize_question(f"{tname} {cols[0]}", style, add_distractor=True)
						rows.append({
							"id": f"{db_id}:{tname}:{style}:{abs(hash(sql))%10**9}",
							"db_id": db_id,
							"question": q,
							"gold_sql": sql,
						})

	# Write dataset
	out_jsonl = os.path.join(args.out_root, "fusion_dataset.jsonl")
	with open(out_jsonl, "w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")
	print(f"Wrote {len(rows)} examples to {out_jsonl}")


if __name__ == "__main__":
	main()
