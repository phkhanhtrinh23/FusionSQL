import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .acquire import discover_csvs
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
				for style in ["formal", "colloquial", "imperative", "interrogative"]:
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
