import os
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .refine import is_near_duplicate


def cluster_tables(dfs: Dict[str, pd.DataFrame], sim_threshold: int = 80) -> List[List[str]]:
	keys = list(dfs.keys())
	clusters: List[List[str]] = []
	used = set()
	for i, ki in enumerate(keys):
		if ki in used:
			continue
		cluster = [ki]
		used.add(ki)
		headers_i = list(map(str, dfs[ki].columns))
		for j, kj in enumerate(keys):
			if kj in used or kj == ki:
				continue
			headers_j = list(map(str, dfs[kj].columns))
			if is_near_duplicate(headers_i, headers_j, threshold=sim_threshold):
				cluster.append(kj)
				used.add(kj)
		clusters.append(cluster)
	return clusters


def infer_foreign_keys(df_left: pd.DataFrame, df_right: pd.DataFrame, sample: int = 1000) -> List[Tuple[str, str]]:
	# Simple heuristic: exact value overlap of candidate key columns
	fks: List[Tuple[str, str]] = []
	left_cols = list(map(str, df_left.columns))
	right_cols = list(map(str, df_right.columns))
	for lc in left_cols:
		lv = set(df_left[lc].astype(str).head(sample))
		for rc in right_cols:
			rv = set(df_right[rc].astype(str).head(sample))
			if len(lv) > 0 and len(rv) > 0 and len(lv & rv) / max(1, min(len(lv), len(rv))) > 0.8:
				fks.append((lc, rc))
	return fks


def materialize_sqlite(db_dir: str, db_id: str, tables: Dict[str, pd.DataFrame]) -> str:
	os.makedirs(os.path.join(db_dir, db_id), exist_ok=True)
	path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
	conn = sqlite3.connect(path)
	cur = conn.cursor()
	for tname, df in tables.items():
		df_clean = df.copy()
		df_clean.columns = [str(c).strip().lower().replace(" ", "_") for c in df_clean.columns]
		df_clean.to_sql(tname, conn, if_exists="replace", index=False)
	conn.commit()
	cur.close()
	conn.close()
	return path
