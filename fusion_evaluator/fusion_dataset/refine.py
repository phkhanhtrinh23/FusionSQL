import hashlib
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from langdetect import detect
from rapidfuzz import fuzz


def is_english_text(text: str, p_threshold: float = 0.5) -> bool:
	try:
		lang = detect(text)
		return lang == "en"
	except Exception:
		return False


def structural_ok(df: pd.DataFrame, min_rows: int = 5, min_cols: int = 5) -> bool:
	return df.shape[0] >= min_rows and df.shape[1] >= min_cols


def header_hash(headers: List[str]) -> str:
	canon = "\t".join([h.strip().lower() for h in headers])
	return hashlib.sha1(canon.encode("utf-8", errors="ignore")).hexdigest()


def is_near_duplicate(a_headers: List[str], b_headers: List[str], threshold: int = 90) -> bool:
	a = "|".join([h.strip().lower() for h in a_headers])
	b = "|".join([h.strip().lower() for h in b_headers])
	return fuzz.token_sort_ratio(a, b) >= threshold


def semantic_richness_score(df: pd.DataFrame) -> float:
	# Heuristic proxy for conceptual richness: unique tokens per cell and column entropy
	try:
		text_cells = df.astype(str).fillna("")
		tokens = set()
		for v in text_cells.to_numpy().flatten():
			for t in str(v).split():
				tokens.add(t.lower())
		return min(1.0, len(tokens) / max(1, df.shape[0] * df.shape[1]))
	except Exception:
		return 0.0
