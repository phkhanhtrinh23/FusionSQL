import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
	items: List[Dict[str, Any]] = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			items.append(json.loads(line))
	return items


def _index_key(obj: Dict[str, Any]) -> Optional[str]:
	for key in ("id", "qid", "question_id", "utterance_id"):
		if key in obj:
			return str(obj[key])
	return None


def align_gold_pred(
	gold: List[Dict[str, Any]],
	pred: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	"""Align gold and pred records.

	If IDs exist, align by ID. Otherwise align by order.
	Expected fields: db_id, gold_sql/query, pred_sql/prediction.
	"""
	gold_idx: Dict[str, Dict[str, Any]] = {}
	pred_idx: Dict[str, Dict[str, Any]] = {}
	for g in gold:
		key = _index_key(g)
		if key is not None:
			gold_idx[key] = g
	for p in pred:
		key = _index_key(p)
		if key is not None:
			pred_idx[key] = p

	aligned: List[Dict[str, Any]] = []
	if gold_idx and pred_idx:
		keys = [k for k in gold_idx.keys() if k in pred_idx]
		for k in keys:
			g = gold_idx[k]
			p = pred_idx[k]
			aligned.append(_merge_record(g, p))
	else:
		for g, p in zip(gold, pred):
			aligned.append(_merge_record(g, p))
	return aligned


def _merge_record(g: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"id": _index_key(g) or _index_key(p),
		"db_id": g.get("db_id") or p.get("db_id"),
		"question": g.get("question") or p.get("question"),
		"gold_sql": g.get("query") or g.get("sql") or g.get("gold_sql"),
		"pred_sql": p.get("prediction") or p.get("query") or p.get("pred_sql"),
	}


def load_schema(tables_json_path: Optional[str]) -> Dict[str, Any]:
	if not tables_json_path:
		return {}
	with open(tables_json_path, "r", encoding="utf-8") as f:
		return json.load(f)


def find_sqlite_db(db_root: str, db_id: str) -> Optional[str]:
	"""Best-effort discovery of a .sqlite file under db_root/db_id."""
	candidate_dir = os.path.join(db_root, db_id)
	if not os.path.isdir(candidate_dir):
		return None
	for name in os.listdir(candidate_dir):
		if name.lower().endswith(".sqlite") or name.lower().endswith(".db"):
			return os.path.join(candidate_dir, name)
	return None
