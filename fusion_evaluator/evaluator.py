import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .data.loaders import align_gold_pred, load_schema, read_jsonl
from .exec.engine import SQLiteExecutor
from .metrics.basic import execution_accuracy_metric
from .sql.normalize import normalize_sql


@dataclass
class SampleMetrics:
	id: Optional[str]
	db_id: str
	execution_accuracy: float


@dataclass
class Summary:
    execution_accuracy: float
	num_samples: int


def _evaluate_one(executor: SQLiteExecutor, rec: Dict[str, Any]) -> SampleMetrics:
	db_id = rec.get("db_id") or ""
	gold_sql = rec.get("gold_sql") or ""
	pred_sql = rec.get("pred_sql") or ""

    ea = execution_accuracy_metric(executor, db_id, pred_sql, gold_sql)
    return SampleMetrics(id=rec.get("id"), db_id=db_id, execution_accuracy=ea)


def evaluate(
	gold_path: str,
	pred_path: str,
	db_root: str,
	out_path: Optional[str] = None,
	workers: int = 8,
) -> Dict[str, Any]:
	gold = read_jsonl(gold_path)
	pred = read_jsonl(pred_path)
	records = align_gold_pred(gold, pred)
	executor = SQLiteExecutor(db_root)

	results: List[SampleMetrics] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_evaluate_one, executor, rec) for rec in records]
		for fut in as_completed(futs):
			results.append(fut.result())

	# Stable order by id if possible
	results.sort(key=lambda r: (str(r.id) if r.id is not None else "", r.db_id))

	def avg(key: str) -> float:
		vals = [getattr(r, key) for r in results]
		return sum(vals) / len(vals) if vals else 0.0

    summary = Summary(
        execution_accuracy=avg("execution_accuracy"),
        num_samples=len(results),
    )

	report = {
		"summary": asdict(summary),
		"samples": [asdict(r) for r in results],
	}
	if out_path:
		os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(report, f, ensure_ascii=False, indent=2)
	return report
