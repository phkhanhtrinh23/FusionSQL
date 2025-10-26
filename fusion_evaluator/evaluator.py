import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .data.loaders import align_gold_pred, load_schema, read_jsonl
from .exec.engine import SQLiteExecutor
from .metrics.basic import BasicMetricsResult, exact_match_metric, execution_accuracy_metric
from .metrics.component import ComponentScores, component_f1_metric
from .metrics.fusion import FusionWeights, fusion_score
from .sql.normalize import normalize_sql


@dataclass
class SampleMetrics:
	id: Optional[str]
	db_id: str
	exact_match: float
	execution_accuracy: float
	component_f1: float
	fusion: float


@dataclass
class Summary:
	exact_match: float
	execution_accuracy: float
	component_f1: float
	fusion: float
	num_samples: int


def _evaluate_one(executor: SQLiteExecutor, rec: Dict[str, Any], weights: FusionWeights) -> SampleMetrics:
	db_id = rec.get("db_id") or ""
	gold_sql = rec.get("gold_sql") or ""
	pred_sql = rec.get("pred_sql") or ""

	em = exact_match_metric(pred_sql, gold_sql)
	ea = execution_accuracy_metric(executor, db_id, pred_sql, gold_sql)
	comp = component_f1_metric(pred_sql, gold_sql).micro_f1
	fus = fusion_score(ea, comp, em, weights)
	return SampleMetrics(id=rec.get("id"), db_id=db_id, exact_match=em, execution_accuracy=ea, component_f1=comp, fusion=fus)


def evaluate(
	gold_path: str,
	pred_path: str,
	db_root: str,
	out_path: Optional[str] = None,
	workers: int = 8,
	weights: FusionWeights = FusionWeights(),
) -> Dict[str, Any]:
	gold = read_jsonl(gold_path)
	pred = read_jsonl(pred_path)
	records = align_gold_pred(gold, pred)
	executor = SQLiteExecutor(db_root)

	results: List[SampleMetrics] = []
	with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
		futs = [pool.submit(_evaluate_one, executor, rec, weights) for rec in records]
		for fut in as_completed(futs):
			results.append(fut.result())

	# Stable order by id if possible
	results.sort(key=lambda r: (str(r.id) if r.id is not None else "", r.db_id))

	def avg(key: str) -> float:
		vals = [getattr(r, key) for r in results]
		return sum(vals) / len(vals) if vals else 0.0

	summary = Summary(
		exact_match=avg("exact_match"),
		execution_accuracy=avg("execution_accuracy"),
		component_f1=avg("component_f1"),
		fusion=avg("fusion"),
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
