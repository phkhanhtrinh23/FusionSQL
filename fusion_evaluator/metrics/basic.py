from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..exec.engine import SQLiteExecutor, compare_results
from ..sql.normalize import normalize_sql, parse_sql


@dataclass
class BasicMetricsResult:
	exact_match: float
	execution_accuracy: float


def exact_match_metric(pred_sql: str, gold_sql: str, dialect: str = "sqlite") -> float:
	p_ast = parse_sql(pred_sql, dialect)
	g_ast = parse_sql(gold_sql, dialect)
	if p_ast is None or g_ast is None:
		# fall back to normalized string compare
		p = normalize_sql(pred_sql)
		g = normalize_sql(gold_sql)
		return 1.0 if p.strip() == g.strip() and p != "" else 0.0
	# Compare canonicalized SQL from AST to minimize formatting/literal diffs
	p_norm = p_ast.to_sql(dialect=dialect, pretty=False, normalize=True)
	g_norm = g_ast.to_sql(dialect=dialect, pretty=False, normalize=True)
	return 1.0 if p_norm == g_norm and p_norm != "" else 0.0


def execution_accuracy_metric(
	executor: SQLiteExecutor,
	db_id: str,
	pred_sql: str,
	gold_sql: str,
) -> float:
	pred_rows, pred_err = executor.execute(db_id, pred_sql)
	gold_rows, gold_err = executor.execute(db_id, gold_sql)
	if pred_err or gold_err:
		return 0.0
	return 1.0 if compare_results(pred_rows, gold_rows) else 0.0
