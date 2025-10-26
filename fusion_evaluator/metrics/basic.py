from typing import Any, Dict, Optional

from ..exec.engine import SQLiteExecutor, compare_results


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
