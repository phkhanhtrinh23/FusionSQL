from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp


def normalize_sql(sql: str, dialect: str = "sqlite") -> str:
	if sql is None:
		return ""
	sql = sql.strip().rstrip(";")
	if not sql:
		return ""
	try:
		tree = sqlglot.parse_one(sql, read=dialect)
		return tree.to_sql(dialect=dialect, pretty=False, normalize=True, strip_comments=True)
	except Exception:
		# Fallback: best-effort transpile -> sqlite
		try:
			return sqlglot.transpile(sql, read=None, write=dialect)[0]
		except Exception:
			return sql


def parse_sql(sql: str, dialect: str = "sqlite") -> Optional[exp.Expression]:
	try:
		return sqlglot.parse_one(sql, read=dialect)
	except Exception:
		return None


def _collect_tables(node: exp.Expression) -> Set[str]:
	tables: Set[str] = set()
	for tbl in node.find_all(exp.Table):
		if tbl.this:
			tables.add(str(tbl.this).lower())
	return tables


def _collect_columns(node: exp.Expression) -> Set[str]:
	cols: Set[str] = set()
	for col in node.find_all(exp.Column):
		cols.add(str(col).lower())
	return cols


def _collect_aggs(node: exp.Expression) -> Set[str]:
	aggs: Set[str] = set()
	for func in node.find_all(exp.Func):
		name = func.name.upper() if hasattr(func, "name") and func.name else None
		if name in {"COUNT", "SUM", "AVG", "MIN", "MAX"}:
			aggs.add(name)
	return aggs


def _collect_predicates(node: exp.Expression) -> Set[str]:
	preds: Set[str] = set()
	for cond in node.find_all(exp.Condition):
		preds.add(cond.__class__.__name__)
	for op in node.find_all(exp.Binary):
		preds.add(op.__class__.__name__)
	return preds


def extract_components(node: Optional[exp.Expression]) -> Dict[str, Set[str]]:
	if node is None:
		return {"tables": set(), "columns": set(), "aggs": set(), "predicates": set()}
	return {
		"tables": _collect_tables(node),
		"columns": _collect_columns(node),
		"aggs": _collect_aggs(node),
		"predicates": _collect_predicates(node),
	}
