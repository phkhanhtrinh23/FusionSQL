from typing import Dict, List

import sqlglot
from sqlglot import exp


def simple_select(table: str, columns: List[str]) -> str:
	col = columns[0]
	return f'SELECT "{col}" FROM "{table}"'


def count_rows(table: str) -> str:
	return f'SELECT COUNT(*) FROM "{table}"'


def join_two_tables(t1: str, t2: str, on_col: str) -> str:
	return f'SELECT * FROM "{t1}" JOIN "{t2}" ON "{t1}"."{on_col}" = "{t2}"."{on_col}"'


def group_by_agg(table: str, col_group: str, col_val: str) -> str:
	return f'SELECT "{col_group}", AVG("{col_val}") FROM "{table}" GROUP BY "{col_group}"'


def complex_nested(table: str, col: str) -> str:
	return (
		f'SELECT "{col}" FROM "{table}" WHERE "{col}" IN ('
		f'SELECT "{col}" FROM "{table}" GROUP BY "{col}" HAVING COUNT(*) > 1)'
	)
