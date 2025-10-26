import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from ..data.loaders import find_sqlite_db
from ..sql.normalize import normalize_sql


class SQLiteExecutor:
	def __init__(self, db_root: str, cache_dir: str = "outputs/cache") -> None:
		self.db_root = db_root
		self.cache_dir = cache_dir
		self._lock = threading.Lock()
		self._memory_cache: Dict[str, List[Tuple[Any, ...]]] = {}
		os.makedirs(self.cache_dir, exist_ok=True)

	def _cache_key(self, db_id: str, sql: str) -> str:
		norm = normalize_sql(sql)
		key = f"{db_id}::{norm}".encode("utf-8", errors="ignore")
		return hashlib.sha1(key).hexdigest()

	def _cache_path(self, key: str) -> str:
		return os.path.join(self.cache_dir, f"{key}.json")

	def _load_disk(self, key: str) -> Optional[List[Tuple[Any, ...]]]:
		path = self._cache_path(key)
		if not os.path.isfile(path):
			return None
		try:
			with open(path, "r", encoding="utf-8") as f:
				rows = json.load(f)
				return [tuple(r) for r in rows]
		except Exception:
			return None

	def _save_disk(self, key: str, rows: List[Tuple[Any, ...]]) -> None:
		path = self._cache_path(key)
		try:
			with open(path, "w", encoding="utf-8") as f:
				json.dump([list(r) for r in rows], f)
		except Exception:
			pass

	def execute(self, db_id: str, sql: str, timeout: float = 30.0) -> Tuple[List[Tuple[Any, ...]], Optional[str]]:
		if not sql or not db_id:
			return [], None
		key = self._cache_key(db_id, sql)
		with self._lock:
			if key in self._memory_cache:
				return self._memory_cache[key], None
			rows_disk = self._load_disk(key)
			if rows_disk is not None:
				self._memory_cache[key] = rows_disk
				return rows_disk, None

		db_path = find_sqlite_db(self.db_root, db_id)
		if not db_path:
			return [], f"database not found for db_id={db_id}"

		try:
			conn = sqlite3.connect(db_path, timeout=timeout)
			conn.row_factory = sqlite3.Row
			cur = conn.cursor()
			cur.execute(sql)
			rows = cur.fetchall()
			cur.close()
			conn.close()
			rows_tuples = [tuple(row) for row in rows]
			with self._lock:
				self._memory_cache[key] = rows_tuples
				self._save_disk(key, rows_tuples)
			return rows_tuples, None
		except Exception as e:
			return [], str(e)


def compare_results(a: List[Tuple[Any, ...]], b: List[Tuple[Any, ...]]) -> bool:
	"""Order-agnostic, type-tolerant equality."""
	def normalize_cell(x: Any) -> Any:
		if isinstance(x, float):
			return round(x, 6)
		return x

	def normalize_rows(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
		return sorted([tuple(normalize_cell(c) for c in r) for r in rows])

	return normalize_rows(a) == normalize_rows(b)
