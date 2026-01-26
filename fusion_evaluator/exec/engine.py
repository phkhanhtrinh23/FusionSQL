import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple
import time

from fusion_evaluator import sql

from ..data.loaders import find_sqlite_db
from ..sql.normalize import normalize_sql


class SQLiteExecutor:
    def __init__(self, db_root: str, cache_dir: str = "outputs/cache") -> None:
        self.db_root = db_root
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._memory_cache: Dict[str, List[Tuple[Any, ...]]] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def _db_fingerprint(self, db_path: str) -> str:
        try:
            stat = os.stat(db_path)
            payload = f"{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
            return hashlib.sha1(payload).hexdigest()
        except Exception:
            return ""

    def _cache_key(self, db_id: str, sql: str, db_path: Optional[str] = None) -> str:
        norm = normalize_sql(sql)
        fp = self._db_fingerprint(db_path) if db_path else ""
        key = f"{db_id}::{fp}::{norm}".encode("utf-8", errors="ignore")
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

    def execute(self, db_id: str, sql: str, timeout: float = 30.0, timeout_ms: int = 5000) -> Tuple[List[Tuple[Any, ...]], Optional[str]]:
        if not sql or not db_id:
            return [], None
        db_path = find_sqlite_db(self.db_root, db_id)
        if not db_path:
            return [], f"database not found for db_id={db_id}"
        key = self._cache_key(db_id, sql, db_path)
        with self._lock:
            if key in self._memory_cache:
                return self._memory_cache[key], None
            rows_disk = self._load_disk(key)
            if rows_disk is not None:
                self._memory_cache[key] = rows_disk
                return rows_disk, None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
            conn.row_factory = sqlite3.Row
            # Performance/consistency PRAGMAs
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")

            # Progress handler-based timeout in milliseconds
            start_ns = time.perf_counter_ns()
            budget_ns = max(1, timeout_ms) * 1_000_000
            def _progress() -> int:
                if (time.perf_counter_ns() - start_ns) > budget_ns:
                    return 1  # abort
                return 0
            conn.set_progress_handler(_progress, 1000)
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
