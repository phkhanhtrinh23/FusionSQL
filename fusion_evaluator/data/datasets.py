import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from .loaders import align_gold_pred, read_jsonl


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
	if path.lower().endswith(".jsonl"):
		return read_jsonl(path)
	with open(path, "r", encoding="utf-8") as f:
		obj = json.load(f)
		if isinstance(obj, list):
			return obj
		return obj.get("data", [])


# Spider and Spider 2.0 share the same structure: list of objects with question/query/db_id

def load_spider(gold_path: str, pred_path: str) -> List[Dict[str, Any]]:
	gold = _read_json_or_jsonl(gold_path)
	pred = _read_json_or_jsonl(pred_path)
	# Normalize key names to our evaluator schema
	for g in gold:
		if "gold_sql" not in g:
			g["gold_sql"] = g.get("query") or g.get("sql")
	for p in pred:
		if "pred_sql" not in p:
			p["pred_sql"] = p.get("prediction") or p.get("query")
	return align_gold_pred(gold, pred)


def load_spider20(gold_path: str, pred_path: str) -> List[Dict[str, Any]]:
	return load_spider(gold_path, pred_path)


# BIRD: expected similar to Spider with question/query/db_id; accommodate variations

def load_bird(gold_path: str, pred_path: str) -> List[Dict[str, Any]]:
	gold = _read_json_or_jsonl(gold_path)
	pred = _read_json_or_jsonl(pred_path)
	for g in gold:
		if "gold_sql" not in g:
			g["gold_sql"] = g.get("query") or g.get("sql")
	for p in pred:
		if "pred_sql" not in p:
			p["pred_sql"] = p.get("prediction") or p.get("query")
	return align_gold_pred(gold, pred)


# SParC / CoSQL loaders (multi-turn). We flatten per-utterance and align by composite id.

def _flatten_dialog_like(items: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Best-effort flattening for SParC/CoSQL-like structures.

    Produces list of records with fields: id, db_id, question, gold_sql.
    """
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(items):
        db_id = ex.get("db_id") or ex.get("database_id") or ex.get("db")
        root_id = ex.get("id") or ex.get("interaction_id") or str(i)
        # Common key name for turns
        turns: List[Dict[str, Any]] = (
            ex.get("interaction")
            or ex.get("turns")
            or ex.get("utterances")
            or []
        )
        if isinstance(turns, list) and turns:
            for t_idx, turn in enumerate(turns):
                q = turn.get("utterance") or turn.get("question") or turn.get("text") or ex.get("question")
                sql = turn.get("query") or turn.get("sql") or ex.get("query") or ex.get("sql")
                rec_id = f"{root_id}:{t_idx}"
                out.append({
                    "id": rec_id,
                    "db_id": db_id,
                    "question": q,
                    "gold_sql": sql,
                })
        else:
            # Fallback single-turn
            q = ex.get("utterance") or ex.get("question")
            sql = ex.get("query") or ex.get("sql")
            rec_id = str(root_id)
            out.append({
                "id": rec_id,
                "db_id": db_id,
                "question": q,
                "gold_sql": sql,
            })
    return out


def load_sparc(gold_path: str, pred_path: str) -> List[Dict[str, Any]]:
    gold_raw = _read_json_or_jsonl(gold_path)
    pred_raw = _read_json_or_jsonl(pred_path)
    gold = _flatten_dialog_like(gold_raw, source="sparc")
    # Normalize pred to have pred_sql
    pred: List[Dict[str, Any]] = []
    for ex in _flatten_dialog_like(pred_raw, source="sparc"):
        ex = dict(ex)
        if "pred_sql" not in ex:
            ex["pred_sql"] = ex.get("prediction") or ex.get("query") or ex.get("sql")
        pred.append(ex)
    return align_gold_pred(gold, pred)


def load_cosql(gold_path: str, pred_path: str) -> List[Dict[str, Any]]:
    gold_raw = _read_json_or_jsonl(gold_path)
    pred_raw = _read_json_or_jsonl(pred_path)
    gold = _flatten_dialog_like(gold_raw, source="cosql")
    pred: List[Dict[str, Any]] = []
    for ex in _flatten_dialog_like(pred_raw, source="cosql"):
        ex = dict(ex)
        if "pred_sql" not in ex:
            ex["pred_sql"] = ex.get("prediction") or ex.get("query") or ex.get("sql")
        pred.append(ex)
    return align_gold_pred(gold, pred)


# WikiSQL utilities

_AGG_IDX_TO_NAME = {0: "", 1: "MAX", 2: "MIN", 3: "COUNT", 4: "SUM", 5: "AVG"}
_OP_IDX_TO_SQL = {0: "=", 1: ">", 2: "<", 3: ">=", 4: "<=", 5: "!="}


def _sql_from_wikisql(sql_obj: Dict[str, Any], table: Dict[str, Any], table_name: str) -> str:
	cols = table["header"]
	sel_idx = sql_obj.get("sel", 0)
	agg_idx = sql_obj.get("agg", 0)
	agg = _AGG_IDX_TO_NAME.get(agg_idx, "")
	col_name = cols[sel_idx]
	select_expr = f'"{col_name}"'
	if agg:
		select_expr = f"{agg}({select_expr})"
	where = sql_obj.get("conds", [])
	preds: List[str] = []
	for cond in where:
		c_idx, op_idx, val = cond[0], cond[1], cond[2]
		c_name = cols[c_idx]
		op = _OP_IDX_TO_SQL.get(op_idx, "=")
		val_sql = f"'{str(val).replace("'", "''")}'"
		preds.append(f'"{c_name}" {op} {val_sql}')
	where_sql = f" WHERE {' AND '.join(preds)}" if preds else ""
	return f"SELECT {select_expr} FROM \"{table_name}\"{where_sql}"


def _materialize_wikisql_table(db_path: str, table_name: str, header: List[str], rows: List[List[Any]]) -> None:
	os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
	conn = sqlite3.connect(db_path)
	cur = conn.cursor()
	cols_sql = ", ".join([f'"{c}" TEXT' for c in header])
	cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
	cur.execute(f'CREATE TABLE "{table_name}" ({cols_sql})')
	placeholders = ", ".join(["?" for _ in header])
	cur.executemany(
		f'INSERT INTO "{table_name}" VALUES ({placeholders})',
		[tuple(None if r[i] is None else str(r[i]) for i in range(len(header))) for r in rows],
	)
	conn.commit()
	cur.close()
	conn.close()


def build_wikisql_dbs(tables_jsonl_path: str, out_root: str, table_prefix: str = "w") -> Dict[str, str]:
	"""Build one SQLite per WikiSQL table under out_root/<table_id>.sqlite.
	Returns mapping table_id -> sqlite path.
	"""
	with open(tables_jsonl_path, "r", encoding="utf-8") as f:
		tables = [json.loads(line) for line in f if line.strip()]
	id_to_path: Dict[str, str] = {}
	for t in tables:
		tid = t["id"]
		header = t["header"]
		rows = t["rows"]
		db_path = os.path.join(out_root, tid, f"{tid}.sqlite")
		_materialize_wikisql_table(db_path, f"{table_prefix}_{tid}", header, rows)
		id_to_path[tid] = db_path
	return id_to_path


def load_wikisql(
	gold_jsonl_path: str,
	pred_jsonl_path: str,
	tables_jsonl_path: str,
	db_out_root: str,
	table_prefix: str = "w",
) -> List[Dict[str, Any]]:
	"""Load WikiSQL and produce aligned records with textual SQL and db_id.

	- Builds one sqlite DB per table under db_out_root.
	- Generates gold SQL text if only structured SQL is present.
	- Sets db_id to the table_id; uses a single table name per DB as table_prefix_<table_id>.
	"""
	# Build DBs
	build_wikisql_dbs(tables_jsonl_path, db_out_root, table_prefix)

	# Map table_id -> table meta for SQL generation
	table_meta: Dict[str, Dict[str, Any]] = {}
	with open(tables_jsonl_path, "r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			obj = json.loads(line)
			table_meta[obj["id"]] = obj

	gold = read_jsonl(gold_jsonl_path)
	pred = read_jsonl(pred_jsonl_path)
	for g in gold:
		tid = g.get("table_id") or g.get("table") or g.get("db_id")
		g["db_id"] = tid
		# If textual query missing, synthesize from structured SQL
		if not g.get("gold_sql"):
			q = g.get("query") or g.get("sql")
			if isinstance(q, str) and q:
				g["gold_sql"] = q
			else:
				sql_obj = g.get("sql") or {}
				table = table_meta[tid]
				tname = f"{table_prefix}_{tid}"
				g["gold_sql"] = _sql_from_wikisql(sql_obj, table, tname)
	for p in pred:
		tid = p.get("table_id") or p.get("db_id")
		p["db_id"] = tid
		if "pred_sql" not in p:
			p["pred_sql"] = p.get("prediction") or p.get("query")
	return align_gold_pred(gold, pred)
