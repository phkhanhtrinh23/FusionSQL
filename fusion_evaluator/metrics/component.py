from dataclasses import dataclass
from typing import Dict, Set

from ..sql.normalize import extract_components, parse_sql


@dataclass
class ComponentScores:
	tables_f1: float
	columns_f1: float
	aggs_f1: float
	predicates_f1: float
	micro_f1: float


def _f1(pred: Set[str], gold: Set[str]) -> float:
	if not pred and not gold:
		return 1.0
	if not pred or not gold:
		return 0.0
	tp = len(pred & gold)
	prec = tp / len(pred) if pred else 0.0
	rec = tp / len(gold) if gold else 0.0
	return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def component_f1_metric(pred_sql: str, gold_sql: str, dialect: str = "sqlite") -> ComponentScores:
	pred_ast = parse_sql(pred_sql, dialect)
	gold_ast = parse_sql(gold_sql, dialect)
	pred_comp = extract_components(pred_ast)
	gold_comp = extract_components(gold_ast)

	t_f1 = _f1(pred_comp["tables"], gold_comp["tables"])
	c_f1 = _f1(pred_comp["columns"], gold_comp["columns"])
	a_f1 = _f1(pred_comp["aggs"], gold_comp["aggs"])
	p_f1 = _f1(pred_comp["predicates"], gold_comp["predicates"])

	# Micro-average over all tokens
	pred_all = (
		pred_comp["tables"]
		| pred_comp["columns"]
		| pred_comp["aggs"]
		| pred_comp["predicates"]
	)
	gold_all = (
		gold_comp["tables"]
		| gold_comp["columns"]
		| gold_comp["aggs"]
		| gold_comp["predicates"]
	)
	micro = _f1(pred_all, gold_all)

	return ComponentScores(
		tables_f1=t_f1,
		columns_f1=c_f1,
		aggs_f1=a_f1,
		predicates_f1=p_f1,
		micro_f1=micro,
	)
