import argparse
import json
from tabulate import tabulate

from .evaluator import evaluate
from .data.datasets import load_spider, load_spider20, load_bird, load_wikisql, load_sparc, load_cosql
from .data.loaders import read_jsonl


def main() -> None:
	parser = argparse.ArgumentParser(description="FusionSQL Evaluator")
	parser.add_argument("--dataset", choices=["spider", "spider2", "bird", "wikisql", "sparc", "cosql"], required=True)
	parser.add_argument("--gold", required=True, help="Path to gold file (json/jsonl)")
	parser.add_argument("--pred", required=True, help="Path to predictions (json/jsonl)")
	parser.add_argument("--db_root", required=False, help="Root folder containing per-DB sqlite files (Spider/BIRD)")
	parser.add_argument("--tables_json", required=False, help="Path to tables.json (Spider/BIRD optionally)")
	parser.add_argument("--wikisql_tables", required=False, help="Path to WikiSQL tables.jsonl (wikisql only)")
	parser.add_argument("--wikisql_db_out", required=False, help="Output root to build WikiSQL sqlite DBs (wikisql only)")
	parser.add_argument("--out", default="outputs/report.json", help="Path to write JSON report")
	parser.add_argument("--workers", type=int, default=8)
	args = parser.parse_args()


	# Prepare aligned records and pick db_root
	if args.dataset == "spider":
		if not args.db_root:
			parser.error("--db_root is required for spider")
		records = load_spider(args.gold, args.pred)
		db_root = args.db_root
	elif args.dataset == "spider2":
		if not args.db_root:
			parser.error("--db_root is required for spider2")
		records = load_spider20(args.gold, args.pred)
		db_root = args.db_root
	elif args.dataset == "bird":
		if not args.db_root:
			parser.error("--db_root is required for bird")
		records = load_bird(args.gold, args.pred)
		db_root = args.db_root
	elif args.dataset == "wikisql":
		if not args.wikisql_tables or not args.wikisql_db_out:
			parser.error("--wikisql_tables and --wikisql_db_out are required for wikisql")
		records = load_wikisql(args.gold, args.pred, args.wikisql_tables, args.wikisql_db_out)
		db_root = args.wikisql_db_out
	elif args.dataset == "sparc":
		if not args.db_root:
			parser.error("--db_root is required for sparc")
		records = load_sparc(args.gold, args.pred)
		db_root = args.db_root
	else:  # cosql
		if not args.db_root:
			parser.error("--db_root is required for cosql")
		records = load_cosql(args.gold, args.pred)
		db_root = args.db_root

	# Write a temp aligned file to reuse evaluate() path
	aligned_path = args.out + ".aligned.jsonl"
	with open(aligned_path, "w", encoding="utf-8") as f:
		for r in records:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")

	from .evaluator import evaluate as core_eval  # local import to avoid cycles

	report = core_eval(aligned_path, aligned_path, db_root, out_path=args.out, workers=args.workers)

	s = report["summary"]
	table = [[
		round(s["execution_accuracy"], 4),
		s["num_samples"],
	]]
	print(tabulate(table, headers=["ExecAcc", "N"]))


if __name__ == "__main__":
	main()
