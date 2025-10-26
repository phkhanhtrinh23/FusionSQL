# FusionSQL Evaluator

## Install

```
pip install -r requirements.txt
```

## Datasets and expected layout

- Spider / Spider 2.0 / BIRD
	- Gold and predictions are JSON/JSONL with fields: `question`, `query` (gold) or `prediction` (pred), and `db_id`.
	- Databases are under `db_root/DBID/DB.sqlite`.

- WikiSQL
	- Gold/pred are JSONL; WikiSQL tables file is `tables.jsonl` (original format with `id`, `header`, `rows`).
	- This tool materializes one SQLite per table under a specified output directory.

## Usage

Spider:
```
python -m fusion_evaluator.cli \
	--dataset spider \
	--gold path/to/dev_gold.json \
	--pred path/to/predictions.jsonl \
	--db_root path/to/spider/database \
	--out outputs/spider_report.json
```

Spider 2.0:
```
python -m fusion_evaluator.cli \
	--dataset spider2 \
	--gold path/to/spider2_gold.json \
	--pred path/to/spider2_preds.jsonl \
	--db_root path/to/spider2/database \
	--out outputs/spider2_report.json
```

BIRD:
```
python -m fusion_evaluator.cli \
	--dataset bird \
	--gold path/to/bird_gold.jsonl \
	--pred path/to/bird_preds.jsonl \
	--db_root path/to/bird/database \
	--out outputs/bird_report.json
```

WikiSQL:
```
python -m fusion_evaluator.cli \
	--dataset wikisql \
	--gold path/to/wikisql_gold.jsonl \
	--pred path/to/wikisql_preds.jsonl \
	--wikisql_tables path/to/tables.jsonl \
	--wikisql_db_out databases/wikisql \
	--out outputs/wikisql_report.json
```

Adjust fusion weights if desired:
```
  --w_exec 0.5 --w_comp 0.4 --w_exact 0.1
```

## Output

- JSON report at `--out` containing `summary` and per-sample metrics.
- Console table with Exact, ExecAcc, CompF1, Fusion, and N.

## Project layout

- `fusion_evaluator/`
	- `data/` dataset and schema loaders, dataset adapters
	- `sql/` SQL normalization and parsing (sqlglot)
	- `exec/` SQLite execution with caching
	- `metrics/` exact, execution, component, fusion
	- `evaluator.py` orchestrator
	- `cli.py` command-line entry
- `configs/` sample configs
- `tests/` smoke tests
- `outputs/` reports and caches

## Notes
## FusionDataset construction

Build a synthetic, diverse dataset as described in the paper.

```
python -m fusion_evaluator.fusion_dataset.cli \
	--sources /path/to/tablib_csvs /path/to/kaggledbqa_csvs \
	--out_root outputs/fusion_dataset \
	--max_tables 1000
```

This will:
- acquire CSVs, filter tables (language, structure, near-dup),
- synthesize relational DBs (SQLite under `outputs/fusion_dataset/databases`),
- generate SQLs and paraphrased questions with distractors,
- write `outputs/fusion_dataset/fusion_dataset.jsonl`.

## Evaluator training and inference

Compute shift descriptors and fit a small regressor as in the LaTeX.

Training (requires source and target embedding arrays and the observed metric for the target workload):
```
python -m fusion_evaluator.evaluator_training.cli train \
	--source_embeddings path/to/source.npy \
	--target_embeddings path/to/target.npy \
	--observed_metric 0.712 \
	--out outputs/fusionsql_model.joblib
```

Inference (predict on a new workload without labels):
```
python -m fusion_evaluator.evaluator_training.cli infer \
	--model outputs/fusionsql_model.joblib \
	--source_embeddings path/to/source.npy \
	--target_embeddings path/to/unseen.npy
```


- Uses SQLite for portable execution; for Spider/BIRD, place DBs under `db_root/DBID/DB.sqlite`.
- SQL parsing via `sqlglot`; normalization is dialect-aware.
- Parallelization via threads; results cached on disk and in memory.