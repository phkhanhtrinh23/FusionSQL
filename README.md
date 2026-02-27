![visitors](https://visitor-badge.laobi.icu/badge?page_id=phkhanhtrinh23.FusionSQL)

# FusionSQL

> Text2SQL evaluation, FusionDataset construction, and shift-aware regression for Text-to-SQL.

![Motivation](figure/introduction_pipeline.png)

#### Citation
```
@inproceedings{fusionsql,
  author       = {Khanh Trinh Pham and Thanh Tam Nguyen and Viet Huynh and Hongzhi Yin and Quoc Viet Hung Nguyen},
  title        = {An Efficient and Effective Evaluator for Text2SQL Models on Unseen and Unlabeled Data},
  booktitle    = {ICDE},
  publisher    = {IEEE},
  year         = {2026},
}
```

## What is it?

FusionSQL provides:

- A portable evaluator that reports execution accuracy for Spider, Spider 2.0, BIRD, SParC, CoSQL, and WikiSQL.
- A pipeline to construct a synthetic FusionDataset of databases, SQLs, and paraphrased questions.
- Shift descriptors (Frechet-like, Mahalanobis, Sliced-Wasserstein) between a target workload and the training set.
- An MLP regressor that learns to predict execution accuracy for a given base model with minimal MAE.

All metrics and reports here are execution-accuracy by design.

![FusionSQL Framework](figure/framework.png)

## Project layout

- `fusion_evaluator/`
	- `data/` dataset loaders and adapters
	- `sql/` SQL normalization and parsing (sqlglot)
	- `exec/` SQLite execution with caching
	- `metrics/` execution
	- `evaluator.py` orchestrator
	- `cli.py` evaluation entrypoint
- `figure/` diagrams
- `outputs/` reports and caches

## Getting Started

### 0) Dependencies

- Python 3.10+
- SQLite (comes with Python stdlib `sqlite3`)
- Recommended: a GPU with CUDA if embedding large datasets

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Torch wheels differ by platform/GPU. If the default install fails or is slow, install a matching build from the official site: [PyTorch Install](https://pytorch.org/get-started/locally/).

### 1) Datasets and expected layout

- Spider / Spider 2.0 / BIRD / SParC / CoSQL
	- Gold and predictions are JSON/JSONL with fields: `question`, `query` (gold) or `prediction` (pred), and `db_id`.
	- Databases are under `db_root/DBID/DB.sqlite`.

- WikiSQL
	- Gold/pred are JSONL; tables file is `tables.jsonl` (`id`, `header`, `rows`).
	- We materialize one SQLite per table into an output directory.

Download links:

- Spider: [Project page](https://yale-lily.github.io/spider)
- Spider 2.0: [Project page](https://spider2-sql.github.io/)
- BIRD: [Project page](https://bird-bench.github.io/)
- SParC: [Project page](https://yale-lily.github.io/sparc)
- CoSQL: [Project page](https://yale-lily.github.io/cosql)
- WikiSQL: [GitHub](https://github.com/salesforce/WikiSQL)

Place gold/pred files accordingly and provide `--db_root` pointing to per-DB folders with `DB.sqlite` for Spider/Spider2/BIRD/SParC/CoSQL.

## FusionDataset

Construct a synthetic, diverse dataset from CSV sources:

```bash
python -m fusion_evaluator.fusion_dataset.cli \
	--sources /path/to/csv_sources ... \
	--out_root outputs/fusion_dataset \
	--max_tables 1000
```

Optional LLM-driven question generation and rewrites (provide both to enable):

```bash
python -m fusion_evaluator.fusion_dataset.cli \
  --sources /path/to/csv_sources \
  --out_root outputs/fusion_dataset \
  --prompts fusion_evaluator/fusion_dataset/prompts.yaml \
  --hf_model Qwen/Qwen2.5-72B-Instruct \
  --device cuda --torch_dtype fp16 \
  --q_per_sql 4 \
  --enable_rewrites --rw_per_cat 2
```

This will:
- acquire CSVs, filter tables (language, structure, near-dup),
- synthesize relational DBs (SQLite under `outputs/fusion_dataset/databases`),
- generate SQLs and paraphrased questions with distractors (LLM-backed if provided),
- optionally produce rewritten Q/A pairs for semantic rewriting, numeric condition transforms, and query logic adjustments,
- write `outputs/fusion_dataset/fusion_dataset.jsonl`.

## FusionSQL

We embed SQLs (or questions) with a Hugging Face model, compute shift descriptors between a training workload and FusionDataset, and fit an MLP to predict execution accuracy.

### 1) Compute embeddings directly

```bash
python -m fusion_evaluator.evaluator_training.cli embed \
	--input outputs/fusion_dataset/fusion_dataset.jsonl \
	--output outputs/fusion_dataset/fusion_emb.npy \
	--model Qwen/Qwen2.5-72B-Instruct \
	--field sql \
	--device cuda \
	--batch_size 8 \
	--max_length 256 \
	--torch_dtype fp16
```

You can pass any compatible encoder from Hugging Face. Common choices include:

- `Qwen/Qwen2.5-72B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`
- `deepseek-ai/deepseek-coder-33b-instruct`
- `XGenerationLab/XiYanSQL-QwenCoder-14B-2502`
- `cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-7B-Instruct`

### 2) Train the regressor (from precomputed embeddings)

```bash
python -m fusion_evaluator.evaluator_training.cli train \
	--source_embeddings path/to/source.npy \
	--target_embeddings path/to/fusion.npy \
	--observed_metric 0.712 \
	--slices 34 \
	--hybrid_swd --pca_k 10 --rand_r 24 --pca_subsample 8192 \
	--out outputs/regressor.joblib
```

### 3) End-to-end training with FusionDataset

```bash
python -m fusion_evaluator.evaluator_training.pipeline train \
    --dataset spider \
    --gold path/to/spider_dev_gold.json \
    --pred path/to/spider_dev_preds.jsonl \
    --db_root path/to/spider/database \
    --fusion_jsonl outputs/fusion_dataset/fusion_dataset.jsonl \
    --exec_accuracy 0.712 \
    --model_name Qwen/Qwen2.5-72B-Instruct \
    --hybrid_swd --pca_k 10 --rand_r 24 --pca_subsample 8192 \
    --slices 34 \
    --out outputs/regressor_spider_qwen.joblib
```

### 4) Inference with FusionSQL

```bash
python -m fusion_evaluator.evaluator_training.pipeline infer \
    --dataset spider \
    --gold path/to/spider_dev_gold.json \
    --pred path/to/spider_dev_preds.jsonl \
    --db_root path/to/spider/database \
    --fusion_jsonl outputs/fusion_dataset/fusion_dataset.jsonl \
    --model_name Qwen/Qwen2.5-72B-Instruct \
    --hybrid_swd --pca_k 10 --rand_r 24 --pca_subsample 8192 \
    --slices 34 \
    --model outputs/regressor_spider_qwen.joblib
```

The regressor predicts execution accuracy for the target workload and chosen base model.

### 5) Sampling-based shift + true execution accuracy (small example)

This helper script repeatedly samples target subsets (e.g., 500 examples), computes shift descriptors between the training workload and each subset, then estimates **true execution accuracy** by generating SQL with a model and executing against the databases. It saves the 100 shift vectors and their accuracies, then fits a 3-layer MLP regressor.

**Example (BIRD dev):**
```bash
python -m fusion_evaluator.evaluator_training.shift_sampling_train \
  --db_root fusion_evaluator/data/bird/dev/dev_databases \
  --source fusion_evaluator/data/spider/sft_spider_train_text2sql.json \
  --target fusion_evaluator/data/bird/sft_bird_dev_text2sql.json \
  --target_limit 500 \
  --num_sets 100 \
  --seed 0 \
  --device cuda --batch_size 8 --torch_dtype fp16
```

**What it does:**
- Builds prompts from **question + schema** (same format as `shift_from_json.py`).
- Uses `Qwen/Qwen2.5-3B-Instruct` to generate SQL.
- Computes execution accuracy by running SQL against SQLite databases under `--db_root`.
- Samples 100 subsets of size 500 (no replacement per subset).
- Computes 100 shift vectors and their 100 accuracies.
- Trains a 3-layer MLP regressor `(256, 128, 64)` on these vectors.

**Outputs:**
- `outputs/shift_samples/shift_samples.npz` containing:
  - `deltas`: `(num_sets, 5)` shift vectors
  - `accuracies`: `(num_sets,)` true execution accuracies
  - `sample_indices`: `(num_sets, target_limit)` indices into the target set
- `outputs/shift_samples/shift_mlp.joblib` trained regressor

**Notes:**
- For Spider, set `--db_root` to `fusion_evaluator/data/spider/database` (or `test_database` if needed).
- If you want to reuse a different generation model, set `--model`.
- To embed with a different model than generation, set `--embed_model`.

<details>
<summary><b>Show additional usage (Spider, Spider2, BIRD, SParC, CoSQL, WikiSQL)</b></summary>

```bash
# Spider
python -m fusion_evaluator.cli \
  --dataset spider \
  --gold path/to/dev_gold.json \
  --pred path/to/predictions.jsonl \
  --db_root path/to/spider/database \
  --out outputs/spider_report.json

# Spider 2.0
python -m fusion_evaluator.cli \
  --dataset spider2 \
  --gold path/to/spider2_gold.json \
  --pred path/to/spider2_preds.jsonl \
  --db_root path/to/spider2/database \
  --out outputs/spider2_report.json

# BIRD
python -m fusion_evaluator.cli \
  --dataset bird \
  --gold path/to/bird_gold.jsonl \
  --pred path/to/bird_preds.jsonl \
  --db_root path/to/bird/database \
  --out outputs/bird_report.json

# SParC
python -m fusion_evaluator.cli \
  --dataset sparc \
  --gold path/to/sparc_dev.json \
  --pred path/to/preds.jsonl \
  --db_root path/to/spider/database \
  --out outputs/sparc_report.json

# CoSQL
python -m fusion_evaluator.cli \
  --dataset cosql \
  --gold path/to/cosql_dev.json \
  --pred path/to/preds.jsonl \
  --db_root path/to/spider/database \
  --out outputs/cosql_report.json

# WikiSQL
python -m fusion_evaluator.cli \
  --dataset wikisql \
  --gold path/to/wikisql_gold.jsonl \
  --pred path/to/wikisql_preds.jsonl \
  --wikisql_tables path/to/tables.jsonl \
  --wikisql_db_out databases/wikisql \
  --out outputs/wikisql_report.json
```

Output:
- JSON report at `--out` with `summary` and per-sample metrics.
- Console table: `ExecAcc`.
</details>

---

If you run into issues or need helper scripts for dataset downloads/materialization, open an issue or reach out.

-----------
**Backup Statistics**

![Visitors](https://margherita-gustatory-zane.ngrok-free.dev/badge/phkhanhtrinh23%2FFusionSQL.svg?ngrok-skip-browser-warning=true)
