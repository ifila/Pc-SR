# Piecewise Symbolic Regression (Pc-SR) – Modular Demo

This repo contains a two-stage pipeline:

- **Stage-1:** CART (Decision Tree Regressor) pruning + per-leaf PySR, alpha selection, and bundle save.
- **Stage-2 (optional):** Post-hoc merging of similar leaves via text embeddings + numeric/structural checks, with retraining on merged clusters.

---

# Repository Migration Notice

The active development and maintenance of this software have been transferred to the official INESC TEC repository:

https://github.com/INESCTEC/pcsr-modular-demo

## Structure

pcsr/
init.py # re-exports public API
DTR_ccp_a_prune_SR_stage1.py # Stage-1 (alpha sweep, per-leaf SR, selection, bundle save)
posthoc_merge.py # Stage-2 (embeddings + numeric/structural checks + retrain on merges)

scripts/
run_pipeline.py # ENTRY POINT – edit paths/settings here and run

data/
synthetic_dataset.csv # demo data (features + 'y' target)

outputs/ # created automatically; holds all artifacts



---

## Quick Start

### 1) Create a fresh environment (Python 3.10 recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
```



### 2) Install dependencies

pip install -r requirements.txt

### 3) Set your dataset path --> Open scripts/run_pipeline.py.

Edit the USER SETTINGS at the top:

DATA_PATH → CSV file (demo: data/synthetic_dataset.csv)

TARGET_COL → target column name (demo: "y")

Optionally set FEATURES (list of column names to use) or leave None to use all non-target columns.

Optionally set DROP_COLS to drop columns before feature selection.

Choose output dir with OUTDIR.

Toggle Stage-2 merging with DO_MERGE = True/False.


---

### 4) Run

--In your IDE: right-click scripts/run_pipeline.py → Run.

--Or from terminal: python scripts/run_pipeline.py

Artifacts will be written under OUTDIR (default outputs/).

### What Stage-1 Saves
Inside OUTDIR/final_bundle/:

chosen_Pc-SR_model_tree.joblib – the selected pruned DecisionTreeRegressor

chosen_Pc-SR_model_leaves.json – per-leaf equations & diagnostics (leaf id, n, complexity, MSE)

chosen_Pc-SR_model_leaves.csv – same as CSV

chosen_Pc-SR_model_config.json – full nested config used

chosen_Pc-SR_model.manifest.json – small index of the above

If cfg.out.save_* flags are enabled, you may also see:

trees_by_alpha/alpha_*/ folders

alpha_summary_raw.csv, alpha_summary.csv (normalized)

optional tree PNGs, leaf CSVs, and prediction CSVs

### Stage-2 (Optional): Post-hoc Merge
If DO_MERGE = True, Stage-2 will:

initialize clusters from Stage-1 leaves,

compare per-cluster equations via text embeddings + structure checks,

validate merges numerically (cross-MSE across clusters),

retrain PySR on the merged clusters each iteration,

stop when no more merges pass the criteria.

Saves to OUTDIR/final_bundle/posthoc_merged/:

merged_Pc-SR_model.json – final cluster equations + initial→final mapping

### Key Flags (where to tweak)
Stage-1 (in DTR_ccp_a_prune_SR_stage1.py, via Config)
cfg.tree

base_depth, min_samples_per_leaf – CART shape

max_points – max α candidates (distinct partitions)

random_state – reproducibility

cfg.pysr

niterations, maxsize – PySR search budget

binary_operators, unary_operators, nested_constraints – search space

deterministic=True, parallelism="serial", random_state – reproducibility

output_directory, run_id_prefix, keep_runs, keep_on_error – PySR I/O

cfg.score

w=(MSE, CVaR, Lp) – composite score weights

cvar_beta – tail risk quantile (e.g., 0.95)

lp_p – Lp aggregation order across leaves

tol_select – choose simplest within +X% of best score

robust_percentile – min/robust-min normalization anchor

cfg.out

outdir – root for artifacts

save_tree_png, save_tree_text, save_leaf_csv, save_preds_csv – extra artifacts

verbose – "QUIET" | "INFO" | "DEBUG"

### Orchestration

val_frac=None – train MSE only; set to e.g. 0.2 for per-leaf train/val

n_jobs – joblib workers for parallel per-leaf fits (keep moderate on Windows)

### Stage-2 (in run_pipeline.py and used by posthoc_merge.py)
STRICT_SIM – cosine similarity to auto-merge (e.g., 0.90)

LOOSE_SIM – if above this but below strict, do numeric cross-MSE test (e.g., 0.80)

MSE_THRESH – average cross-MSE needed to allow merge (e.g., 1e-2)

MAX_ITERS – max merge iterations

(advanced) EMBEDDER_MODEL – sentence-transformer name (default all-MiniLM-L6-v2)

### Requirements
Python 3.10

See requirements.txt for exact versions. Core libs:

numpy, pandas, scikit-learn, matplotlib

sympy, pysr (with Julia backend installed), joblib

sentence-transformers, torch (CPU is fine)

Note (PySR/Julia): PySR uses a Julia backend. If you haven’t used PySR before on this machine, the first run may trigger package compilation (one-time). Follow PySR’s install notes if needed.

Troubleshooting
ModuleNotFoundError: pcsr
Run scripts/run_pipeline.py from the repo root so Python can find the pcsr/ package. In PyCharm, set the Working Directory to the project root in your Run Configuration.




Author: Ifigeneia Lamprianidou

If you use this code in academic work, please cite {remember to include the paper once it is accepted).

