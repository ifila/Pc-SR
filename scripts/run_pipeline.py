import os, json, time
import pandas as pd

# --- make project root importable when running this script directly
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pcsr import (
    Config, TreeCfg, PySRCfg, ScoreCfg, OutputCfg,
    run_alpha_sweep, save_final_bundle,
)
from pcsr.posthoc_merge import run_posthoc_merging

# ===== USER SETTINGS (edit these) ============================================
DATA_PATH   = r"C:\Users\mib20154\PycharmProjects\Symbolic_Regression_PySR\Piecewise symbolic regression with DTR\Synthetic dataset validation - modular version\data\synthetic_dataset.csv"   # <-- put your CSV here
TARGET_COL  = "y"                                  # name of your target column
FEATURES    = None                                   # None = use all non-target columns, or list like ["d1","d2",...]
DROP_COLS   = []                                     # optional list of columns to drop before FEATURES selection

OUTDIR      = r"outputs\my_run"                      # where to save artifacts
DO_MERGE    = True                                   # run Stage-2 post-hoc merging

# Post-hoc merge thresholds (only used if DO_MERGE=True)
STRICT_SIM  = 0.90
LOOSE_SIM   = 0.80
MSE_THRESH  = 1e-2
MAX_ITERS   = 5
MERGE_SUBDIR = "merge_stage"
# ============================================================================

def load_dataset_simple(path, target, features=None, drop=None):
    df = pd.read_csv(path)
    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])
    if features is None:
        features = [c for c in df.columns if c != target]
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    return df, X, y, features

def build_cfg(outdir):
    default_constraints = {
        "sin": {"sin": 0, "cos": 0},
        "cos": {"cos": 0, "sin": 0},
        "sqrt": {"sqrt": 0},
        "log": {"log": 0},
    }
    return Config(
        tree=TreeCfg(base_depth=10, min_samples_per_leaf=50, max_points=15, random_state=42),
        pysr=PySRCfg(
            niterations=50, maxsize=10, deterministic=True, parallelism="serial",
            random_state=12345,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "cos", "log", "sqrt", "abs"),
            nested_constraints=default_constraints,
            verbosity=0, progress=False,
            output_directory="pysr_runs",
            run_id_prefix="leafrun",
            keep_runs=False, keep_on_error=False,
        ),
        score=ScoreCfg(w=(0.5, 0.3, 0.2), cvar_beta=0.95, lp_p=4, tol_select=0.05,
                       eps_close_best=0.0, robust_percentile=0.0),
        out=OutputCfg(outdir=outdir, save_tree_png=False, save_tree_text=False,
                      save_leaf_csv=False, save_preds_csv=False, verbose="INFO"),
        val_frac=None,
        n_jobs=6,   # adjust to your machine
    )

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"📂 Loading dataset: {DATA_PATH}")
    df, X, y, feature_names = load_dataset_simple(DATA_PATH, TARGET_COL, FEATURES, DROP_COLS)
    print(f"✅ Data shape: X={X.shape}, y={y.shape}, features={len(feature_names)}")

    cfg = build_cfg(OUTDIR)

    # ---------------- Stage-1: Alpha sweep ----------------
    t0 = time.time()
    result = run_alpha_sweep(X, y, feature_names, cfg)
    t1 = time.time()
    print(f"⏱️ Stage-1 completed in {t1 - t0:.2f}s")
    print(f"   Chosen α: {result['alpha_chosen']:.6f} | "
          f"leaves={int(result['chosen_row']['leaves'])} | depth={int(result['chosen_row']['depth'])}")

    # Save bundle to reuse/inspect later
    final_bundle_dir = os.path.join(OUTDIR, "final_bundle")
    os.makedirs(final_bundle_dir, exist_ok=True)
    save_final_bundle(
        outdir=final_bundle_dir,
        tree=result["tree"],
        leaf_rows=result["leaf_rows"],
        feature_names=feature_names,
        cfg=cfg,
        bundle_name="chosen_Pc-SR_model",
    )

    with open(os.path.join(final_bundle_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"data_path": os.path.abspath(DATA_PATH),
             "target": TARGET_COL,
             "n_samples": int(len(y)),
             "n_features": int(len(feature_names)),
             "feature_names": feature_names},
            f, indent=2
        )

    # ---------------- Stage-2: Post-hoc merging (optional) ----------------
    if DO_MERGE:
        print("🔁 Running Stage-2 post-hoc merging…")
        merged = run_posthoc_merging(
            X=X,
            y=y,
            feature_names=feature_names,
            bundle_dir=final_bundle_dir,
            cfg=cfg,
            threshold_strict=STRICT_SIM,
            threshold_loose=LOOSE_SIM,
            mse_threshold=MSE_THRESH,
            max_iterations=MAX_ITERS,
            # optional:
            embedder_name="all-MiniLM-L6-v2",
            rounding_digits=1,
            save_merged=True,
            merged_bundle_name="merged_Pc-SR_model",
            verbose=False,
        )
        print(f"✅ Stage-2 done. Final clusters: {merged['final_n_clusters']}, "
              f"global MSE: {merged['final_mse']:.6f}")
    else:
        print("ℹ️ Skipping Stage-2 merging (set DO_MERGE=True to enable).")

    print("🎉 Pipeline complete. Artifacts under:", os.path.abspath(OUTDIR))

if __name__ == "__main__":
    main()