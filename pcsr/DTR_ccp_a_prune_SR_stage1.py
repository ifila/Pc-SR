# =============================
# Piecewise SR with CART pruning — consolidated framework (nested config edition)
# =============================
import json, os, joblib
from typing import Dict, Any, Optional, Tuple, List
from sympy import sympify, lambdify
import uuid
import shutil
import sys
import hashlib
import logging
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import matplotlib
# matplotlib.use('TkAgg')  # Set backend first (kept from your original)

from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from pysr import PySRRegressor

# NEW: joblib for parallel per-leaf fits
from joblib import Parallel, delayed


# =============================
# 0. Config & Logging
# =============================
# We use a nested config: cfg.tree, cfg.pysr, cfg.score, cfg.out
# This makes it crystal clear which knobs belong to which subsystem.

@dataclass(frozen=True)
class TreeCfg:
    """Parameters for the CART tree and pruning path."""
    base_depth: int = 10
    min_samples_per_leaf: int = 50
    max_points: int = 15                    # distinct α partitions to evaluate
    random_state: int = 42                  # tree seed (used in all trees)


@dataclass(frozen=True)
class PySRCfg:
    """Parameters for PySR per-leaf symbolic regression."""
    # Search & reproducibility
    niterations: int = 50
    maxsize: int = 10
    deterministic: bool = True              # to make PySR search deterministic, use parallelism='serial' and a fixed random_state
    parallelism: str = "serial"             # 'serial' for determinism; 'multithreading'/'multiprocessing' would break determinism
    random_state: int = 12345               # base seed for PySR; we keep it universal for simplicity

    # Operators & constraints (you can edit at runtime via cfg = replace(cfg, pysr=replace(cfg.pysr, ...)))
    binary_operators: Tuple[str, ...] = ("+", "-", "*", "/")
    unary_operators: Tuple[str, ...]  = ("sin", "cos", "log", "sqrt", "abs")
    nested_constraints: Optional[Dict[str, Dict[str, int]]] = None  # set below in main if you want defaults

    # I/O for PySR
    verbosity: int = 0
    progress: bool = False
    output_directory: str = "pysr_runs"     # parent dir; we still make a unique subdir per leaf
    run_id_prefix: str = "leafrun"          # used to compose unique run_id per leaf
    keep_runs: bool = False                 # keep or delete per-leaf folders after fit
    keep_on_error: bool = True              # keep artifacts if the fit raises (for debugging)

    def as_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for PySRRegressor from this config (run_id set at call site)."""
        return dict(
            model_selection="best",
            niterations=self.niterations,
            binary_operators=list(self.binary_operators),
            unary_operators=list(self.unary_operators),
            nested_constraints=self.nested_constraints,
            maxsize=self.maxsize,
            verbosity=self.verbosity,
            progress=self.progress,
            random_state=self.random_state,
            deterministic=self.deterministic,
            parallelism=self.parallelism,
            output_directory=self.output_directory,
        )


@dataclass(frozen=True)
class ScoreCfg:
    """Scoring and selection hyperparameters for α selection."""
    w: Tuple[float, float, float] = (0.5, 0.3, 0.2)  # (MSE, CVaR, Lp)
    cvar_beta: float = 0.95
    lp_p: int = 4

    # selection rules
    tol_select: float = 0.05          # choose simplest within +5% of best
    eps_close_best: float = 0.00      # tie-break margin for absolute best

    # normalization across α
    robust_percentile: float = 0.0    # 0 = min; try 5 for robust min


@dataclass(frozen=True)
class OutputCfg:
    """Artifacts and logging."""
    outdir: str = "trees_by_alpha"
    save_tree_png: bool = False
    save_tree_text: bool = False
    save_leaf_csv: bool = False
    save_preds_csv: bool = False
    verbose: str = "INFO"             # "QUIET" | "INFO" | "DEBUG"


@dataclass(frozen=True)
class Config:
    """Top-level configuration passed everywhere."""
    tree: TreeCfg = TreeCfg()
    pysr: PySRCfg = PySRCfg()
    score: ScoreCfg = ScoreCfg()
    out: OutputCfg = OutputCfg()

    # Orchestration
    val_frac: Optional[float] = None  # None → use train MSE per leaf; otherwise split train/val per leaf
    n_jobs: int = 1                   # joblib workers across leaves (keep moderate on Windows)


def make_logger(level: str = "INFO") -> logging.Logger:
    level_map = {"QUIET": logging.WARNING, "INFO": logging.INFO, "DEBUG": logging.DEBUG}
    lvl = level_map.get(level.upper(), logging.INFO)
    logger = logging.getLogger("pwsr")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(lvl)
    logger.propagate = False
    return logger


def save_config(cfg: Config, path: str):
    """Persist the entire (nested) configuration next to artifacts."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


'''explicitly save a self-contained “chosen_Pc-SR_model” of the chosen tree + per-leaf equations. It helps with 
reproducibility, deployment, and later audits without needing to re-run the sweep or open dozens of per-α 
folders.'''
def save_final_bundle(
    outdir: str,
    tree: DecisionTreeRegressor,
    leaf_rows: list,                 # from evaluate_tree_with_pysr (chosen tree)
    feature_names: list,
    cfg: Config,
    bundle_name: str = "chosen_Pc-SR_model",
):
    os.makedirs(outdir, exist_ok=True)

    # 1) sklearn tree
    tree_path = os.path.join(outdir, f"{bundle_name}_tree.joblib")
    joblib.dump(tree, tree_path)

    # 2) Leaf manifest (JSON) + 3) CSV (pretty)
    manifest = {
        "feature_names": list(feature_names),
        "notes": "Per-leaf PySR equations for the chosen pruned tree.",
        "leaves": []
    }
    csv_lines = ["leaf_id,n,complexity,mse_val,equation"]

    for r in leaf_rows:
        if r.get("skip", False):
            continue
        entry = {
            "leaf_id": int(r["leaf_id"]),
            "n": int(r["n"]),
            "complexity": int(r["complexity"]),
            "mse_val": float(r["mse_val"]),
            "equation": str(r["equation"]),
        }
        manifest["leaves"].append(entry)
        eq_csv = entry["equation"].replace(",", ";")
        csv_lines.append(f'{entry["leaf_id"]},{entry["n"]},{entry["complexity"]},{entry["mse_val"]:.6g},"{eq_csv}"')

    with open(os.path.join(outdir, f"{bundle_name}_leaves.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(outdir, f"{bundle_name}_leaves.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    # 4) Config next to the bundle
    save_config(cfg, os.path.join(outdir, f"{bundle_name}_config.json"))

    # 5) Small index manifest
    index = {
        "tree_path": tree_path,
        "leaves_json": os.path.join(outdir, f"{bundle_name}_leaves.json"),
        "leaves_csv": os.path.join(outdir, f"{bundle_name}_leaves.csv"),
        "config_json": os.path.join(outdir, f"{bundle_name}_config.json"),
        "note": "Leaf callables are NOT serialized; rebuild from JSON when needed.",
    }
    with open(os.path.join(outdir, f"{bundle_name}.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)



# =============================
# 1. Small utilities
# =============================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_leaves(tree: DecisionTreeRegressor) -> int:
    t = tree.tree_
    return int(np.sum(t.children_left == -1))


def plot_tree_png(tree, feature_names, filename, max_depth=None, figsize=(26, 16), dpi=160):
    ensure_dir(os.path.dirname(filename) or ".")
    plt.figure(figsize=figsize)
    plot_tree(
        tree,
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        precision=3,
        max_depth=max_depth,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


def summarize_top(tree, feature_names, max_depth=3) -> str:
    try:
        return export_text(tree, feature_names=list(feature_names), max_depth=max_depth)
    except Exception:
        return "(export_text failed)"


def _leaf_key(idxs: np.ndarray) -> str:
    """Compact hash for a set of row indices (cache key)."""
    a = np.asarray(idxs, dtype=np.int32)
    return hashlib.md5(a.tobytes()).hexdigest()


def make_callable(eq_str: str, var_names: List[str]):
    """Turn a PySR string equation into a fast numpy callable."""
    expr = sympify(eq_str)
    f = lambdify(var_names, expr, modules="numpy")
    return f


def stitch_predict(tree, leaf_rows, X, y=None):
    """
    Use each leaf's callable to predict on the samples routed to that leaf.
    If a callable is missing, fill with NaN (we can optionally fall back to leaf mean if y given).
    """
    yhat = np.empty(len(X), dtype=float)
    lid = tree.apply(X)
    by_leaf = {int(r["leaf_id"]): r for r in leaf_rows if not r.get("skip", False)}

    for leaf in np.unique(lid):
        idx = np.where(lid == leaf)[0]
        r = by_leaf.get(int(leaf))
        if r is None or "callable" not in r:
            yhat[idx] = np.nan
        else:
            f = r["callable"]
            # f expects variables in the order of feature_names used to build it
            yhat[idx] = np.asarray(f(*[X[idx, j] for j in range(X.shape[1])]), dtype=float)

    # Fallback for NaNs (rare): if y provided, replace with leaf mean y
    if y is not None and np.isnan(yhat).any():
        for leaf in np.unique(lid[np.isnan(yhat)]):
            idx = np.where(lid == leaf)[0]
            yhat[idx] = np.mean(y[idx])
    return yhat



# =============================
# 2. Per-leaf symbolic regression
# =============================
def train_pysr_for_leaf(
    X_leaf: np.ndarray,
    y_leaf: np.ndarray,
    var_names: List[str],
    cfg: Config,
    run_id: str,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit PySR on one leaf and return equation + diagnostics (NO model object returned).
    All PySR params are sourced from cfg.pysr (single source of truth).
    Folder cleanup controlled by cfg.pysr.keep_runs / keep_on_error.
    """
    kwargs = cfg.pysr.as_kwargs()
    kwargs["run_id"] = run_id

    # Isolated subdir per run to avoid collisions (safe for parallel leaves)
    unique_dir = os.path.join(cfg.pysr.output_directory, run_id)
    os.makedirs(unique_dir, exist_ok=True)
    kwargs["output_directory"] = unique_dir

    success = False
    model = None
    try:
        model = PySRRegressor(**kwargs)
        model.fit(X_leaf, y_leaf, variable_names=var_names)

        best = model.get_best()
        eq   = str(best["equation"])
        comp = int(best.get("complexity", best.get("size", 0)))

        # Train MSE
        yhat_tr = model.predict(X_leaf)
        mse_tr  = float(mean_squared_error(y_leaf, yhat_tr))

        # Optional validation MSE
        if X_val is not None and y_val is not None:
            yhat_va = model.predict(X_val)
            mse_va  = float(mean_squared_error(y_val, yhat_va))
        else:
            mse_va  = mse_tr

        success = True
        return {"equation": eq, "complexity": comp, "mse_train": mse_tr, "mse_val": mse_va}

    finally:
        # Help Windows release file handles before removing the folder
        try:
            del model
        except Exception:
            pass

        # Cleanup policy matrix:
        # - success=True  -> delete if keep_runs is False
        # - success=False -> delete only if keep_on_error is False
        try:
            if success:
                if not getattr(cfg.pysr, "keep_runs", False):
                    shutil.rmtree(unique_dir, ignore_errors=True)
            else:
                if not getattr(cfg.pysr, "keep_on_error", True):
                    shutil.rmtree(unique_dir, ignore_errors=True)
        except Exception:
            # Never let cleanup errors crash the caller
            pass



# NEW: small worker to fit a single leaf (used by joblib.Parallel)
def _fit_one_leaf(
    lid: int,
    idxs: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cfg: Config,
) -> Dict[str, Any]:
    """Run PySR for one leaf; returns a lightweight row dict (no model)."""
    Xc = X[idxs]
    yc = y[idxs]

    # Simple, reproducible seeding: universal PySR seed from cfg.pysr.random_state.
    # (If you want per-leaf seeds later, derive from idxs; for now we keep it simple.)

    # Unique run_id per leaf (prevents file collisions even in parallel)
    run_id = f"{cfg.pysr.run_id_prefix}_{lid}_{uuid.uuid4().hex[:6]}"

    if cfg.val_frac is None:
        info = train_pysr_for_leaf(
            Xc, yc, feature_names,
            cfg=cfg,
            run_id=run_id,
        )
    else:
        Xtr, Xva, ytr, yva = train_test_split(
            Xc, yc, test_size=cfg.val_frac, random_state=cfg.pysr.random_state
        )
        info = train_pysr_for_leaf(
            Xtr, ytr, feature_names,
            cfg=cfg,
            run_id=run_id,
            X_val=Xva, y_val=yva,
        )

    row = {
        "leaf_id": int(lid),
        "n": int(len(idxs)),
        "skip": False,
        "reason": "",
        "equation": info["equation"],
        "mse_train": float(info["mse_train"]),
        "mse_val": float(info["mse_val"]),
        "complexity": int(info["complexity"]),
        "cached": False,
        # Note: no 'model' here; we'll compile a callable later in the parent.
    }
    return row



# =============================
# 3. Evaluate a tree with per-leaf PySR
# =============================
def evaluate_tree_with_pysr(
    tree: DecisionTreeRegressor,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cfg: Config,
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
    LOGGER: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Fit PySR per leaf; return per-leaf rows + stitched predictions and squared errors."""
    if LOGGER is None:
        LOGGER = make_logger("INFO")

    # Build leaf clusters
    leaf_ids = tree.apply(X)
    clusters: Dict[int, List[int]] = {}
    for i, lid in enumerate(leaf_ids):
        clusters.setdefault(int(lid), []).append(i)

    rows: List[Dict[str, Any]] = []
    tasks: List[Tuple[int, np.ndarray]] = []

    # 1) Serve cache hits immediately; queue misses for parallel fit
    for lid, idx_list in clusters.items():
        idxs = np.asarray(idx_list, dtype=int)
        key = _leaf_key(idxs)
        if cache is not None and key in cache:
            info = cache[key].copy()
            row = {
                "leaf_id": int(lid),
                "n": int(len(idxs)),
                "skip": False,
                "reason": "",
                "equation": info["equation"],
                "mse_train": float(info["mse_train"]),
                "mse_val": float(info["mse_val"]),
                "complexity": int(info["complexity"]),
                "cached": True,
            }
            rows.append(row)
            LOGGER.debug("[leaf %d] cache-hit | n=%d mse_val=%.6f cx=%s",
                         row["leaf_id"], row["n"], row["mse_val"], row["complexity"])
        else:
            tasks.append((lid, idxs))

    # 2) Fit all missing leaves in parallel (or serial if n_jobs==1)
    if tasks:
        LOGGER.debug("Fitting %d leaves (parallel n_jobs=%d)...", len(tasks), cfg.n_jobs)
        if cfg.n_jobs == 1 or len(tasks) == 1:
            fitted = [_fit_one_leaf(lid, idxs, X, y, feature_names, cfg) for lid, idxs in tasks]
        else:
            # Note: 'loky' => process-based; good on Windows; keep n_jobs moderate.
            fitted = Parallel(n_jobs=cfg.n_jobs, backend="loky", verbose=0)(
                delayed(_fit_one_leaf)(lid, idxs, X, y, feature_names, cfg) for lid, idxs in tasks
            )
        rows.extend(fitted)

        # Update cache (store only lightweight info)
        if cache is not None:
            for (_, idxs), r in zip(tasks, fitted):
                cache[_leaf_key(idxs)] = {
                    "equation": r["equation"],
                    "mse_train": r["mse_train"],
                    "mse_val": r["mse_val"],
                    "complexity": r["complexity"],
                }

    # 3) Keep row order stable by leaf_id
    rows = sorted(rows, key=lambda r: r["leaf_id"])

    # Compile callables from equation strings (now every row has r["callable"])
    for r in rows:
        if not r.get("skip", False):
            r["callable"] = make_callable(r["equation"], feature_names)

    # stitched predictions (fallback to leaf mean handled inside)
    yhat = stitch_predict(tree, rows, X, y=y)
    err2 = (y - yhat) ** 2
    return rows, yhat, err2



# =============================
# 4. Global metrics, normalization, selection
# =============================
def cvar_tail(err2: np.ndarray, beta: float = 0.95) -> float:
    q = np.quantile(err2, beta)
    tail = err2[err2 >= q]
    return float(tail.mean() if tail.size else err2.mean())


def leaf_lp_mean(rows: List[Dict[str, Any]], p: int = 4) -> float:
    mses = np.array([r["mse_val"] for r in rows if not r.get("skip", False)], dtype=float)
    ns = np.array([r["n"] for r in rows if not r.get("skip", False)], dtype=float)
    if mses.size == 0:
        return np.inf
    w = ns / ns.sum()
    return float((w * (mses ** p)).sum()) ** (1.0 / p)


def composite_score(mse: float, cvar95: float, lp_mean: float, w: Tuple[float, float, float]) -> float:
    wm, wc, wl = w
    s = wm + wc + wl
    if s <= 0:
        raise ValueError("All weights are zero.")
    wm, wc, wl = wm / s, wc / s, wl / s
    return float(wm * mse + wc * cvar95 + wl * lp_mean)


def normalize_ratio_to_best(
    df: pd.DataFrame, mapping: Dict[str, str], robust_percentile: float = 0.0, eps: float = 1e-12
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Normalize each metric to its 'best' (min or robust min) for dimensionless comparison."""
    df_out = df.copy()
    stars: Dict[str, float] = {}
    for raw_col, norm_col in mapping.items():
        vals = df_out[raw_col].astype(float).values
        star = float(max(np.percentile(vals, robust_percentile), eps))  # 0 → min; 5 → robust min
        df_out[norm_col] = df_out[raw_col] / star
        stars[raw_col] = star
    return df_out, stars


def select_effective_alphas(X: np.ndarray, y: np.ndarray, cfg: Config, LOGGER: logging.Logger) -> np.ndarray:
    """Keep only α's that actually change the partition (leaf memberships)."""
    base = DecisionTreeRegressor(
        max_depth=cfg.tree.base_depth,
        min_samples_leaf=cfg.tree.min_samples_per_leaf,
        splitter="best",
        max_features=None,
        random_state=cfg.tree.random_state,
        ccp_alpha=0.0,
    ).fit(X, y)
    path = base.cost_complexity_pruning_path(X, y)
    alphas = np.sort(np.unique(path.ccp_alphas))[::-1]  # large→small (simple→complex)


    kept: List[float] = []
    last_key = None
    for a in alphas:
        t = DecisionTreeRegressor(
            max_depth=cfg.tree.base_depth,
            min_samples_leaf=cfg.tree.min_samples_per_leaf,
            splitter="best",
            max_features=None,
            random_state=cfg.tree.random_state,
            ccp_alpha=a,
        ).fit(X, y)
        key = tuple(t.apply(X).tolist())
        if key != last_key:
            kept.append(float(a))
            last_key = key
            if len(kept) >= cfg.tree.max_points:
                break
    LOGGER.info("Effective α's (simple→complex): %s", np.round(kept, 6).tolist())

    return np.array(kept, dtype=float)


def run_alpha_sweep(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cfg: Config,
    verbose: Optional[bool] = None,
) -> Dict[str, Any]:
    """Master routine: sweep α, fit per-leaf SR, compute metrics, select α, retrain final tree."""
    LOGGER = make_logger(cfg.out.verbose if verbose is None else ("DEBUG" if verbose else "QUIET"))
    ensure_dir(cfg.out.outdir)

    # 1) α candidates that actually change routing
    alphas = select_effective_alphas(X, y, cfg, LOGGER=LOGGER)

    # 2) Evaluate each α: build tree, per-leaf PySR, global metrics (RAW)
    rows_all: List[Dict[str, Any]] = []
    cache: Dict[str, Dict[str, Any]] = {}  # cache keyed by leaf membership (index-set hash)
    for a in alphas:
        tree = DecisionTreeRegressor(
            max_depth=cfg.tree.base_depth,
            min_samples_leaf=cfg.tree.min_samples_per_leaf,
            splitter="best",
            max_features=None,
            random_state=cfg.tree.random_state,
            ccp_alpha=a,
        ).fit(X, y)

        leaves = count_leaves(tree)
        depth = tree.get_depth()

        rows, yhat, err2 = evaluate_tree_with_pysr(
            tree, X, y, feature_names, cfg, cache=cache, LOGGER=LOGGER
        )
        mse = float(err2.mean())
        cvar = cvar_tail(err2, beta=cfg.score.cvar_beta)
        lp = leaf_lp_mean(rows, p=cfg.score.lp_p)

        LOGGER.info("α=%.6f | leaves=%3d | depth=%2d | MSE=%.6f | CVaR95=%.6f | LpLeaf=%.6f",
                    a, leaves, depth, mse, cvar, lp)

        # optional artifacts per α
        a_dir = os.path.join(cfg.out.outdir, f"alpha_{a:.6f}")
        if cfg.out.save_tree_png:
            plot_tree_png(tree, feature_names, os.path.join(a_dir, "tree_full.png"))
            plot_tree_png(tree, feature_names, os.path.join(a_dir, "tree_depth3.png"), max_depth=3)
        if cfg.out.save_tree_text:
            ensure_dir(a_dir)
            with open(os.path.join(a_dir, "tree_top.txt"), "w", encoding="utf-8") as f:
                f.write(summarize_top(tree, feature_names, max_depth=3))
        if cfg.out.save_leaf_csv:
            ensure_dir(a_dir)
            # Drop non-serializable items like callables:
            leaf_df = pd.DataFrame([{k: (None if k == "callable" else v) for k, v in r.items()} for r in rows])
            leaf_df.to_csv(os.path.join(a_dir, "leaf_diagnostics.csv"), index=False)
        if cfg.out.save_preds_csv:
            ensure_dir(a_dir)
            pd.DataFrame({"y": y, "yhat": yhat, "err2": err2}).to_csv(
                os.path.join(a_dir, "predictions.csv"), index=False
            )

        rows_all.append(
            {"alpha": a, "leaves": leaves, "depth": depth, "mse": mse, "cvar95": cvar, "lp_mean": lp}
        )

    raw = pd.DataFrame(rows_all).sort_values("alpha", ascending=False).reset_index(drop=True)

    # 3) Normalize across α → compute composite score (NORMALIZED SUMMARY)
    norm, stars = normalize_ratio_to_best(
        raw,
        mapping={"mse": "mse_norm", "cvar95": "cvar_norm", "lp_mean": "lp_norm"},
        robust_percentile=cfg.score.robust_percentile,
    )
    wm, wc, wl = cfg.score.w
    norm["score"] = wm * norm["mse_norm"] + wc * norm["cvar_norm"] + wl * norm["lp_norm"]

    # save both raw & normalized
    raw.to_csv(os.path.join(cfg.out.outdir, "alpha_summary_raw.csv"), index=False)
    norm.to_csv(os.path.join(cfg.out.outdir, "alpha_summary.csv"), index=False)
    save_config(cfg, os.path.join(cfg.out.outdir, "config.json"))
    LOGGER.info("Normalization stars (best values): %s", stars)

    # 4) Select α (absolute best + simplest within tol)
    best_score = norm["score"].min()
    best_pool = norm[norm["score"] <= best_score * (1.0 + cfg.score.eps_close_best)].copy()
    best_row = best_pool.sort_values(["leaves", "depth", "alpha"], ascending=[True, True, False]).iloc[0]

    tol_pool = norm[norm["score"] <= best_score * (1.0 + cfg.score.tol_select)].copy()
    chosen_row = tol_pool.sort_values(["leaves", "depth", "alpha"], ascending=[True, True, False]).iloc[0]

    alpha_best = float(best_row["alpha"])
    alpha_chosen = float(chosen_row["alpha"])
    LOGGER.info("Best α=%.6f (score=%.6f, leaves=%d, depth=%d)",
                alpha_best, float(best_row["score"]), int(best_row["leaves"]), int(best_row["depth"]))
    LOGGER.info("Chosen α (simplest ≤ %.0f%% of best)=%.6f (score=%.6f, leaves=%d, depth=%d, Δ=%.1f%%)",
                100 * cfg.score.tol_select,
                alpha_chosen,
                float(chosen_row["score"]),
                int(chosen_row["leaves"]),
                int(chosen_row["depth"]),
                100.0 * (float(chosen_row["score"]) / best_score - 1.0),
               )

    # 5) Retrain final tree (use chosen α by default) and re-fit per-leaf SR once more (no cache)
    final_tree = DecisionTreeRegressor(
        max_depth=cfg.tree.base_depth,
        min_samples_leaf=cfg.tree.min_samples_per_leaf,
        splitter="best",
        max_features=None,
        random_state=cfg.tree.random_state,
        ccp_alpha=alpha_chosen,
    ).fit(X, y)
    final_rows, final_yhat, final_err2 = evaluate_tree_with_pysr(
        final_tree, X, y, feature_names, cfg, cache=None, LOGGER=LOGGER
    )

    return {
        "summary_raw": raw,
        "summary_norm": norm,
        "stars": stars,
        "alpha_best": alpha_best,
        "alpha_chosen": alpha_chosen,
        "best_row": best_row.to_dict(),
        "chosen_row": chosen_row.to_dict(),
        "tree": final_tree,
        "leaf_rows": final_rows,
        "yhat": final_yhat,
        "err2": final_err2,
        "LOGGER": LOGGER,
    }



# =============================
# 5. Plotting helpers (composite score vs leaves)
# =============================
def curve_for_weights(df_norm: pd.DataFrame, w: Tuple[float, float, float]) -> pd.DataFrame:
    """Build 'score vs leaves' curve for a given weight w using normalized columns."""
    w_mse, w_cvar, w_lp = w
    tmp = df_norm.copy()
    tmp["score_w"] = (w_mse * tmp["mse_norm"].values +
                      w_cvar * tmp["cvar_norm"].values +
                      w_lp * tmp["lp_norm"].values)
    best_per_leaves = (
        tmp.sort_values(["leaves", "score_w"]).groupby("leaves", as_index=False).first()
    )
    return best_per_leaves[["leaves", "score_w", "alpha", "depth"]].sort_values("leaves")


def select_alpha_from_norm(df_norm: pd.DataFrame, w=(0.5, 0.3, 0.2), tol=0.05, eps_close=0.0):
    """Selector working on normalized table (mirrors run_alpha_sweep selection)."""
    df = df_norm.copy()
    df["score_w"] = (w[0] * df["mse_norm"].values +
                     w[1] * df["cvar_norm"].values +
                     w[2] * df["lp_norm"].values)
    # best
    best = df["score_w"].min()
    best_pool = df[df["score_w"] <= best * (1.0 + eps_close)].copy()
    best_row = best_pool.sort_values(["leaves", "depth", "alpha"], ascending=[True, True, False]).iloc[0]
    # chosen
    tol_pool = df[df["score_w"] <= best * (1.0 + tol)].copy()
    chosen_row = tol_pool.sort_values(["leaves", "depth", "alpha"], ascending=[True, True, False]).iloc[0]
    rel_gap_pct = 100.0 * (float(chosen_row["score_w"]) / best - 1.0)
    return {
        "best": {"alpha": float(best_row["alpha"]), "leaves": int(best_row["leaves"]),
                 "depth": int(best_row["depth"]), "score": float(best_row["score_w"])},
        "chosen": {"alpha": float(chosen_row["alpha"]), "leaves": int(chosen_row["leaves"]),
                   "depth": int(chosen_row["depth"]), "score": float(chosen_row["score_w"])},
        "rel_gap_pct": rel_gap_pct
    }


def plot_composite_curves(summary_norm: pd.DataFrame, cfg: Config, annotate_best=True):
    """Plot composite score vs leaves for a few weightings (normalized metrics)."""
    # fixed colors/styles
    styles = [
        {"w": (1.00, 0.00, 0.00), "label": "MSE-only",  "color": "blue",  "ls": "-"},
        {"w": (0.00, 1.00, 0.00), "label": "CVaR-only", "color": "red",   "ls": "-"},
        {"w": (0.00, 0.00, 1.00), "label": "Lp-only",   "color": "green", "ls": "-"},
        {"w": cfg.score.w,        "label": "Baseline",  "color": "orange","ls": "--"},
    ]
    tol = cfg.score.tol_select

    # annotation offsets for "best" markers (tuned layout)
    anno_best = {
        "MSE-only":  (-1,  45,  "left",  "bottom"),
        "CVaR-only": (-25, 12,  "right", "bottom"),
        "Lp-only":   (10,  45,  "left",  "bottom"),
        "Baseline":  (-23, 85,  "left",  "bottom"),
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for st in styles:
        curve = curve_for_weights(summary_norm, st["w"])
        ax.plot(curve["leaves"], curve["score_w"],
                marker="o", markersize=4, linewidth=1.9,
                color=st["color"], linestyle=st["ls"], label="_nolegend_")

        sel = select_alpha_from_norm(summary_norm, w=st["w"], tol=tol, eps_close=0.0)

        # best (hollow circle)
        ax.scatter([sel["best"]["leaves"]], [sel["best"]["score"]],
                   s=80, facecolors="none", edgecolors="k", zorder=3, label="_nolegend_")

        # chosen (white diamond)
        ax.scatter([sel["chosen"]["leaves"]], [sel["chosen"]["score"]],
                   s=90, marker="D", facecolors="white", edgecolors="k", zorder=3, label="_nolegend_")

        if annotate_best:
            dx, dy, ha, va = anno_best.get(st["label"], (10, 10, "left", "bottom"))
            txt = ax.annotate(f"{st['label']} — best\nscore={sel['best']['score']:.3f}",
                              xy=(sel["best"]["leaves"], sel["best"]["score"]),
                              xycoords="data",
                              xytext=(dx, dy), textcoords="offset points",
                              fontsize=9, color=st["color"], ha=ha, va=va,
                              arrowprops=dict(arrowstyle="->", lw=0.9, color=st["color"]))
            # optional halo for readability:
            # import matplotlib.patheffects as pe
            # txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Composite score (dimensionless)")

    curve_handles = [
        Line2D([], [], color=st["color"], linestyle=st["ls"], marker="o",
               label=f"{st['label']}  w={tuple(round(x,2) for x in st['w'])}")
        for st in styles
    ]
    best_proxy = Line2D([], [], linestyle="None", marker="o",
                        markerfacecolor="none", markeredgecolor="k",
                        label="best α (global min)")
    chosen_proxy = Line2D([], [], linestyle="None", marker="D",
                          markerfacecolor="white", markeredgecolor="k",
                          label=f"simplest ≤{int(tol*100)}% of best")
    ax.legend(handles=curve_handles + [best_proxy, chosen_proxy], loc="best", frameon=True)

    ax.margins(x=0.03, y=0.06)
    plt.tight_layout()
    return fig, ax


