# pcsr/posthoc_merge.py
"""
Post-hoc merging of per-leaf symbolic models produced by Stage-1.

Inputs
------
- X, y, feature_names: dataset used in Stage-1 (same ordering)
- bundle_dir: path to Stage-1 final bundle (contains *_tree.joblib, *_leaves.json, *_config.json)
- cfg: pcsr.Config (we reuse cfg.pysr for retraining)
- thresholds & options (see run_posthoc_merging signature)

Outputs
-------
dict with:
  - "tree": original sklearn tree (routing unchanged)
  - "cluster_id": np.ndarray [n_samples] with merged cluster ids
  - "cluster_models": dict[int -> PySRRegressor]
  - "cluster_equations": dict[int -> str]
  - "history": list of iterations (sizes, MSE, eqs)
  - "mapping_init_to_final": dict[int -> int]
  - "merged_manifest_path": optional path to saved merged bundle (if save_merged=True)

Notes
-----
Merging is *post-hoc*: we keep the same routing (same leaf membership),
but unify clusters if their expressions are sufficiently similar/compatible,
then **retrain** a single PySR model on the union.
"""

from __future__ import annotations
import os, json, warnings
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

import sympy as sp
from sympy import sympify, Float, Rational, simplify, preorder_traversal, lambdify, symbols

from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.metrics import mean_squared_error

from sentence_transformers import SentenceTransformer, util
from pysr import PySRRegressor

from .DTR_ccp_a_prune_SR_stage1 import (
    Config, make_callable
)

# -------------------------
# Utilities: I/O
# -------------------------
def _load_stage1_bundle(bundle_dir: str):
    """Load Stage-1 artifacts."""
    mani = os.path.join(bundle_dir, "chosen_Pc-SR_model.manifest.json")
    if not os.path.exists(mani):
        raise FileNotFoundError(f"Bundle manifest not found: {mani}")
    with open(mani, "r", encoding="utf-8") as f:
        m = json.load(f)

    tree = joblib.load(m["tree_path"])
    with open(m["leaves_json"], "r", encoding="utf-8") as f:
        leaves = json.load(f)
    with open(m["config_json"], "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    feature_names = leaves.get("feature_names", None)

    return tree, leaves["leaves"], feature_names, cfg_dict, m

def _initial_leaf_clusters(tree: DecisionTreeRegressor, X: np.ndarray) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Return leaf id per sample and dict leaf_id->row indices."""
    leaf_ids = tree.apply(X).astype(int)
    clusters: Dict[int, List[int]] = {}
    for i, lid in enumerate(leaf_ids):
        clusters.setdefault(lid, []).append(i)
    return leaf_ids, clusters

def _compact_id_map(ids: List[int]) -> Dict[int, int]:
    """Map arbitrary ids to compact 0..K-1 order-preserving."""
    uniq = sorted(set(ids))
    return {old: new for new, old in enumerate(uniq)}

# -------------------------
# Expression preprocessing & cache
# -------------------------
def _normalize_expr(expr: sp.Expr, rounding_digits: int = 1) -> sp.Expr:
    """Normalize divisions by floats (1/x ≈ c) and round floats; simplify."""
    def normalize_divisions(e):
        repl = {}
        for sub in preorder_traversal(e):
            if isinstance(sub, sp.Pow):
                base, exp = sub.args
                if exp == -1 and isinstance(base, (Float, Rational)) and base != 0:
                    inv = round(1 / float(base), rounding_digits)
                    repl[sub] = Float(inv)
            elif isinstance(sub, sp.Mul):
                new_args = []
                changed = False
                for arg in sub.args:
                    if isinstance(arg, sp.Pow) and arg.exp == -1 and isinstance(arg.base, (Float, Rational)):
                        inv = round(1 / float(arg.base), rounding_digits)
                        new_args.append(Float(inv))
                        changed = True
                    else:
                        new_args.append(arg)
                if changed:
                    repl[sub] = sp.Mul(*new_args)
        return e.xreplace(repl)

    e2 = normalize_divisions(expr)
    e2 = e2.xreplace({n: round(float(n), rounding_digits) for n in e2.atoms(Float)})
    e2 = simplify(e2)
    # normalize symbol usage to 1*xi to stabilize tree walk
    e2 = e2.replace(lambda z: isinstance(z, sp.Symbol), lambda z: sp.Mul(Float(1.0), z, evaluate=True))
    return e2

def _extract_vars_ops(expr: sp.Expr) -> Tuple[List[str], List[str]]:
    vars_ = sorted({str(s) for s in expr.free_symbols})
    ops = set()
    for node in preorder_traversal(expr):
        if isinstance(node, sp.Symbol):
            continue
        if hasattr(node, "func"):
            name = getattr(node.func, "__name__", None)
            if name == "Float" or name == "Integer":
                ops.add("Number")
            elif name:
                ops.add(name)
    return vars_, sorted(ops)

class ExprCache:
    """Cache normalized structure for equation strings to avoid repeated SymPy work."""
    def __init__(self, var_names: List[str], rounding_digits: int = 1):
        self._symbols = dict(zip(var_names, symbols(var_names)))
        self._round = rounding_digits
        self._cache: Dict[str, Tuple[str, List[str], List[str], sp.Expr]] = {}

    def get(self, eq_str: str) -> Tuple[str, List[str], List[str], sp.Expr]:
        if eq_str in self._cache:
            return self._cache[eq_str]
        expr = sympify(eq_str, locals=self._symbols)
        norm = _normalize_expr(expr, self._round)
        vars_, ops = _extract_vars_ops(norm)
        norm_str = str(norm)
        self._cache[eq_str] = (norm_str, vars_, ops, norm)
        return self._cache[eq_str]

# -------------------------
# Embedding model (lazy)
# -------------------------
def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        warnings.warn(f"Could not load SentenceTransformer ({e}); embedding similarity disabled.")
        return None

# -------------------------
# Train PySR on a cluster
# -------------------------
def _train_symbolic_model(X: np.ndarray, y: np.ndarray, feature_names: List[str], pysr_cfg) -> PySRRegressor:
    kwargs = pysr_cfg.as_kwargs()
    # Ensure each retrain has its own run directory to avoid artifact collisions
    run_id = f"merge_cluster"
    kwargs["run_id"] = run_id
    model = PySRRegressor(**kwargs)
    model.fit(X, y, variable_names=feature_names)
    return model

# -------------------------
# Merge decision for two equations
# -------------------------
def _can_merge(
    eq1: str,
    eq2: str,
    df: pd.DataFrame,
    feature_names: List[str],
    cache: ExprCache,
    embedder,
    threshold_strict: float = 0.90,
    threshold_loose: float = 0.80,
    mse_threshold: float = 1e-2,
    cluster1_rows: Optional[pd.DataFrame] = None,
    cluster2_rows: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> Tuple[bool, str, float]:
    n1, vars1, ops1, e1 = cache.get(eq1)
    n2, vars2, ops2, e2 = cache.get(eq2)

    # same variable set requirement
    if vars1 != vars2:
        return False, "Variable sets differ", 0.0

    sim = 0.0
    if embedder is not None:
        emb1 = embedder.encode(n1, convert_to_tensor=True)
        emb2 = embedder.encode(n2, convert_to_tensor=True)
        sim = float(util.cos_sim(emb1, emb2).item())
        if verbose:
            print(f"[merge-check] cosine={sim:.3f}")

    # strict pass on text similarity
    if sim >= threshold_strict:
        return True, "High cosine similarity", sim

    # loose pass → numeric cross-check required
    if sim >= threshold_loose:
        if cluster1_rows is None or cluster2_rows is None:
            return False, "Need numeric rows for loose check", sim

        symbs = sp.symbols(feature_names)
        f1 = lambdify(symbs, e1, modules="numpy")
        f2 = lambdify(symbs, e2, modules="numpy")

        X1 = cluster1_rows[feature_names].to_numpy()
        X2 = cluster2_rows[feature_names].to_numpy()
        y1 = cluster1_rows["y"].to_numpy()
        y2 = cluster2_rows["y"].to_numpy()

        y1_hat1 = np.array([f1(*row) for row in X1])
        y1_hat2 = np.array([f2(*row) for row in X1])
        y2_hat1 = np.array([f1(*row) for row in X2])
        y2_hat2 = np.array([f2(*row) for row in X2])

        m11 = mean_squared_error(y1, y1_hat1)
        m12 = mean_squared_error(y1, y1_hat2)
        m22 = mean_squared_error(y2, y2_hat2)
        m21 = mean_squared_error(y2, y2_hat1)

        if verbose:
            print(f"[merge-check] MSEs: e1@c1={m11:.4g}  e2@c1={m12:.4g}  e2@c2={m22:.4g}  e1@c2={m21:.4g}")

        avg1 = 0.5 * (m11 + m21)
        avg2 = 0.5 * (m22 + m12)

        if avg1 < mse_threshold and avg1 < avg2:
            return True, "Expr1 generalizes via MSE", sim
        if avg2 < mse_threshold and avg2 < avg1:
            return True, "Expr2 generalizes via MSE", sim

        return False, "High MSEs for both", sim

    return False, "Low cosine similarity", sim

# -------------------------
# Main API
# -------------------------
def run_posthoc_merging(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    bundle_dir: str,
    cfg: Config,
    *,
    threshold_strict: float = 0.90,
    threshold_loose: float = 0.80,
    mse_threshold: float = 1e-2,
    max_iterations: int = 10,
    embedder_name: str = "all-MiniLM-L6-v2",
    rounding_digits: int = 1,
    save_merged: bool = True,
    merged_bundle_name: str = "merged_Pc-SR_model",
    verbose: bool = False,
) -> Dict[str, Any]:

    tree, leaf_manifest, feat_from_bundle, cfg_dict, mani_index = _load_stage1_bundle(bundle_dir)
    if feat_from_bundle is not None and list(feat_from_bundle) != list(feature_names):
        warnings.warn("feature_names differ from bundle's feature list. Make sure order is consistent.")

    # 1) Build initial cluster ids = leaf ids (reindexed to 0..K-1 for convenience)
    leaf_ids, leaf_to_rows = _initial_leaf_clusters(tree, X)
    leaf_ids_compact = np.array([_compact_id_map(leaf_ids.tolist())[lid] for lid in leaf_ids], dtype=int)

    # 2) Seed cluster equations from Stage-1 leaves
    # Map: compact cluster id -> stage1 eq string
    by_leafid_to_eq: Dict[int, str] = {}
    # manifest entries have original leaf_id from sklearn
    # convert to compact ids:
    comp_map = _compact_id_map(list(leaf_to_rows.keys()))
    for entry in leaf_manifest:
        lid_orig = int(entry["leaf_id"])
        if lid_orig in comp_map:
            by_leafid_to_eq[comp_map[lid_orig]] = entry["equation"]

    # 3) Prepare df with y and features for convenience
    df = pd.DataFrame(X, columns=feature_names)
    df["y"] = y
    df["ClusterID"] = leaf_ids_compact.copy()
    df["InitialClusterID"] = df["ClusterID"].copy()

    # 4) Train symbolic models for each current cluster (to refresh and standardize)
    cluster_models: Dict[int, PySRRegressor] = {}
    cluster_equations: Dict[int, str] = {}

    for cid in sorted(df["ClusterID"].unique()):
        rows = df[df["ClusterID"] == cid]
        Xc = rows[feature_names].to_numpy()
        yc = rows["y"].to_numpy()
        model = _train_symbolic_model(Xc, yc, feature_names, cfg.pysr)
        cluster_models[cid] = model
        cluster_equations[cid] = str(model.get_best()["equation"])

    # 5) Merge loop
    cache = ExprCache(feature_names, rounding_digits=rounding_digits)
    embedder = _get_embedder(embedder_name)
    history: List[Dict[str, Any]] = []

    for it in range(1, max_iterations + 1):
        changed = False
        merged = set()
        cluster_ids = sorted(cluster_models.keys())

        if verbose:
            print(f"\n[merge] Iteration {it}: candidate pairs = {len(cluster_ids)} clusters")

        for i, ci in enumerate(cluster_ids):
            for cj in cluster_ids[i+1:]:
                if ci in merged or cj in merged:
                    continue

                expr_ci = cluster_equations[ci]
                expr_cj = cluster_equations[cj]
                rows_c1 = df[df["ClusterID"] == ci]
                rows_c2 = df[df["ClusterID"] == cj]

                can, reason, sim = _can_merge(
                    expr_ci, expr_cj, df, feature_names, cache, embedder,
                    threshold_strict=threshold_strict, threshold_loose=threshold_loose,
                    mse_threshold=mse_threshold,
                    cluster1_rows=rows_c1, cluster2_rows=rows_c2,
                    verbose=verbose,
                )

                if can:
                    # Merge cj into ci
                    df.loc[df["ClusterID"] == cj, "ClusterID"] = ci
                    merged.add(cj)
                    changed = True
                    if verbose:
                        print(f"[merge] ✅ Merged {cj} → {ci} | reason: {reason} | cos={sim:.3f}")

        # Retrain after any merges
        cluster_models.clear()
        cluster_equations.clear()
        for cid in sorted(df["ClusterID"].unique()):
            rows = df[df["ClusterID"] == cid]
            Xc = rows[feature_names].to_numpy()
            yc = rows["y"].to_numpy()
            model = _train_symbolic_model(Xc, yc, feature_names, cfg.pysr)
            cluster_models[cid] = model
            cluster_equations[cid] = str(model.get_best()["equation"])

        # Track global MSE using stitched predictions
        y_pred = np.empty_like(y, dtype=float)
        for cid, model in cluster_models.items():
            idx = (df["ClusterID"].values == cid)
            Xc = df.loc[idx, feature_names].to_numpy()
            y_pred[idx] = model.predict(Xc)
        mse_global = float(mean_squared_error(y, y_pred))
        sizes = df["ClusterID"].value_counts().sort_index().to_dict()
        history.append({"iteration": it, "cluster_sizes": sizes, "mse_global": mse_global,
                        "equations": cluster_equations.copy()})

        if verbose:
            print(f"[merge] Iter {it} → clusters={len(sizes)}, MSE={mse_global:.6f}")

        if not changed:
            if verbose:
                print("[merge] Converged: no further merges.")
            break

    # Build initial→final mapping
    mapping = {}
    for init_id in df["InitialClusterID"].unique():
        final_id = int(df.loc[df["InitialClusterID"] == init_id, "ClusterID"].mode().iloc[0])
        mapping[int(init_id)] = final_id

    out: Dict[str, Any] = {
        "tree": tree,
        "cluster_id": df["ClusterID"].to_numpy(),
        "cluster_models": cluster_models,
        "cluster_equations": cluster_equations,
        "history": history,
        "mapping_init_to_final": mapping,
    }

    # Optional: save merged manifest (equations + mapping)
    if save_merged:
        merged_dir = os.path.join(bundle_dir, "posthoc_merged")
        os.makedirs(merged_dir, exist_ok=True)
        manifest = {
            "feature_names": list(feature_names),
            "notes": "Post-hoc merged clusters with retrained PySR models.",
            "mapping_init_to_final": mapping,
            "clusters": [
                {
                    "final_cluster_id": int(cid),
                    "n": int((df['ClusterID']==cid).sum()),
                    "equation": str(eq),
                }
                for cid, eq in sorted(cluster_equations.items())
            ],
        }
        out_path = os.path.join(merged_dir, f"{merged_bundle_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        out["merged_manifest_path"] = out_path

    out["final_n_clusters"] = len(set(df["ClusterID"].tolist()))
    out["final_mse"] = history[-1]["mse_global"] if history else float("nan")

    return out
