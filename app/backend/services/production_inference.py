"""Production inference for drop-in mllabiome-ii deployment packages.

Put exported model zip packages in ``app/backend/production_models/`` and restart
the backend. The service loads generic MPMA-E package schemas and legacy object
artifacts through local compatibility aliases.

Runtime wheels for optional manuscript estimators (xgboost, lightgbm, catboost, flaml)
are declared in the backend requirements. Those libraries are intentionally not
mocked: serialized estimator objects need the real package at unpickle/predict
time.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

import joblib

# Import/register compatibility modules before any joblib.load call. Some IBS
# artifacts reference top-level ibs_mpmae_runtime; older artifacts reference
# top-level ibs_mpmae_model_defs. In the packaged app the implementation lives
# under app.backend, so both historical top-level names are explicitly mapped
# to the packaged runtime module before unpickling.
try:  # normal installed/editable package mode
    from app.backend import ibs_mpmae_runtime as ibs_runtime
except ImportError:  # direct execution from app/backend during local debugging
    import ibs_mpmae_runtime as ibs_runtime  # type: ignore


def _register_unpickle_compat_modules() -> None:
    """Register legacy/private module aliases needed by exported model pickles.

    The manuscript/export environment can pickle deployment helpers under
    historical top-level module names. Some scikit-learn estimators can also
    persist private compiled extension classes as a short top-level module name
    such as ``_loss``. Those names are not part of this app, but they can be
    mapped to the installed sklearn implementation before ``joblib.load``.
    """
    sys.modules["ibs_mpmae_runtime"] = ibs_runtime
    sys.modules["ibs_mpmae_model_defs"] = ibs_runtime

    # scikit-learn private Cython loss module. Artifacts containing
    # SGDClassifier / LogisticRegression internals may reference this as a
    # top-level module on some sklearn/Python builds. Do not fail unpickling
    # just because the private module was recorded under the short name.
    try:
        from sklearn._loss import _loss as sklearn_loss  # type: ignore
    except Exception:
        try:
            import sklearn._loss._loss as sklearn_loss  # type: ignore
        except Exception:
            sklearn_loss = None  # type: ignore[assignment]
    if sklearn_loss is not None:
        sys.modules.setdefault("_loss", sklearn_loss)

    # Optional estimator libraries used by manuscript sweeps. If installed,
    # import them before joblib.load so their sklearn wrapper modules are
    # registered in sys.modules. If absent, joblib.load will still raise a
    # clear ModuleNotFoundError naming the missing dependency.
    for _optional_module in ("xgboost", "lightgbm", "catboost", "flaml"):
        if importlib.util.find_spec(_optional_module) is not None:
            try:
                __import__(_optional_module)
            except Exception:
                pass


_register_unpickle_compat_modules()
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from scipy.stats import rankdata


def _relative(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.clip(X, 0.0, None)
    s = X.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return (X / s).astype(np.float32)


def _class_labels_from_manifest(manifest: dict[str, Any]) -> tuple[str, str]:
    positive = str(
        manifest.get("positive_label") or manifest.get("target") or "Positive"
    )
    negative = str(manifest.get("negative_label") or "Control")
    return negative, positive


def _clean_feature_list(features: Optional[list[str] | str]) -> list[str]:
    if features is None:
        return []
    if isinstance(features, str):
        raw = features.replace("\r", "\n").replace(",", "\n").split("\n")
    else:
        raw = features
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


_EXPLAINABILITY_DEPENDENCY_MODULES: dict[str, str] = {"shap": "shap", "lime": "lime"}

_FAST_EXPLAIN_MAX_FEATURES = int(os.getenv("MLLABIOME_EXPLAIN_MAX_FEATURES", "12"))
_FAST_EXPLAIN_CANDIDATE_POOL = int(os.getenv("MLLABIOME_EXPLAIN_CANDIDATE_POOL", "32"))
_FAST_EXPLAIN_SCREEN_MAX_FEATURES = int(
    os.getenv("MLLABIOME_EXPLAIN_SCREEN_MAX_FEATURES", "2048")
)
_FAST_EXPLAIN_MAX_BACKGROUND = int(os.getenv("MLLABIOME_EXPLAIN_MAX_BACKGROUND", "12"))
_FAST_EXPLAIN_SHAP_SAMPLES = int(os.getenv("MLLABIOME_EXPLAIN_SHAP_SAMPLES", "128"))
_FAST_EXPLAIN_LIME_SAMPLES = int(os.getenv("MLLABIOME_EXPLAIN_LIME_SAMPLES", "192"))


def _bounded_explain_feature_count(num_features: int) -> int:
    return max(
        1,
        min(
            int(num_features or _FAST_EXPLAIN_MAX_FEATURES), _FAST_EXPLAIN_MAX_FEATURES
        ),
    )


def _bounded_candidate_count(num_features: int) -> int:
    """Return the bounded SHAP/LIME candidate count."""
    n = _bounded_explain_feature_count(num_features)
    return max(
        n,
        min(
            max(_FAST_EXPLAIN_CANDIDATE_POOL, n),
            max(n, _FAST_EXPLAIN_SCREEN_MAX_FEATURES),
        ),
    )


def _bounded_screen_candidate_count(num_features: int) -> int:
    """Maximum number of raw taxa considered in the cheap pre-screen.

    This is deliberately much larger than the number displayed. The app first
    screens the uploaded/sample feature space, then runs SHAP/LIME on the top
    screened candidates, and only then displays the strongest method-specific
    local attributions.
    """
    n = _bounded_explain_feature_count(num_features)
    return max(_bounded_candidate_count(n), int(_FAST_EXPLAIN_SCREEN_MAX_FEATURES))


def _bounded_shap_samples(num_samples: int, n_features: int) -> int:
    requested = int(num_samples or _FAST_EXPLAIN_SHAP_SAMPLES)
    floor = min(_FAST_EXPLAIN_SHAP_SAMPLES, max(32, 2 * int(n_features) + 8))
    return max(floor, min(requested, _FAST_EXPLAIN_SHAP_SAMPLES))


def _bounded_lime_samples(num_samples: int, n_features: int) -> int:
    requested = int(num_samples or _FAST_EXPLAIN_LIME_SAMPLES)
    floor = min(_FAST_EXPLAIN_LIME_SAMPLES, max(64, 4 * int(n_features) + 16))
    return max(floor, min(requested, _FAST_EXPLAIN_LIME_SAMPLES))


def _dependency_status() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label, module in _EXPLAINABILITY_DEPENDENCY_MODULES.items():
        spec = importlib.util.find_spec(module)
        out[label] = {"available": spec is not None, "module": module}
    return out


def _normalise_support(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not features:
        return features
    max_abs = (
        max(
            float(f.get("abs_importance", abs(float(f.get("importance", 0.0)))))
            for f in features
        )
        or 0.0
    )
    if max_abs <= 0:
        max_abs = 1.0
    for rank, feat in enumerate(features, start=1):
        importance = float(feat.get("importance", 0.0))
        feat.setdefault("abs_importance", abs(importance))
        feat["support"] = float(abs(importance) / max_abs)
        feat["direction"] = "toward" if importance >= 0 else "against"
        feat["method"] = "local_baseline_ablation"
        feat["rank"] = rank
    return features


def _normalise_explainer_features(
    features: list[dict[str, Any]], method: str
) -> list[dict[str, Any]]:
    """Sort and scale local SHAP/LIME feature contributions for the UI."""
    cleaned: list[dict[str, Any]] = []
    for feat in features:
        try:
            value = float(feat.get("importance", 0.0))
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        out = dict(feat)
        out["importance"] = value
        out["abs_importance"] = abs(value)
        out["method"] = method
        cleaned.append(out)
    cleaned.sort(key=lambda item: float(item.get("abs_importance", 0.0)), reverse=True)
    max_abs = (
        max((float(f.get("abs_importance", 0.0)) for f in cleaned), default=1.0) or 1.0
    )
    for rank, feat in enumerate(cleaned, start=1):
        feat["support"] = float(float(feat.get("abs_importance", 0.0)) / max_abs)
        feat["direction"] = (
            "positive" if float(feat.get("importance", 0.0)) >= 0 else "negative"
        )
        feat["rank"] = rank
    return cleaned


def _local_background_matrix(
    observed: np.ndarray,
    instance: np.ndarray,
    max_rows: int = _FAST_EXPLAIN_MAX_BACKGROUND,
) -> np.ndarray:
    """Build a small numeric background for local model-agnostic explainers."""
    inst = np.asarray(instance, dtype=np.float64).reshape(1, -1)
    obs = np.asarray(observed, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.size == 0 or obs.shape[1] != inst.shape[1]:
        obs = inst.copy()
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    obs = np.clip(obs, 0.0, None)

    rows: list[np.ndarray] = []
    if obs.shape[0] > 1:
        if obs.shape[0] > max_rows:
            idx = np.linspace(0, obs.shape[0] - 1, max_rows).round().astype(int)
            rows.append(obs[idx])
        else:
            rows.append(obs)
        rows.append(np.nanmedian(obs, axis=0, keepdims=True))
    else:
        multipliers = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64).reshape(
            -1, 1
        )
        rows.append(inst * multipliers)
    rows.append(np.zeros_like(inst))
    rows.append(inst)
    bg = np.vstack(rows).astype(np.float64)

    # Remove exact duplicate rows while preserving order.
    unique_rows = []
    seen: set[tuple[float, ...]] = set()
    for row in bg:
        key = tuple(np.round(row, 12).tolist())
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    bg = np.vstack(unique_rows) if unique_rows else inst.copy()

    # LIME estimates feature scales from the background.  Add a tiny deterministic
    # spread for constant columns to avoid divide-by-zero handling in older lime.
    if bg.shape[0] < 2:
        bg = np.vstack([bg, bg.copy()])
    for j in range(bg.shape[1]):
        if float(np.nanmax(bg[:, j]) - np.nanmin(bg[:, j])) <= 1e-12:
            eps = max(abs(float(inst[0, j])) * 1e-6, 1e-9)
            bg[-1, j] = max(0.0, bg[-1, j] + eps)
    return bg.astype(np.float64)


def _coerce_shap_vector(raw_values: Any, n_features: int) -> np.ndarray:
    vals = raw_values
    if isinstance(vals, (list, tuple)):
        vals = vals[-1]
    arr = np.asarray(vals, dtype=float)
    if arr.ndim == 3:
        # Common shapes: (n_outputs, n_samples, n_features) or
        # (n_samples, n_features, n_outputs). Use the positive-probability output.
        if arr.shape[-1] == 2 and arr.shape[1] == n_features:
            arr = arr[0, :, -1]
        elif arr.shape[0] == 2 and arr.shape[-1] == n_features:
            arr = arr[-1, 0, :]
        else:
            arr = arr.reshape(-1, n_features)[0]
    elif arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr.reshape(-1)[:n_features]
    elif arr.ndim == 0:
        arr = np.zeros(n_features, dtype=float)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size < n_features:
        arr = np.pad(arr, (0, n_features - arr.size), constant_values=0.0)
    return arr[:n_features]


def _kernel_shap_features(
    predict_positive,
    background: np.ndarray,
    instance: np.ndarray,
    feature_names: list[str],
    num_features: int,
    num_samples: int,
) -> list[dict[str, Any]]:
    try:
        import shap  # type: ignore
    except Exception as exc:
        raise ImportError(
            "SHAP is not installed. Install backend requirements with: python -m pip install -r app/backend/requirements.txt"
        ) from exc

    bg = np.asarray(background, dtype=float)
    x = np.asarray(instance, dtype=float).reshape(1, -1)
    explainer = shap.KernelExplainer(
        lambda z: np.asarray(predict_positive(z), dtype=float), bg, link="identity"
    )
    nsamples = _bounded_shap_samples(num_samples, len(feature_names))
    l1_reg = f"num_features({min(int(num_features), len(feature_names))})"
    try:
        raw_values = explainer.shap_values(
            x, nsamples=nsamples, l1_reg=l1_reg, silent=True
        )
    except TypeError:
        try:
            raw_values = explainer.shap_values(x, nsamples=nsamples, l1_reg=l1_reg)
        except TypeError:
            raw_values = explainer.shap_values(x, nsamples=nsamples)
    values = _coerce_shap_vector(raw_values, len(feature_names))
    baseline = (
        np.nanmedian(bg, axis=0)
        if bg.size
        else np.zeros(len(feature_names), dtype=float)
    )
    rows = [
        {
            "feature": str(name),
            "importance": float(val),
            "std": 0.0,
            "n_models": 1,
            "value": float(x[0, idx]),
            "baseline": float(baseline[idx]) if idx < len(baseline) else 0.0,
        }
        for idx, (name, val) in enumerate(zip(feature_names, values))
    ]
    return _normalise_explainer_features(rows, "kernel_shap")[:num_features]


def _lime_tabular_features(
    predict_proba,
    background: np.ndarray,
    instance: np.ndarray,
    feature_names: list[str],
    num_features: int,
    num_samples: int,
) -> list[dict[str, Any]]:
    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    except Exception as exc:
        raise ImportError(
            "LIME is not installed. Install backend requirements with: python -m pip install -r app/backend/requirements.txt"
        ) from exc

    bg = np.asarray(background, dtype=float)
    x = np.asarray(instance, dtype=float).reshape(-1)
    explainer = LimeTabularExplainer(
        training_data=bg,
        feature_names=feature_names,
        class_names=["negative", "positive"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
        verbose=False,
    )
    exp = explainer.explain_instance(
        x,
        predict_proba,
        labels=(1,),
        num_features=min(int(num_features), len(feature_names)),
        num_samples=_bounded_lime_samples(num_samples, len(feature_names)),
    )
    coeffs = np.zeros(len(feature_names), dtype=float)
    for idx, value in exp.as_map().get(1, []):
        if 0 <= int(idx) < len(coeffs):
            coeffs[int(idx)] = float(value)
    baseline = (
        np.nanmedian(bg, axis=0)
        if bg.size
        else np.zeros(len(feature_names), dtype=float)
    )
    rows = [
        {
            "feature": str(name),
            "importance": float(val),
            "std": 0.0,
            "n_models": 1,
            "value": float(x[idx]),
            "baseline": float(baseline[idx]) if idx < len(baseline) else 0.0,
        }
        for idx, (name, val) in enumerate(zip(feature_names, coeffs))
    ]
    return _normalise_explainer_features(rows, "lime")[:num_features]


def _screen_local_explainer_candidates(
    predict_positive,
    observed: np.ndarray,
    instance: np.ndarray,
    feature_names: list[str],
    base_probability: float,
    pool_size: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Fast local pre-screen over candidate taxa before SHAP/LIME.

    KernelSHAP/LIME cannot be run directly over thousands of taxa in an
    interactive request.  This screen performs a single batched one-feature
    replacement for every candidate taxon, ranks by absolute local probability
    change, and keeps a modest candidate pool. SHAP/LIME then compute their own
    local rankings within that screened pool.
    """
    if not feature_names:
        return feature_names, observed, instance
    n = len(feature_names)
    keep_n = max(1, min(int(pool_size), n))
    if n <= keep_n:
        return feature_names, observed, instance

    obs = np.asarray(observed, dtype=float)
    inst = np.asarray(instance, dtype=float).reshape(-1)
    if obs.ndim != 2 or obs.shape[1] != n or inst.size != n:
        return (
            feature_names[:keep_n],
            np.asarray(observed)[:, :keep_n],
            np.asarray(instance)[:keep_n],
        )

    baseline = np.nanmedian(obs, axis=0) if obs.size else np.zeros(n, dtype=float)
    baseline = np.nan_to_num(baseline, nan=0.0, posinf=0.0, neginf=0.0)
    Z = np.repeat(inst.reshape(1, -1), n, axis=0)
    for j in range(n):
        Z[j, j] = baseline[j]
    try:
        alt = np.asarray(predict_positive(Z), dtype=float).reshape(-1)
        delta = np.abs(float(base_probability) - alt)
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        delta = np.zeros(n, dtype=float)
    abundance = np.nan_to_num(np.abs(inst), nan=0.0, posinf=0.0, neginf=0.0)
    order = sorted(
        range(n), key=lambda j: (float(delta[j]), float(abundance[j])), reverse=True
    )[:keep_n]

    # Keep the screened taxa ordered by local screening strength. The SHAP/LIME
    # results returned to the UI are still independently sorted by each method.
    names = [feature_names[j] for j in order]
    return names, obs[:, order].astype(float), inst[order].astype(float)


def _compute_local_shap_lime(
    predict_positive,
    predict_proba,
    observed: np.ndarray,
    instance: np.ndarray,
    feature_names: list[str],
    num_features: int,
    num_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not feature_names:
        return [], []
    num_features = _bounded_explain_feature_count(num_features)
    # Do not truncate the candidate matrix to the display count here.  SHAP and
    # LIME should rank within the screened local candidate pool and then return
    # the top displayed features.
    background = _local_background_matrix(observed, instance)
    shap_features = _kernel_shap_features(
        predict_positive=predict_positive,
        background=background,
        instance=instance,
        feature_names=feature_names,
        num_features=num_features,
        num_samples=num_samples,
    )
    lime_features = _lime_tabular_features(
        predict_proba=predict_proba,
        background=background,
        instance=instance,
        feature_names=feature_names,
        num_features=num_features,
        num_samples=num_samples,
    )
    return shap_features, lime_features


def _normalise_interactions(
    edges: list[dict[str, Any]], max_edges: int = 12
) -> list[dict[str, Any]]:
    """Prepare local pairwise perturbation edges for the frontend network."""
    cleaned: list[dict[str, Any]] = []
    for edge in edges:
        try:
            strength = float(edge.get("strength", 0.0))
        except Exception:
            continue
        if not np.isfinite(strength) or abs(strength) <= 0:
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target or source == target:
            continue
        cleaned.append(
            {
                "source": source,
                "target": target,
                "strength": strength,
                "abs_strength": abs(strength),
                "direction": "synergy" if strength >= 0 else "redundancy",
                "method": "local_pairwise_ablation",
            }
        )
    cleaned.sort(key=lambda x: x["abs_strength"], reverse=True)
    top = cleaned[:max_edges]
    max_abs = max((float(e["abs_strength"]) for e in top), default=1.0) or 1.0
    for e in top:
        e["support"] = float(e["abs_strength"] / max_abs)
    return top


def _pairwise_feature_names(
    features: list[dict[str, Any]], max_nodes: int = 7
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for feat in features:
        name = str(feat.get("feature", "")).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
        if len(names) >= max_nodes:
            break
    return names


def _effect_grid(values: np.ndarray, num_points: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.asarray([], dtype=np.float32)
    vmax = float(np.nanmax(vals))
    if vmax <= 0:
        return np.asarray([], dtype=np.float32)
    positive = vals[vals > 0]
    upper = float(np.nanpercentile(positive if positive.size else vals, 95))
    upper = max(upper, vmax if upper <= 0 else upper)
    if upper <= 0:
        return np.asarray([], dtype=np.float32)
    n = max(3, min(int(num_points), 100))
    return np.linspace(0.0, upper, n, dtype=np.float32)


def _mr(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64).copy()
    out[out <= 0] = eps
    return out


def _ilr_basis(D: int) -> np.ndarray:
    B = np.zeros((max(D - 1, 0), D), dtype=np.float64)
    for k in range(1, D):
        scale = math.sqrt(k / (k + 1.0))
        B[k - 1, :k] = scale / k
        B[k - 1, k] = -scale
    return B


def _apply_transform_state(state: dict[str, Any], X_raw: np.ndarray) -> np.ndarray:
    name = str(state["name"])
    X = np.asarray(X_raw, dtype=np.float32)
    if name == "none":
        return _relative(X)
    if name == "binary":
        return (X > 0).astype(np.float32)
    if name == "sqrt":
        return np.sqrt(np.clip(_relative(X), 0, None)).astype(np.float32)
    if name == "hellinger":
        return np.sqrt(_relative(X)).astype(np.float32)
    if name == "arcsin_sqrt":
        return np.arcsin(np.sqrt(np.clip(_relative(X), 0, 1))).astype(np.float32)
    if name == "log":
        return np.log1p(np.clip(_relative(X), 0, None)).astype(np.float32)
    if name == "log2":
        return np.log2(np.clip(_relative(X), 0, None) + 1).astype(np.float32)
    if name == "log10":
        return np.log10(np.clip(_relative(X), 0, None) + 1).astype(np.float32)
    if name == "log_tss_floor":
        return np.log(np.maximum(_relative(X), 1e-10)).astype(np.float32)
    if name == "zi_log":
        out = np.zeros_like(X, dtype=np.float64)
        mask = X > 0
        out[mask] = np.log(np.clip(X[mask], 1e-300, None))
        return out.astype(np.float32)
    if name == "symlog":
        return (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
    if name == "zscore":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd < 1e-10] = 1.0
        return ((X - mu) / sd).astype(np.float32)
    if name == "log_std":
        lx = np.log1p(np.clip(_relative(X), 0, None))
        mu = lx.mean(axis=1, keepdims=True)
        sd = lx.std(axis=1, keepdims=True)
        sd[sd < 1e-10] = 1.0
        return ((lx - mu) / sd).astype(np.float32)
    if name == "log_unit":
        lx = np.log1p(np.clip(_relative(X), 0, None))
        n = np.sqrt((lx**2).sum(axis=1, keepdims=True))
        n[n < 1e-10] = 1.0
        return (lx / n).astype(np.float32)
    if name == "rank_frac":
        return (np.apply_along_axis(rankdata, 1, X) / (X.shape[1] + 1.0)).astype(
            np.float32
        )
    if name == "rank_std":
        r = np.apply_along_axis(rankdata, 1, X).astype(np.float32)
        mu = r.mean(axis=1, keepdims=True)
        sd = r.std(axis=1, keepdims=True)
        sd[sd < 1e-10] = 1.0
        return ((r - mu) / sd).astype(np.float32)
    if name == "rank_unit":
        r = np.apply_along_axis(rankdata, 1, X).astype(np.float32)
        n = np.sqrt((r**2).sum(axis=1, keepdims=True))
        n[n < 1e-10] = 1.0
        return (r / n).astype(np.float32)
    if name in {"clr", "scikit-bio_clr"}:
        lx = np.log(_mr(_relative(X)))
        return (lx - lx.mean(axis=1, keepdims=True)).astype(np.float32)
    if name in {"alr", "scikit-bio_alr"}:
        Z = _mr(_relative(X))
        return np.log(Z[:, :-1] / Z[:, -1:]).astype(np.float32)
    if name in {"ilr", "scikit-bio_ilr"}:
        Z = _mr(_relative(X))
        D = Z.shape[1]
        if D < 2:
            return Z.astype(np.float32)
        return (np.log(Z) @ _ilr_basis(D).T).astype(np.float32)
    if name == "clr_std":
        lx = np.log(_mr(_relative(X)))
        base = (lx - lx.mean(axis=1, keepdims=True)).astype(np.float32)
        return ((base - state["mu"]) / state["sd"]).astype(np.float32)
    if name == "ilr_std":
        Z = _mr(_relative(X))
        D = Z.shape[1]
        base = (
            Z.astype(np.float32)
            if D < 2
            else (np.log(Z) @ _ilr_basis(D).T).astype(np.float32)
        )
        return ((base - state["mu"]) / state["sd"]).astype(np.float32)
    if name == "rank_col":
        out = np.zeros_like(X, dtype=np.float32)
        n = max(float(state["n_train"]), 1.0)
        for j, col in enumerate(state["sorted_cols"]):
            out[:, j] = np.searchsorted(col, X[:, j], side="right") / n
        return out
    if name in {"prev_weighted", "prev_weigthed"}:
        weighted = X * state["prev"]
        s = weighted.sum(axis=1, keepdims=True)
        s[s <= 0] = 1.0
        return (weighted / s).astype(np.float32)
    if name in {"robust", "power", "quantile"}:
        return state["estimator"].transform(X).astype(np.float32)
    if name == "pairwise_logratio":
        i_idx = np.asarray(state["i_idx"], dtype=int)
        j_idx = np.asarray(state["j_idx"], dtype=int)
        if len(i_idx) == 0:
            return X.astype(np.float32)
        Z = _mr(X)
        return np.log(Z[:, i_idx] / Z[:, j_idx]).astype(np.float32)
    raise ValueError(f"Unsupported transform in deployment artifact: {name!r}")


def _renormalize(p: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if p.shape[1] != n_classes:
        q = np.zeros((p.shape[0], n_classes), dtype=float)
        q[:, : min(n_classes, p.shape[1])] = p[:, : min(n_classes, p.shape[1])]
        p = q
    p = np.nan_to_num(
        p, nan=1.0 / n_classes, posinf=1.0 / n_classes, neginf=1.0 / n_classes
    )
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    empty = s.squeeze() <= 1e-12
    s = np.where(s > 1e-12, s, 1.0)
    p = p / s
    if np.any(empty):
        p[empty, :] = 1.0 / n_classes
    return p.astype(np.float32)


def _coerce_X_for_estimator(estimator: Any, X: np.ndarray, feature_names: list[str]):
    names = getattr(estimator, "feature_names_in_", None)
    if names is not None:
        names = [str(x) for x in list(names)]
        if len(names) == X.shape[1]:
            return pd.DataFrame(X, columns=names)
    return pd.DataFrame(X, columns=feature_names)


def _predict_proba(
    estimator: Any, X: np.ndarray, classes: np.ndarray, feature_names: list[str]
) -> np.ndarray:
    X_in = _coerce_X_for_estimator(estimator, X, feature_names)
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(X_in), dtype=float)
        if raw.ndim == 1:
            raw = np.column_stack([1.0 - raw, raw])
        learned = np.asarray(getattr(estimator, "classes_", classes), dtype=int)
        out = np.zeros((X.shape[0], len(classes)), dtype=float)
        for j, c in enumerate(learned):
            loc = np.where(classes == c)[0]
            if len(loc) and j < raw.shape[1]:
                out[:, int(loc[0])] = raw[:, j]
        return _renormalize(out, len(classes))
    if hasattr(estimator, "decision_function"):
        score = np.asarray(estimator.decision_function(X_in), dtype=float)
        if score.ndim == 1:
            p1 = expit(score)
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)
        return softmax(score, axis=1).astype(np.float32)
    pred = np.asarray(estimator.predict(X_in), dtype=int)
    out = np.zeros((X.shape[0], len(classes)), dtype=float)
    for i, p in enumerate(pred):
        loc = np.where(classes == p)[0]
        if len(loc):
            out[i, int(loc[0])] = 1.0
    return _renormalize(out, len(classes))


def _aggregate(stack: np.ndarray, weights: np.ndarray, method: str) -> np.ndarray:
    stack = np.asarray(stack, dtype=float)
    weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0)
    method = str(method)
    if method == "median_proba":
        proba = np.median(stack, axis=0)
    elif method == "weighted_mean_proba":
        w = np.clip(weights - np.nanmin(weights), 0, None) + 1e-8
        w = w / w.sum()
        proba = np.tensordot(w, stack, axes=(0, 0))
    elif method == "rank_mean":
        ranked = np.empty_like(stack)
        for m in range(stack.shape[0]):
            for c in range(stack.shape[2]):
                ranked[m, :, c] = rankdata(stack[m, :, c])
        proba = ranked.mean(axis=0)
    elif method == "majority_vote":
        preds = stack.argmax(axis=2)
        proba = np.zeros(stack.shape[1:], dtype=float)
        for i in range(stack.shape[1]):
            counts = np.bincount(preds[:, i], minlength=stack.shape[2]).astype(float)
            proba[i, :] = counts / max(1.0, counts.sum())
    elif method == "max_proba":
        proba = stack.max(axis=0)
    elif method == "min_proba":
        proba = stack.min(axis=0)
    else:
        proba = stack.mean(axis=0)
    return _renormalize(proba, stack.shape[2])


def _aggregate_package(stack: np.ndarray, package: dict[str, Any]) -> np.ndarray:
    method = str(package.get("aggregation_strategy", "mean_proba"))
    if method.startswith("superlearner__"):
        meta_model = package.get("meta_model")
        if meta_model is None:
            raise ValueError(f"{method} requires meta_model in the deployment package")
        X_meta = np.asarray(stack, dtype=float)[:, :, -1].T
        if hasattr(meta_model, "predict_proba"):
            raw = np.asarray(meta_model.predict_proba(X_meta), dtype=float)
            if raw.ndim == 1:
                p1 = raw
            else:
                classes = list(getattr(meta_model, "classes_", [0, 1]))
                col = classes.index(1) if 1 in classes else min(1, raw.shape[1] - 1)
                p1 = raw[:, col]
        else:
            p1 = expit(np.asarray(meta_model.predict(X_meta), dtype=float))
        return _renormalize(np.column_stack([1.0 - p1, p1]), stack.shape[2])
    weights = np.asarray(package.get("weights", np.ones(stack.shape[0])), dtype=float)
    return _aggregate(stack, weights, method)


SINGLE_LEVELS = [
    "domain",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "strain",
]
LEVEL_DEPTH = {name: i + 1 for i, name in enumerate(SINGLE_LEVELS)}
LEVEL_MARKERS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "domain": (("d__",), ("___p__",)),
    "phylum": (("___p__",), ("___c__",)),
    "class": (("___c__",), ("___o__",)),
    "order": (("___o__",), ("___f__",)),
    "family": (("___f__",), ("___g__",)),
    "genus": (("___g__",), ("___s__",)),
    "species": (("___s__",), ("___t__",)),
    "strain": (("___t__",), ()),
}


def _is_lineage_name(name: str) -> bool:
    text = str(name).strip()
    return bool(
        text.startswith(("d__", "k__", "p__", "c__", "o__", "f__", "g__", "s__", "t__"))
        or "|p__" in text
        or "___p__" in text
    )


def _truncate_lineage_for_level(name: str, level: str) -> str:
    raw = str(name).strip()
    depth = LEVEL_DEPTH[level]
    if "|" in raw:
        return (
            "|".join(raw.split("|")[:depth])
            .replace("k__", "d__", 1)
            .replace("|", "___")
        )
    return "___".join(raw.split("___")[:depth]).replace("k__", "d__", 1)


def _feature_name_ok_for_level(name: str, level: str) -> bool:
    must_have, must_not = LEVEL_MARKERS[level]
    text = str(name)
    if level == "domain":
        has = text.startswith("d__")
    else:
        has = any(marker in text for marker in must_have)
    return bool(has and not any(marker in text for marker in must_not))


def _infer_feature_level(name: str) -> Optional[str]:
    text = str(name)
    for level in reversed(SINGLE_LEVELS):
        if _feature_name_ok_for_level(text, level):
            return level
    return None


def _numeric_feature_columns(df: pd.DataFrame) -> list[Any]:
    cols: list[Any] = []
    for col in df.columns:
        if not _is_lineage_name(str(col)):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            cols.append(col)
    return cols


def _align_raw(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Align an uploaded sample-by-feature table to deployment features.

    Supports exact feature columns and LAMPP benchmark wide CSVs whose columns are
    strain-level MetaPhlAn lineages such as ``k__...|p__...|...|t__...``.  When a
    deployment feature is an aggregated rank (phylum/class/order/family/genus),
    uploaded strain columns are automatically truncated and summed to that rank.
    Missing deployment features are filled with zero.
    """
    raw = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    out = np.zeros((len(df), len(feature_names)), dtype=np.float32)
    missing: list[tuple[int, str]] = []

    for j, name in enumerate(feature_names):
        col = raw.get(name) or lower.get(name.lower())
        if col is not None:
            out[:, j] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        else:
            missing.append((j, str(name)))

    if not missing:
        return out

    needed_levels = sorted(
        {lv for _, name in missing if (lv := _infer_feature_level(name))},
        key=SINGLE_LEVELS.index,
    )
    if not needed_levels:
        return out

    numeric_cols = _numeric_feature_columns(df)
    if not numeric_cols:
        return out

    grouped: dict[str, dict[str, np.ndarray]] = {level: {} for level in needed_levels}
    for col in numeric_cols:
        values = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        for level in needed_levels:
            key = _truncate_lineage_for_level(str(col), level)
            if not _feature_name_ok_for_level(key, level):
                continue
            if key in grouped[level]:
                grouped[level][key] = grouped[level][key] + values
            else:
                grouped[level][key] = values.copy()

    for j, name in missing:
        level = _infer_feature_level(name)
        if level is None:
            continue
        values = grouped.get(level, {}).get(name)
        if values is not None:
            out[:, j] = values.astype(np.float32)
    return out


_BENCHMARK_FILENAME_ALIASES: dict[str, str] = {
    "crc": "crc.csv",
    "colorectal cancer": "crc.csv",
    "ghs": "ghs.csv",
    "general health status": "ghs.csv",
    "ibd": "ibd.csv",
    "inflammatory bowel disease": "ibd.csv",
    "scz": "scz.csv",
    "schizophrenia": "scz.csv",
    "dm": "dmw.csv",
    "delivery mode": "dmw.csv",
    "dmw": "dmw.csv",
    "western delivery mode": "dmw.csv",
    "birth delivery mode with western test set": "dmw.csv",
    "dmnw": "dmnw.csv",
    "non-western delivery mode": "dmnw.csv",
    "birth delivery mode with non-western test set": "dmnw.csv",
    "ibs": "ibs.csv",
}


def _safe_submission_filename(model_id: str, manifest: dict[str, Any]) -> str:
    fields = [
        str(model_id),
        str(manifest.get("target", "")),
        str(manifest.get("display_name", "")),
        str(manifest.get("positive_label", "")),
    ]
    haystack = " ".join(fields).lower().replace("_", "-")
    for key, filename in _BENCHMARK_FILENAME_ALIASES.items():
        if key in haystack:
            return filename
    cleaned = (
        "".join(ch if ch.isalnum() else "_" for ch in str(model_id).lower()).strip("_")
        or "predictions"
    )
    return f"{cleaned}.csv"


class ProductionModelService:
    def __init__(self, models_dir: str = "production_models"):
        configured = os.getenv("PRODUCTION_MODELS_DIR")
        self.models_dir = (
            Path(configured).expanduser().resolve()
            if configured
            else Path(__file__).parent.parent / models_dir
        )
        self.loaded_models: dict[str, Any] = {}
        self.available_models: dict[str, dict[str, Any]] = {}
        self._discover_models()

    def _discover_models(self) -> None:
        self.available_models = {}
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
            return
        for path in sorted(self.models_dir.glob("*.zip")):
            try:
                with zipfile.ZipFile(path) as zf:
                    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                if manifest.get("artifact_schema") not in {
                    "mllabiome.mpmae.v1",
                    "mllabiome.ibs_mpmae_object.v1",
                    "mllabiome.mpmae_object.v1",
                }:
                    continue
                model_id = str(manifest.get("model_id") or path.stem)
                self.available_models[model_id] = {
                    "kind": "mpmae_zip",
                    "path": path,
                    "model_id": model_id,
                    "display_name": manifest.get("display_name", model_id),
                    "description": manifest.get(
                        "description", "mllabiome-ii MPMA-E deployment model"
                    ),
                    "target": manifest.get("target", "label"),
                    "n_models": int(
                        manifest.get(
                            "n_members",
                            len(manifest.get("members", []))
                            or len(manifest.get("selected_members", [])),
                        )
                    ),
                    "calibrated": False,
                    "manifest": manifest,
                }
            except Exception:
                continue
        for path in sorted(self.models_dir.glob("*.joblib")):
            try:
                package = joblib.load(path)
                if package.get("artifact_schema") != "mllabiome.mpmae.v1":
                    continue
                model_id = str(package.get("model_id") or path.stem)
                self.available_models.setdefault(
                    model_id,
                    {
                        "kind": "mpmae_joblib",
                        "path": path,
                        "model_id": model_id,
                        "display_name": package.get("display_name", model_id),
                        "description": package.get(
                            "description", "mllabiome-ii MPMA-E deployment model"
                        ),
                        "target": package.get("target", "label"),
                        "n_models": len(package.get("members", [])),
                        "calibrated": False,
                        "manifest": {
                            k: v for k, v in package.items() if k not in {"members"}
                        },
                    },
                )
            except Exception:
                continue

    def explainability_status(self) -> dict[str, Any]:
        """Report availability of local SHAP and LIME explainers."""
        return {
            "method": "local_shap_lime",
            "method_label": "SHAP + LIME (fast local)",
            "basis": "",
            "dependencies": _dependency_status(),
            "strict_methods": ["kernel_shap", "lime_tabular"],
            "style_palette": {
                "ink": "#0f172a",
                "mid": "#64748b",
                "track": "#e2e8f0",
                "accent": "#1565a8",
                "accent_dark": "#0c4a6e",
                "accent_light": "#e6f4fb",
                "sky": "#7dcfea",
                "navy": "#0c4a6e",
            },
        }

    def list_models(self) -> list[dict[str, Any]]:
        self._discover_models()
        return [
            {
                "model_id": model_id,
                "display_name": info["display_name"],
                "description": info["description"],
                "target": info["target"],
                "n_models": info["n_models"],
                "calibrated": info["calibrated"],
                "submission_filename": _safe_submission_filename(
                    model_id, info.get("manifest", {})
                ),
            }
            for model_id, info in self.available_models.items()
        ]

    def submission_filename(
        self, model_id: str, uploaded_filename: str | None = None
    ) -> str:
        """Return the benchmark CSV filename.

        Most tasks map one model to one benchmark filename. Delivery-mode has
        two benchmark uploads that share the same trained model, so when the
        uploaded file name says dmnw/dmw we preserve that specific target name.
        """
        upload = (uploaded_filename or "").lower()
        if "dmnw" in upload or "non-western" in upload or "non_western" in upload:
            return "dmnw.csv"
        if "dmw" in upload or "western" in upload:
            return "dmw.csv"
        self._discover_models()
        info = self.available_models.get(model_id)
        manifest = info.get("manifest", {}) if info else {}
        return _safe_submission_filename(model_id, manifest)

    def load_model(self, model_id: str) -> bool:
        if model_id in self.loaded_models:
            return True
        self._discover_models()
        info = self.available_models.get(model_id)
        if info is None:
            return False
        try:
            # Ensure compatibility aliases are present immediately before unpickling.
            _register_unpickle_compat_modules()
            if info["kind"] == "mpmae_zip":
                with zipfile.ZipFile(info["path"]) as zf:
                    package = joblib.load(io.BytesIO(zf.read("model.joblib")))
            else:
                package = joblib.load(info["path"])
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", str(exc))
            raise RuntimeError(
                f"Could not load model {model_id!r}: artifact references missing module {missing!r}. "
                "This mllabiome-ii release declares XGBoost, LightGBM, CatBoost, and FLAML in "
                "app/backend/requirements.txt and registers compatibility shims for "
                "ibs_mpmae_runtime, ibs_mpmae_model_defs, and sklearn's private _loss "
                "module. Install/update backend dependencies with: "
                "python -m pip install -r app/backend/requirements.txt"
            ) from exc
        # Two supported package forms:
        # 1. dict schema used by the generic app exporter;
        # 2. object schema used by legacy/custom MPMA-E exporters.
        if isinstance(package, dict):
            if package.get("artifact_schema") != "mllabiome.mpmae.v1":
                return False
        elif not hasattr(package, "predict_proba_from_profile_dataframe"):
            return False
        self.loaded_models[model_id] = package
        return True

    def _predict_package(
        self, package: dict[str, Any], data: pd.DataFrame
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        classes = np.asarray(package["classes"], dtype=int)
        member_probas = []
        for member in package["members"]:
            X_raw = _align_raw(data, member["raw_feature_names"])
            X = _apply_transform_state(member["transform_state"], X_raw)
            member_probas.append(
                _predict_proba(
                    member["estimator"], X, classes, member["transformed_feature_names"]
                )
            )
        stack = np.stack(member_probas, axis=0)
        proba = _aggregate_package(stack, package)
        return proba, member_probas

    def _labels(self, package: dict[str, Any]) -> list[str]:
        return [str(x) for x in package.get("class_labels", ["Negative", "Positive"])]

    def predict(self, model_id: str, features: dict[str, float]) -> dict[str, Any]:
        return self.predict_batch(
            model_id, pd.DataFrame([features]), sample_id_column=None
        )[0]

    def predict_batch(
        self, model_id: str, data: pd.DataFrame, sample_id_column: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if not self.load_model(model_id):
            raise ValueError(
                f"Model '{model_id}' not found. Place an exported model *.zip in app/backend/production_models and restart the backend."
            )
        package = self.loaded_models[model_id]
        if sample_id_column and sample_id_column in data.columns:
            sample_ids = data[sample_id_column].astype(str).tolist()
            feature_df = data.drop(columns=[sample_id_column])
        else:
            sample_ids = [str(i) for i in data.index]
            feature_df = data.copy()
        if not isinstance(package, dict) and hasattr(
            package, "predict_proba_from_profile_dataframe"
        ):
            pred_df = package.predict_proba_from_profile_dataframe(feature_df)
            results = []
            manifest = getattr(package, "manifest", {}) or {}
            negative_label, positive_label = _class_labels_from_manifest(manifest)
            for i, row in pred_df.reset_index(drop=True).iterrows():
                p_pos = float(row.get("proba_positive", 0.5))
                pred = int(row.get("prediction", int(p_pos >= 0.5)))
                results.append(
                    {
                        "prediction": pred,
                        "label": positive_label if pred == 1 else negative_label,
                        "probability": p_pos,
                        "confidence": float(abs(p_pos - 0.5) * 2),
                        "model_id": model_id,
                        "n_models_used": int(
                            manifest.get(
                                "n_members", len(getattr(package, "members", []))
                            )
                        ),
                        "sample_id": sample_ids[i] if i < len(sample_ids) else str(i),
                    }
                )
            return results
        proba, _ = self._predict_package(package, feature_df)
        classes = np.asarray(package["classes"], dtype=int)
        labels = self._labels(package)
        positive_idx = (
            int(
                np.where(classes == int(package.get("positive_class", classes[-1])))[0][
                    0
                ]
            )
            if len(classes)
            else -1
        )
        results = []
        for i, sid in enumerate(sample_ids):
            cls_idx = int(np.argmax(proba[i]))
            pred = int(classes[cls_idx])
            p_pos = (
                float(proba[i, positive_idx])
                if positive_idx >= 0
                else float(proba[i, -1])
            )
            results.append(
                {
                    "prediction": pred,
                    "label": labels[pred] if 0 <= pred < len(labels) else str(pred),
                    "probability": p_pos,
                    "confidence": float(abs(p_pos - 0.5) * 2),
                    "model_id": model_id,
                    "n_models_used": len(package["members"]),
                    "sample_id": sid,
                }
            )
        return results

    def predict_submission_dataframe(
        self,
        model_id: str,
        data: pd.DataFrame,
        sample_id_column: Optional[str] = "sample_id",
    ) -> pd.DataFrame:
        """Return benchmark submission format: sample_id,prediction.

        The ``prediction`` column is the probability assigned to class 1 / the
        package positive class, not the hard class label.
        """
        rows = self.predict_batch(
            model_id=model_id, data=data, sample_id_column=sample_id_column
        )
        return pd.DataFrame(
            {
                "sample_id": [
                    str(row.get("sample_id", i)) for i, row in enumerate(rows)
                ],
                "prediction": [float(row.get("probability", 0.5)) for row in rows],
            }
        )

    def _sample_context(
        self,
        data: pd.DataFrame,
        sample_id_column: Optional[str],
        sample_id: Optional[str],
    ) -> tuple[pd.DataFrame, list[str], list[int]]:
        """Return feature table, sample ids and selected row indices for explanation."""
        if sample_id_column and sample_id_column in data.columns:
            sample_ids = data[sample_id_column].astype(str).tolist()
            feature_df = data.drop(columns=[sample_id_column])
        else:
            sample_ids = [str(i) for i in data.index]
            feature_df = data.copy()
        if sample_id is None or str(sample_id) == "":
            indices = list(range(len(feature_df)))
        else:
            wanted = str(sample_id)
            indices = [i for i, sid in enumerate(sample_ids) if str(sid) == wanted]
            if not indices:
                raise ValueError(
                    f"Sample {wanted!r} was not found in the uploaded table"
                )
            indices = indices[:1]
        return feature_df, sample_ids, indices

    def _generic_candidate_features(
        self,
        package: dict[str, Any],
        feature_df: pd.DataFrame,
        row_idx: int,
        max_features: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Choose deployment-space taxa for local model-agnostic explanation."""
        stats: dict[str, dict[str, list[float]]] = {}
        for member in package.get("members", []):
            raw_names = [str(x) for x in list(member.get("raw_feature_names", []))]
            if not raw_names:
                continue
            X = _align_raw(feature_df, raw_names)
            baseline = np.asarray(
                member.get("raw_feature_baseline", np.zeros(len(raw_names))),
                dtype=np.float64,
            )
            if baseline.size < len(raw_names):
                baseline = np.pad(
                    baseline, (0, len(raw_names) - baseline.size), constant_values=0.0
                )
            for j, name in enumerate(raw_names):
                val = float(X[row_idx, j]) if row_idx < X.shape[0] else 0.0
                base = float(baseline[j]) if j < baseline.size else 0.0
                slot = stats.setdefault(name, {"score": [], "value": []})
                slot["score"].append(abs(val - base))
                slot["value"].append(abs(val))
        ranked = []
        for name, payload in stats.items():
            score = float(np.nanmean(payload.get("score", [0.0])))
            value = float(np.nanmean(payload.get("value", [0.0])))
            ranked.append((name, score, value))
        ranked.sort(key=lambda item: (item[1] > 0, item[1], item[2]), reverse=True)
        names = [
            name for name, _, _ in ranked[: max(1, min(max_features, len(ranked)))]
        ]
        if not names:
            return (
                [],
                np.empty((len(feature_df), 0), dtype=float),
                np.empty(0, dtype=float),
            )
        observed = _align_raw(feature_df, names).astype(float)
        instance = observed[row_idx].astype(float)
        return names, observed, instance

    def _explain_object_package(
        self,
        package: Any,
        data: pd.DataFrame,
        sample_id_column: Optional[str],
        sample_id: Optional[str],
        num_features: int,
        num_samples: int,
    ) -> list[dict[str, Any]]:
        feature_df, provided_ids, selected_indices = self._sample_context(
            data, sample_id_column, sample_id
        )
        X_raw, parsed_sample_ids, raw_names = package._profile_matrix_from_dataframe(
            feature_df
        )
        sample_ids = (
            provided_ids
            if provided_ids and len(provided_ids) == X_raw.shape[0]
            else parsed_sample_ids
        )

        member_probas = []
        for member in package.members:
            X_member = package._build_X_for_member(X_raw, raw_names, member)
            Xt = member.transform.transform(X_member)
            member_probas.append(ibs_runtime._proba_pos(member.estimator, Xt))
        base_p = ibs_runtime._aggregate_predictions(
            np.vstack(member_probas),
            package.weights,
            package.aggregation_strategy,
            package.meta_model,
        )

        manifest = getattr(package, "manifest", {}) or {}
        negative_label, positive_label = _class_labels_from_manifest(manifest)
        n_members = int(manifest.get("n_members", len(getattr(package, "members", []))))
        out: list[dict[str, Any]] = []

        for row_idx in selected_indices:
            sid = sample_ids[row_idx] if row_idx < len(sample_ids) else str(row_idx)
            p_pos = float(base_p[row_idx])
            pred = int(p_pos >= 0.5)
            candidates = np.flatnonzero(np.abs(X_raw[row_idx]) > 1e-12).tolist()
            if not candidates:
                candidates = list(range(len(raw_names)))
            candidates = sorted(
                candidates, key=lambda j: abs(float(X_raw[row_idx, j])), reverse=True
            )
            screen_limit = _bounded_screen_candidate_count(num_features)
            candidates = candidates[: max(1, min(screen_limit, len(candidates)))]
            feature_names = [str(raw_names[j]) for j in candidates]
            observed = (
                X_raw[:, candidates].astype(float)
                if candidates
                else np.empty((X_raw.shape[0], 0), dtype=float)
            )
            instance = (
                X_raw[row_idx, candidates].astype(float)
                if candidates
                else np.empty(0, dtype=float)
            )

            def predict_positive(z):
                z = np.asarray(z, dtype=float)
                if z.ndim == 1:
                    z = z.reshape(1, -1)
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                altered = np.repeat(X_raw[row_idx : row_idx + 1], z.shape[0], axis=0)
                if candidates:
                    altered[:, candidates] = np.clip(z[:, : len(candidates)], 0.0, None)
                alt_member_probas = []
                for member in package.members:
                    X_alt_member = package._build_X_for_member(
                        altered, raw_names, member
                    )
                    Xt_alt = member.transform.transform(X_alt_member)
                    alt_member_probas.append(
                        ibs_runtime._proba_pos(member.estimator, Xt_alt)
                    )
                alt_p = ibs_runtime._aggregate_predictions(
                    np.vstack(alt_member_probas),
                    package.weights,
                    package.aggregation_strategy,
                    package.meta_model,
                )
                return np.asarray(alt_p, dtype=float).reshape(-1)

            def predict_proba(z):
                p = np.clip(predict_positive(z), 0.0, 1.0)
                return np.column_stack([1.0 - p, p])

            candidate_lookup = {str(raw_names[j]): int(j) for j in candidates}
            feature_names, observed, instance = _screen_local_explainer_candidates(
                predict_positive=predict_positive,
                observed=observed,
                instance=instance,
                feature_names=feature_names,
                base_probability=p_pos,
                pool_size=_bounded_candidate_count(num_features),
            )
            candidates = [
                candidate_lookup[name]
                for name in feature_names
                if name in candidate_lookup
            ]
            if candidates:
                observed = X_raw[:, candidates].astype(float)
                instance = X_raw[row_idx, candidates].astype(float)

            shap_features, lime_features = _compute_local_shap_lime(
                predict_positive=predict_positive,
                predict_proba=predict_proba,
                observed=observed,
                instance=instance,
                feature_names=feature_names,
                num_features=num_features,
                num_samples=num_samples,
            )

            out.append(
                {
                    "sample_id": str(sid),
                    "prediction": pred,
                    "label": positive_label if pred == 1 else negative_label,
                    "probability": p_pos,
                    "top_features": shap_features,
                    "shap_features": shap_features,
                    "lime_features": lime_features,
                    "interactions": [],
                    "interaction_method": None,
                    "n_models_explained": n_members,
                    "explanation_method": "local_shap_lime",
                    "explanation_basis": "",
                }
            )
        return out

    def explain_instance(
        self,
        model_id: str,
        data: pd.DataFrame,
        sample_id_column: Optional[str] = None,
        sample_id: Optional[str] = None,
        num_features: int = _FAST_EXPLAIN_MAX_FEATURES,
        num_samples: int = _FAST_EXPLAIN_SHAP_SAMPLES,
    ) -> list[dict[str, Any]]:
        if not self.load_model(model_id):
            raise ValueError(f"Model '{model_id}' not found")
        package = self.loaded_models[model_id]
        if not isinstance(package, dict) and hasattr(
            package, "predict_proba_from_profile_dataframe"
        ):
            return self._explain_object_package(
                package, data, sample_id_column, sample_id, num_features, num_samples
            )

        feature_df, sample_ids, selected_indices = self._sample_context(
            data, sample_id_column, sample_id
        )
        base_proba, _ = self._predict_package(package, feature_df)
        classes = np.asarray(package["classes"], dtype=int)
        labels = self._labels(package)
        positive_class = (
            int(package.get("positive_class", classes[-1])) if len(classes) else 1
        )
        positive_idx = (
            int(np.where(classes == positive_class)[0][0])
            if len(classes) and np.any(classes == positive_class)
            else -1
        )
        results = []

        for row_idx in selected_indices:
            sid = sample_ids[row_idx] if row_idx < len(sample_ids) else str(row_idx)
            pred_idx = int(np.argmax(base_proba[row_idx]))
            pred = (
                int(classes[pred_idx])
                if len(classes)
                else int(base_proba[row_idx, -1] >= 0.5)
            )
            p_pos = (
                float(base_proba[row_idx, positive_idx])
                if positive_idx >= 0
                else float(base_proba[row_idx, -1])
            )

            feature_names, observed, instance = self._generic_candidate_features(
                package=package,
                feature_df=feature_df,
                row_idx=row_idx,
                max_features=_bounded_screen_candidate_count(num_features),
            )

            base_row = feature_df.iloc[[row_idx]].copy()

            def predict_positive(z):
                z = np.asarray(z, dtype=float)
                if z.ndim == 1:
                    z = z.reshape(1, -1)
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                altered = pd.concat([base_row] * z.shape[0], ignore_index=True)
                for col_idx, name in enumerate(feature_names):
                    altered[name] = np.clip(z[:, col_idx], 0.0, None)
                alt_proba, _ = self._predict_package(package, altered)
                col = positive_idx if positive_idx >= 0 else -1
                return np.asarray(alt_proba[:, col], dtype=float).reshape(-1)

            def predict_proba(z):
                p = np.clip(predict_positive(z), 0.0, 1.0)
                return np.column_stack([1.0 - p, p])

            feature_names, observed, instance = _screen_local_explainer_candidates(
                predict_positive=predict_positive,
                observed=observed,
                instance=instance,
                feature_names=feature_names,
                base_probability=p_pos,
                pool_size=_bounded_candidate_count(num_features),
            )

            shap_features, lime_features = _compute_local_shap_lime(
                predict_positive=predict_positive,
                predict_proba=predict_proba,
                observed=observed,
                instance=instance,
                feature_names=feature_names,
                num_features=num_features,
                num_samples=num_samples,
            )

            results.append(
                {
                    "sample_id": str(sid),
                    "prediction": pred,
                    "label": labels[pred] if 0 <= pred < len(labels) else str(pred),
                    "probability": p_pos,
                    "top_features": shap_features,
                    "shap_features": shap_features,
                    "lime_features": lime_features,
                    "interactions": [],
                    "interaction_method": None,
                    "n_models_explained": len(package.get("members", [])),
                    "explanation_method": "local_shap_lime",
                    "explanation_basis": "",
                }
            )
        return results

    def _effect_curves_for_object_package(
        self,
        package: Any,
        data: pd.DataFrame,
        sample_id_column: Optional[str],
        features: list[str],
        num_points: int,
    ) -> list[dict[str, Any]]:
        if sample_id_column and sample_id_column in data.columns:
            feature_df = data.drop(columns=[sample_id_column])
        else:
            feature_df = data.copy()

        X_raw, _, raw_names = package._profile_matrix_from_dataframe(feature_df)
        if not features:
            variances = np.var(X_raw, axis=0)
            order = np.argsort(-variances)[:10]
            features = [raw_names[int(i)] for i in order if variances[int(i)] > 0]

        raw_index = {str(name): i for i, name in enumerate(raw_names)}
        member_count = len(getattr(package, "members", []))
        base_pred = package.predict_proba_from_profile_dataframe(feature_df)
        baseline_mean = float(
            np.mean(
                pd.to_numeric(base_pred["proba_positive"], errors="coerce").fillna(0.5)
            )
        )
        curves: list[dict[str, Any]] = []

        for feature in features:
            if feature not in raw_index:
                continue
            j = raw_index[feature]
            grid = _effect_grid(X_raw[:, j], num_points)
            if grid.size == 0:
                continue
            mean_probability: list[float] = []
            for value in grid:
                X_alt = X_raw.copy()
                X_alt[:, j] = float(value)
                member_probas = []
                for member in package.members:
                    X_member = package._build_X_for_member(X_alt, raw_names, member)
                    Xt = member.transform.transform(X_member)
                    member_probas.append(ibs_runtime._proba_pos(member.estimator, Xt))
                p = ibs_runtime._aggregate_predictions(
                    np.vstack(member_probas),
                    package.weights,
                    package.aggregation_strategy,
                    package.meta_model,
                )
                mean_probability.append(float(np.mean(p)))
            curves.append(
                {
                    "feature": feature,
                    "grid": [float(x) for x in grid],
                    "mean_probability": mean_probability,
                    "centered_effect": [
                        float(x - baseline_mean) for x in mean_probability
                    ],
                    "baseline_mean_probability": baseline_mean,
                    "n_samples": int(X_raw.shape[0]),
                    "n_models": int(member_count),
                }
            )
        return curves

    def _effect_curves_for_dict_package(
        self,
        package: dict[str, Any],
        data: pd.DataFrame,
        sample_id_column: Optional[str],
        features: list[str],
        num_points: int,
    ) -> list[dict[str, Any]]:
        if sample_id_column and sample_id_column in data.columns:
            feature_df = data.drop(columns=[sample_id_column])
        else:
            feature_df = data.copy()
        numeric = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if not features:
            variances = numeric.var(axis=0).sort_values(ascending=False)
            features = [
                str(c) for c in variances.index[:10] if float(variances.loc[c]) > 0
            ]
        base_proba, _ = self._predict_package(package, numeric)
        classes = np.asarray(package["classes"], dtype=int)
        positive_idx = (
            int(
                np.where(classes == int(package.get("positive_class", classes[-1])))[0][
                    0
                ]
            )
            if len(classes)
            else -1
        )
        baseline_mean = float(
            np.mean(
                base_proba[:, positive_idx] if positive_idx >= 0 else base_proba[:, -1]
            )
        )
        col_lookup = {str(c): c for c in numeric.columns}
        curves: list[dict[str, Any]] = []
        for feature in features:
            col = col_lookup.get(feature)
            if col is None:
                continue
            grid = _effect_grid(numeric[col].to_numpy(dtype=np.float32), num_points)
            if grid.size == 0:
                continue
            mean_probability: list[float] = []
            for value in grid:
                altered = numeric.copy()
                altered[col] = float(value)
                proba, _ = self._predict_package(package, altered)
                pos = proba[:, positive_idx] if positive_idx >= 0 else proba[:, -1]
                mean_probability.append(float(np.mean(pos)))
            curves.append(
                {
                    "feature": feature,
                    "grid": [float(x) for x in grid],
                    "mean_probability": mean_probability,
                    "centered_effect": [
                        float(x - baseline_mean) for x in mean_probability
                    ],
                    "baseline_mean_probability": baseline_mean,
                    "n_samples": int(len(numeric)),
                    "n_models": int(len(package.get("members", []))),
                }
            )
        return curves

    def effect_curves(
        self,
        model_id: str,
        data: Optional[pd.DataFrame] = None,
        sample_id_column: Optional[str] = None,
        features: Optional[list[str] | str] = None,
        num_points: int = 25,
    ) -> list[dict[str, Any]]:
        if not self.load_model(model_id):
            raise ValueError(f"Model '{model_id}' not found")
        package = self.loaded_models[model_id]
        requested_features = _clean_feature_list(features)

        if data is None:
            if isinstance(package, dict):
                curves = list(package.get("effect_curves", []))
            else:
                manifest = getattr(package, "manifest", {}) or {}
                curves = list(manifest.get("effect_curves", []))
            if requested_features:
                wanted = set(requested_features)
                curves = [c for c in curves if str(c.get("feature")) in wanted]
            return curves

        if isinstance(package, dict):
            return self._effect_curves_for_dict_package(
                package, data, sample_id_column, requested_features, num_points
            )
        if hasattr(package, "predict_proba_from_profile_dataframe"):
            return self._effect_curves_for_object_package(
                package, data, sample_id_column, requested_features, num_points
            )
        return []


production_service = ProductionModelService()
