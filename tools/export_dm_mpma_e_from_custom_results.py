from __future__ import annotations

# Thread pinning before numpy/scipy/sklearn imports.
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import importlib.util
import io
import json
import math
import sqlite3
import sys
import zipfile
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import FactorAnalysis, FastICA, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron,
                                  Ridge, RidgeClassifier, RidgeClassifierCV,
                                  SGDClassifier)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import (KNeighborsClassifier, NearestCentroid,
                               RadiusNeighborsClassifier)
from sklearn.preprocessing import (PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.svm import SVC

try:
    import polars as pl
except Exception:  # pragma: no cover - polars is expected in the manuscript environment
    pl = None


# During export, joblib should record deployment classes as coming from
# ibs_mpmae_runtime, which is provided by mllabiome-ii. The exporter itself
# is intentionally not packaged inside the model zip.
RUNTIME_MODULE = "ibs_mpmae_runtime"
sys.modules.setdefault(RUNTIME_MODULE, sys.modules[__name__])

POSITIVE_CLASS = 1
DEFAULT_OPTIMIZE_METRIC = "nMCC"
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


def _load_module(path: Path | None):
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Sweep script does not exist: {path}")

    # Python's default importlib source suffix list is case-sensitive on Unix-like
    # systems and recognizes .py, but not manuscript filenames such as
    # SWEEP_CONFIGS-SCZ-BASE.PY.  Use SourceFileLoader explicitly so both .py
    # and .PY sweep scripts can be imported without renaming the historical files.
    module_name = "ibs_mpmae_model_defs"
    loader = SourceFileLoader(module_name, str(path))
    spec = importlib.util.spec_from_loader(module_name, loader, origin=str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import sweep script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _read_selected_unit(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "ensembling" / "selected_unit.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected ensemble file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("inner_val_best_ensemble")
    if not isinstance(block, dict):
        raise ValueError(f"{path} does not contain inner_val_best_ensemble")
    members = block.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"{path}: inner_val_best_ensemble.members is empty or invalid")
    return data


def _read_configs(experiment_dir: Path) -> pd.DataFrame:
    db_path = experiment_dir / "configs.db"
    tsv_path = experiment_dir / "configs.tsv"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql("SELECT * FROM configs", conn)
        finally:
            conn.close()
    elif tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t")
    else:
        raise FileNotFoundError(
            f"No configs.db or configs.tsv found in {experiment_dir}"
        )
    if "config_id" not in df.columns:
        raise ValueError("Config registry must contain config_id")
    # Normalize expected aliases from manuscript scripts.
    if "transform" not in df.columns and "count_transformation" in df.columns:
        df["transform"] = df["count_transformation"]
    if "model" not in df.columns and "learner" in df.columns:
        df["model"] = df["learner"]
    if "resolution" not in df.columns and "taxa_resolution" in df.columns:
        df["resolution"] = df["taxa_resolution"]
    for col in ("transform", "model", "resolution"):
        if col not in df.columns:
            raise ValueError(f"Config registry is missing required column {col!r}")
    if "levels" not in df.columns:
        df["levels"] = ""
    df["config_id"] = df["config_id"].astype(str)
    return df


def _parse_levels(row: pd.Series) -> tuple[str, ...]:
    raw = str(row.get("levels", "") or "").strip()
    if raw and raw.lower() not in {"nan", "none"}:
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    res = str(row.get("resolution", "") or "").strip()
    if res in SINGLE_LEVELS:
        return (res,)
    if "-" in res:
        lo, hi = [x.strip() for x in res.split("-", 1)]
        if lo in SINGLE_LEVELS and hi in SINGLE_LEVELS:
            a, b = SINGLE_LEVELS.index(lo), SINGLE_LEVELS.index(hi)
            if a > b:
                a, b = b, a
            return tuple(SINGLE_LEVELS[a : b + 1])
    if "+" in res:
        parts = tuple(x.strip() for x in res.split("+") if x.strip())
        if parts and all(x in SINGLE_LEVELS for x in parts):
            return parts
    raise ValueError(
        f"Cannot infer levels for config {row.get('config_id')} resolution={res!r}"
    )


def _infer_study_ids(data_dir: Path) -> list[str]:
    ids = []
    for p in sorted(data_dir.glob("*_profiles.tsv")):
        ids.append(p.name[: -len("_profiles.tsv")])
    if not ids:
        raise FileNotFoundError(f"No *_profiles.tsv files found in {data_dir}")
    return ids


def _truncate_lineage(name: str, depth: int) -> str:
    raw = str(name).strip()
    if "|" in raw:
        return (
            "|".join(raw.split("|")[:depth])
            .replace("k__", "d__", 1)
            .replace("|", "___")
        )
    return "___".join(raw.split("___")[:depth]).replace("k__", "d__", 1)


def _feature_name_ok(name: str, level: str) -> bool:
    must_have, must_not = LEVEL_MARKERS[level]
    if level == "domain":
        has = str(name).startswith("d__")
    else:
        has = any(m in str(name) for m in must_have)
    return bool(has and not any(m in str(name) for m in must_not))


def _renormalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32).copy()
    X[X < 0] = 0.0
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (X / s).astype(np.float32)


def _load_one_study(
    data_dir: Path,
    study_id: str,
    sample_id_col: str,
    target_col: str,
    positive_label: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    prof_path = data_dir / f"{study_id}_profiles.tsv"
    meta_path = data_dir / f"{study_id}_metadata.tsv"
    if not prof_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing profile/metadata pair for {study_id}: {prof_path} / {meta_path}"
        )
    prof = pd.read_csv(prof_path, sep="\t", index_col=0)
    prof.index = prof.index.astype(str).str.strip()
    rank_prefixes = ("d__", "k__", "p__", "c__", "o__", "f__", "g__", "s__", "t__")
    prof = prof[prof.index.str.startswith(rank_prefixes)]
    prof = prof.T
    prof.index = prof.index.astype(str).str.strip()

    meta = pd.read_csv(meta_path, sep="\t", dtype=str)
    if sample_id_col not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: sample_id_col})
    if target_col not in meta.columns:
        raise ValueError(f"{meta_path} is missing target column {target_col!r}")
    meta[sample_id_col] = meta[sample_id_col].astype(str).str.strip()
    meta = meta.drop_duplicates(subset=[sample_id_col]).set_index(sample_id_col)

    common = prof.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError(f"No overlapping samples for study {study_id}")
    prof = prof.loc[common]
    meta = meta.loc[common]
    X = (
        prof.apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    y = (
        meta[target_col].astype(str).str.strip().to_numpy() == str(positive_label)
    ).astype(np.int8)
    sample_ids = [f"{study_id}::{sid}" for sid in common.astype(str).tolist()]
    return X, y, sample_ids, list(prof.columns.astype(str))


@dataclass
class LevelArrays:
    X: np.ndarray
    y: np.ndarray
    sample_ids: list[str]
    feature_names: list[str]


def _build_level_arrays(
    data_dir: Path,
    study_ids: Iterable[str],
    levels_needed: Iterable[str],
    sample_id_col: str,
    target_col: str,
    positive_label: str,
) -> dict[tuple[str, str], LevelArrays]:
    arrays: dict[tuple[str, str], LevelArrays] = {}
    levels = sorted(set(levels_needed), key=lambda x: SINGLE_LEVELS.index(x))
    for study in study_ids:
        X_raw, y, sample_ids, raw_names = _load_one_study(
            data_dir, str(study), sample_id_col, target_col, positive_label
        )
        for level in levels:
            depth = LEVEL_DEPTH[level]
            groups: dict[str, list[int]] = {}
            for j, raw in enumerate(raw_names):
                key = _truncate_lineage(raw, depth)
                groups.setdefault(key, []).append(j)
            names = sorted(n for n in groups if _feature_name_ok(n, level))
            if not names:
                raise ValueError(f"No features survive for study={study} level={level}")
            X = np.zeros((X_raw.shape[0], len(names)), dtype=np.float32)
            for j, name in enumerate(names):
                X[:, j] = X_raw[:, groups[name]].sum(axis=1)
            arrays[(str(study), level)] = LevelArrays(
                _renormalize(X), y.astype(np.int8), sample_ids, names
            )
    return arrays


def _align_study_to_features(arr: LevelArrays, union_features: list[str]) -> np.ndarray:
    pos = {f: i for i, f in enumerate(arr.feature_names)}
    out = np.zeros((arr.X.shape[0], len(union_features)), dtype=np.float32)
    for j, f in enumerate(union_features):
        if f in pos:
            out[:, j] = arr.X[:, pos[f]]
    # The manuscript ResolutionData renormalised after alignment.
    return _renormalize(out)


def _build_member_training_matrix(
    arrays: dict[tuple[str, str], LevelArrays],
    study_ids: list[str],
    levels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    level_unions: dict[str, list[str]] = {}
    for level in levels:
        feats: set[str] = set()
        for study in study_ids:
            feats.update(arrays[(str(study), level)].feature_names)
        level_unions[level] = sorted(feats)

    X_blocks_by_study = []
    y_parts = []
    sid_parts: list[str] = []
    for study in study_ids:
        blocks = []
        for level in levels:
            arr = arrays[(str(study), level)]
            blocks.append(_align_study_to_features(arr, level_unions[level]))
        X_study = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        X_blocks_by_study.append(X_study)
        first = arrays[(str(study), levels[0])]
        y_parts.append(first.y)
        sid_parts.extend(first.sample_ids)
    X_full = np.concatenate(X_blocks_by_study, axis=0).astype(np.float32)
    y_full = np.concatenate(y_parts, axis=0).astype(np.int8)
    schema = {
        "levels": list(levels),
        "level_features": level_unions,
        "feature_names": [f for lv in levels for f in level_unions[lv]],
    }
    return X_full, y_full, sid_parts, schema


# ---------------------------------------------------------------------------
# LAMPP wide-CSV data loading helpers
# ---------------------------------------------------------------------------

LAMPP_META_COLS = {"label", "sample_id", "subject_id", "study_id"}


def _truncate_lampp_lineage(col: str, depth: int) -> str:
    raw = str(col).strip()
    if "|" in raw:
        return (
            "|".join(raw.split("|")[:depth])
            .replace("k__", "d__", 1)
            .replace("|", "___")
        )
    return _truncate_lineage(raw, depth)


def _encode_lampp_labels(values, positive_value: str) -> np.ndarray:
    positive_raw = str(positive_value).strip()
    positive_num: float | None
    try:
        positive_num = float(positive_raw)
    except Exception:
        positive_num = None
    out: list[int] = []
    for value in values:
        if pd.isna(value):
            out.append(0)
            continue
        if positive_num is not None:
            try:
                out.append(1 if float(value) == positive_num else 0)
                continue
            except Exception:
                pass
        out.append(1 if str(value).strip() == positive_raw else 0)
    return np.asarray(out, dtype=np.int8)


def _parse_study_ids_arg(raw: str | None) -> list[Any]:
    if raw is None:
        return []
    out: list[Any] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except Exception:
            out.append(item)
    return out


def _filter_lampp_studies(
    df: pd.DataFrame, study_id_col: str, study_ids: list[Any]
) -> pd.DataFrame:
    if not study_ids:
        return df.copy()
    if study_id_col not in df.columns:
        raise ValueError(
            f"--study-ids was supplied, but CSV is missing study id column {study_id_col!r}"
        )
    wanted_str = {str(x) for x in study_ids}
    mask = df[study_id_col].astype(str).isin(wanted_str)
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(
            f"No rows remain after filtering {study_id_col!r} to {sorted(wanted_str)}. "
            f"Check --study-ids against the DM training CSV."
        )
    return out


def _build_lampp_level_arrays(
    csv_path: Path,
    levels_needed: Iterable[str],
    sample_id_col: str,
    target_col: str,
    positive_value: str,
    metadata_cols: set[str] | None = None,
    study_id_col: str = "study_id",
    study_ids: list[Any] | None = None,
) -> dict[str, LevelArrays]:
    csv_path = Path(csv_path).expanduser().resolve()
    df = pd.read_csv(csv_path, low_memory=False)
    df = _filter_lampp_studies(df, study_id_col, list(study_ids or []))
    if target_col not in df.columns:
        raise ValueError(f"CSV missing target column {target_col!r}")
    sample_ids = (
        df[sample_id_col].astype(str).tolist()
        if sample_id_col in df.columns
        else [str(i) for i in df.index]
    )
    y = _encode_lampp_labels(df[target_col].tolist(), positive_value)

    meta = set(metadata_cols or LAMPP_META_COLS) | {
        sample_id_col,
        target_col,
        study_id_col,
    }
    src_feat_cols = [str(c) for c in df.columns if str(c) not in meta and "|" in str(c)]
    if not src_feat_cols:
        # Also support already-normalized d__...___p__... feature columns.
        src_feat_cols = [
            str(c)
            for c in df.columns
            if str(c) not in meta
            and (str(c).startswith(("d__", "k__")) or "___" in str(c))
        ]
    if not src_feat_cols:
        raise ValueError(
            "No taxonomic feature columns found. Expected MetaPhlAn lineage columns containing '|' or '___'."
        )

    src_matrix = (
        df[src_feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    out: dict[str, LevelArrays] = {}
    for level in sorted(set(levels_needed), key=lambda x: SINGLE_LEVELS.index(x)):
        depth = LEVEL_DEPTH[level]
        groups: dict[str, list[int]] = {}
        for col_idx, col in enumerate(src_feat_cols):
            key = _truncate_lampp_lineage(col, depth)
            groups.setdefault(key, []).append(col_idx)
        names = sorted(name for name in groups if _feature_name_ok(name, level))
        if not names:
            raise ValueError(
                f"No features survive filter for level={level!r}; check lineage formatting in {csv_path}"
            )
        X = np.zeros((len(sample_ids), len(names)), dtype=np.float32)
        for j, name in enumerate(names):
            X[:, j] = src_matrix[:, groups[name]].sum(axis=1)
        out[level] = LevelArrays(_renormalize(X), y, sample_ids, names)
    return out


def _build_lampp_member_training_matrix(
    arrays: dict[str, LevelArrays],
    levels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    blocks = [arrays[level].X for level in levels]
    X_full = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    first = arrays[levels[0]]
    schema = {
        "levels": list(levels),
        "level_features": {
            level: list(arrays[level].feature_names) for level in levels
        },
        "feature_names": [f for level in levels for f in arrays[level].feature_names],
    }
    return (
        X_full.astype(np.float32),
        first.y.astype(np.int8),
        list(first.sample_ids),
        schema,
    )


def _versioned_model_out_path(
    out: Path | None, model_id: str, artifact_version: str, overwrite: bool = False
) -> Path:
    if out is None:
        safe_version = (
            str(artifact_version).strip().replace("/", "_").replace(" ", "_") or "v0"
        )
        out_path = Path(f"{model_id}_model_{safe_version}.zip")
    else:
        out_path = Path(out)
    out_path = out_path.expanduser().resolve()
    if overwrite or not out_path.exists():
        return out_path
    for idx in range(2, 1000):
        candidate = out_path.with_name(f"{out_path.stem}_build{idx}{out_path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"Could not find an available versioned output path near {out_path}"
    )


class RidgeProba(RidgeClassifier):
    def predict_proba(self, X):
        df = self.decision_function(X)
        if np.ndim(df) == 1:
            df = np.column_stack([-df, df])
        df = df - df.max(axis=1, keepdims=True)
        e = np.exp(df)
        return e / e.sum(axis=1, keepdims=True)


class NearestCentroidProba(NearestCentroid):
    def predict_proba(self, X):
        d = np.linalg.norm(
            np.asarray(X)[:, None, :] - self.centroids_[None, :, :], axis=2
        )
        z = -d
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class FLAMLClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        time_budget=600,
        metric="roc_auc",
        estimator_list=None,
        n_jobs=1,
        random_state=42,
        verbose=0,
        **kwargs,
    ):
        self.time_budget = time_budget
        self.metric = metric
        self.estimator_list = estimator_list
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, X, y):
        from flaml import AutoML

        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        metric = self.metric
        if len(self.classes_) > 2 and metric == "roc_auc":
            metric = "roc_auc_ovr"
        self.model_ = AutoML()
        fit_kwargs = dict(
            X_train=X,
            y_train=y,
            task="classification",
            time_budget=self.time_budget,
            metric=metric,
            n_jobs=self.n_jobs,
            seed=self.random_state,
            verbose=self.verbose,
            **self.kwargs,
        )
        if self.estimator_list is not None:
            fit_kwargs["estimator_list"] = self.estimator_list
        self.model_.fit(**fit_kwargs)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))


class _SoftmaxMixin:
    """Add a stable binary/multiclass probability view to linear margin models."""

    def predict_proba(self, X):
        df = self.decision_function(X)
        if np.ndim(df) == 1:
            df = np.column_stack([-df, df])
        df = df - df.max(axis=1, keepdims=True)
        e = np.exp(df)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


class SGDProba(_SoftmaxMixin, SGDClassifier):
    pass


class PAProba(_SoftmaxMixin, PassiveAggressiveClassifier):
    pass


class PerceptronProba(_SoftmaxMixin, Perceptron):
    pass


class RidgeCVProba(_SoftmaxMixin, RidgeClassifierCV):
    pass


class RadNCProba(BaseEstimator, ClassifierMixin):
    def __init__(self, radius=0.5, weights="distance", metric="minkowski"):
        self.radius = radius
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self._rnc = RadiusNeighborsClassifier(
            radius=self.radius,
            weights=self.weights,
            metric=self.metric,
            outlier_label=-1,
        )
        self._rnc.fit(X, y)
        self._knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
        self._knn.fit(X, y)
        self.classes_ = self._rnc.classes_
        return self

    def predict_proba(self, X):
        try:
            p = self._rnc.predict_proba(X)
            pred = self._rnc.predict(X)
            outlier = pred == -1
            if np.any(outlier):
                p[outlier] = self._knn.predict_proba(np.asarray(X)[outlier])
            return p
        except Exception:
            return self._knn.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class FAClf(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=50, max_iter=1000, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        nc = max(1, min(int(self.n_components), X.shape[1], X.shape[0] - 1))
        self._fa = FactorAnalysis(
            n_components=nc, max_iter=self.max_iter, random_state=self.random_state
        )
        self._lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self._lr.fit(self._fa.fit_transform(X), y)
        self.classes_ = self._lr.classes_
        return self

    def predict_proba(self, X):
        return self._lr.predict_proba(self._fa.transform(X))

    def predict(self, X):
        return self._lr.predict(self._fa.transform(X))


class DecompLR(BaseEstimator, ClassifierMixin):
    def __init__(self, kind="svd", n_components=50, random_state=42, **kwargs):
        self.kind = kind
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X, y):
        nc = max(1, min(int(self.n_components), X.shape[1], X.shape[0] - 1))
        if self.kind == "svd":
            self._decomp = TruncatedSVD(n_components=nc, random_state=self.random_state)
        elif self.kind == "ica":
            self._decomp = FastICA(
                n_components=nc,
                random_state=self.random_state,
                max_iter=1000,
                **self.kwargs,
            )
        else:
            raise ValueError(self.kind)
        Z = self._decomp.fit_transform(X)
        self._lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self._lr.fit(Z, y)
        self.classes_ = self._lr.classes_
        return self

    def predict_proba(self, X):
        return self._lr.predict_proba(self._decomp.transform(X))

    def predict(self, X):
        return self._lr.predict(self._decomp.transform(X))


class GMMClf(BaseEstimator, ClassifierMixin):
    def __init__(
        self, n_components=2, covariance_type="full", bayesian=False, random_state=42
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.bayesian = bayesian
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.log_priors_ = {}
        for c in self.classes_:
            Xc = np.asarray(X)[np.asarray(y) == c]
            nc = max(1, min(int(self.n_components), len(Xc)))
            cls = BayesianGaussianMixture if self.bayesian else GaussianMixture
            gm = cls(
                n_components=nc,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            gm.fit(Xc)
            self.models_[c] = gm
            self.log_priors_[c] = float(np.log(len(Xc) / max(len(y), 1) + 1e-300))
        return self

    def predict_proba(self, X):
        ll = np.column_stack(
            [
                self.models_[c].score_samples(X) + self.log_priors_[c]
                for c in self.classes_
            ]
        )
        ll = ll - ll.max(axis=1, keepdims=True)
        e = np.exp(ll)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _bagging_classifier(estimator, **kwargs):
    from sklearn.ensemble import BaggingClassifier

    try:
        return BaggingClassifier(estimator=estimator, **kwargs)
    except TypeError:  # older sklearn
        return BaggingClassifier(base_estimator=estimator, **kwargs)


def _model_from_sweep_module(module, model_name: str):
    if module is None or not hasattr(module, "_build_models"):
        return None
    try:
        lookup = {str(k): v for k, v in module._build_models()}
        if model_name in lookup:
            return clone(lookup[model_name])
    except Exception:
        return None
    return None


def _manuscript_model_factory(model_name: str, module=None):
    """Factory for manuscript model names recorded in configs.db/configs.tsv.

    The completed manuscript registry may contain models that are no longer
    active in _build_models(). This function reconstructs those exact names,
    preferring top-level classes from --sweep-script so joblib can pickle them
    through the bundled ibs_mpmae_model_defs.py module.
    """
    name = str(model_name)

    def _cls(attr, default):
        return getattr(module, attr, default) if module is not None else default

    RidgeProbaCls = _cls("RidgeProba", RidgeProba)
    RidgeCVProbaCls = _cls("RidgeCVProba", RidgeCVProba)
    SGDProbaCls = _cls("SGDProba", SGDProba)
    PAProbaCls = _cls("PAProba", PAProba)
    PerceptronProbaCls = _cls("PerceptronProba", PerceptronProba)
    RadNCProbaCls = _cls("RadNCProba", RadNCProba)
    FAClfCls = _cls("FAClf", FAClf)
    FLAMLClassifierCls = _cls("FLAMLClassifier", FLAMLClassifier)

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.decomposition import FactorAnalysis, FastICA, TruncatedSVD
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)
    from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                                  ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import (LogisticRegression,
                                      PassiveAggressiveClassifier, Perceptron,
                                      RidgeClassifier, RidgeClassifierCV,
                                      SGDClassifier)
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
    from sklearn.naive_bayes import (BernoulliNB, ComplementNB, GaussianNB,
                                     MultinomialNB)
    from sklearn.neighbors import (KNeighborsClassifier, NearestCentroid,
                                   RadiusNeighborsClassifier)
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
    from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.tree import DecisionTreeClassifier

    # AutoML
    if name == "FLAML_600s":
        return FLAMLClassifierCls(
            time_budget=600, metric="roc_auc", n_jobs=1, random_state=42
        )
    if name == "FLAML_60s":
        return FLAMLClassifierCls(
            time_budget=60, metric="roc_auc", n_jobs=1, random_state=42
        )

    # Random forests / extra trees / decision trees / boosting
    if name == "RF_default":
        return RandomForestClassifier(n_jobs=1, random_state=42)
    if name == "RF_1000_msl5":
        return RandomForestClassifier(
            n_estimators=1000, min_samples_leaf=5, n_jobs=1, random_state=42
        )
    if name == "RF_1000_msl10_bal":
        return RandomForestClassifier(
            n_estimators=1000,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        )
    if name.startswith("RF_"):
        import re

        m = re.match(r"RF_(\d+)_msl(\d+)(?:_bal)?$", name)
        if m:
            return RandomForestClassifier(
                n_estimators=int(m.group(1)),
                min_samples_leaf=int(m.group(2)),
                class_weight="balanced_subsample" if name.endswith("_bal") else None,
                n_jobs=1,
                random_state=42,
            )
    if name == "ET_default":
        return ExtraTreesClassifier(n_jobs=1, random_state=42)
    if name.startswith("ET_"):
        import re

        m = re.match(r"ET_(\d+)(?:_msl(\d+))?$", name)
        if m:
            return ExtraTreesClassifier(
                n_estimators=int(m.group(1)),
                min_samples_leaf=int(m.group(2) or 1),
                n_jobs=1,
                random_state=42,
            )
    if name in {"GT_default", "GBM_default"}:
        return GradientBoostingClassifier(random_state=42)
    if name == "GBM_300":
        return GradientBoostingClassifier(n_estimators=300, random_state=42)
    if name == "GBM_300_d3":
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=3, random_state=42
        )
    if name == "GBM_300_d4":
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=4, random_state=42
        )
    if name in {"HistGB_default", "HistGB"}:
        return HistGradientBoostingClassifier(random_state=42)
    if name == "HistGB_lr005_d3":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=7, random_state=42
        )
    if name == "HistGB_lr005_d5":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=31, random_state=42
        )
    if name == "HistGB_lr001_d3":
        return HistGradientBoostingClassifier(
            learning_rate=0.01, max_leaf_nodes=7, random_state=42
        )
    if name == "HistGB_lr001_l15":
        return HistGradientBoostingClassifier(
            learning_rate=0.01, max_leaf_nodes=15, random_state=42
        )
    if name == "DT_default":
        return DecisionTreeClassifier(random_state=42)
    if name.startswith("Ada_"):
        return AdaBoostClassifier(n_estimators=int(name.split("_")[1]), random_state=42)

    # Naive Bayes
    if name == "BNB":
        return BernoulliNB()
    if name.startswith("BNB_a"):
        return BernoulliNB(alpha=float(name.split("_a", 1)[1]) / 10.0)
    if name == "GNB":
        return GaussianNB()
    if name.startswith("GNB_vs"):
        return GaussianNB(var_smoothing=float(name.split("vs", 1)[1]))
    if name.startswith("CNB_a"):
        return ComplementNB(alpha=float(name.split("_a", 1)[1]) / 10.0)
    if name == "MNB_binned_a01":
        return Pipeline(
            [
                (
                    "disc",
                    KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile"),
                ),
                ("clf", MultinomialNB(alpha=0.1)),
            ]
        )

    # Linear / ridge / SGD / PA
    if name in {"LR_default", "LR_C1"}:
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    if name.startswith("LR_l2_C"):
        c = name.split("LR_l2_C", 1)[1].replace("_bal", "")
        C = {"001": 0.01, "01": 0.1, "1": 1.0, "10": 10.0}.get(c, float(c))
        return LogisticRegression(
            C=C,
            max_iter=3000,
            solver="saga",
            class_weight="balanced" if name.endswith("_bal") else None,
            random_state=42,
        )
    if name.startswith("LR_l1_C"):
        c = name.split("LR_l1_C", 1)[1].replace("_bal", "")
        C = {"001": 0.01, "01": 0.1, "1": 1.0, "10": 10.0}.get(c, float(c))
        return LogisticRegression(
            penalty="l1",
            C=C,
            max_iter=3000,
            solver="saga",
            class_weight="balanced" if name.endswith("_bal") else None,
            random_state=42,
        )
    if name.startswith("LR_en_l"):
        # Example: LR_en_l050_C01, LR_en_l075_C1_bal
        import re

        m = re.match(r"LR_en_l(\d+)_C(\d+)(?:_bal)?$", name)
        if m:
            l1_ratio = int(m.group(1)) / 100.0
            cstr = m.group(2)
            C = {"001": 0.01, "01": 0.1, "1": 1.0, "10": 10.0}.get(cstr, float(cstr))
            return LogisticRegression(
                penalty="elasticnet",
                C=C,
                max_iter=3000,
                solver="saga",
                l1_ratio=l1_ratio,
                class_weight="balanced" if name.endswith("_bal") else None,
                random_state=42,
            )
    if name == "RidgeProba_default":
        return RidgeProbaCls(random_state=42)
    if name.startswith("Ridge_a"):
        aval = name.split("Ridge_a", 1)[1]
        alpha = {"001": 0.01, "01": 0.1, "1": 1.0, "10": 10.0, "100": 100.0}.get(
            aval, float(aval)
        )
        return RidgeProbaCls(alpha=alpha, random_state=42)
    if name in {"SGD_log", "SGD_log_l2"}:
        return SGDProbaCls(loss="log_loss", penalty="l2", random_state=42)
    if name == "SGD_log_l1":
        return SGDProbaCls(loss="log_loss", penalty="l1", random_state=42)
    if name in {"SGD_en", "SGD_log_en"}:
        return SGDProbaCls(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.5, random_state=42
        )
    if name == "SGD_hinge_l2":
        return SGDProbaCls(loss="hinge", penalty="l2", random_state=42)
    if name == "SGD_hinge_l1":
        return SGDProbaCls(loss="hinge", penalty="l1", random_state=42)
    if name == "SGD_hinge_en":
        return SGDProbaCls(
            loss="hinge", penalty="elasticnet", l1_ratio=0.15, random_state=42
        )
    if name in {"SGD_modhuber", "SGD_huber_l2"}:
        return SGDProbaCls(loss="modified_huber", penalty="l2", random_state=42)
    if name == "SGD_huber_en":
        return SGDProbaCls(
            loss="modified_huber", penalty="elasticnet", l1_ratio=0.15, random_state=42
        )
    if name == "PA_default":
        return PAProbaCls(random_state=42)
    if name.startswith("PA_C"):
        return PAProbaCls(
            C=float(name.split("PA_C", 1)[1])
            / (1000.0 if name.endswith("001") else 10.0),
            random_state=42,
        )
    if name == "PA_sq":
        return PAProbaCls(loss="squared_hinge", random_state=42)
    if name.startswith("PA_sq_C"):
        raw = name.split("PA_sq_C", 1)[1]
        C = {"001": 0.01, "01": 0.1, "10": 10.0}.get(raw, float(raw))
        return PAProbaCls(loss="squared_hinge", C=C, random_state=42)

    # LDA / QDA
    if name == "LDA_lsqr_auto":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "LDA_lsqr_08":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.8)
    if name == "LDA_lsqr_05":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.5)
    if name == "LDA_lsqr_02":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.2)
    if name == "LDA_eigen_auto":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    if name == "QDA_reg01":
        return QuadraticDiscriminantAnalysis(reg_param=0.1)
    if name == "QDA_reg05":
        return QuadraticDiscriminantAnalysis(reg_param=0.5)
    if name == "OASQDA_default":
        return QuadraticDiscriminantAnalysis(reg_param=0.01)

    # SVM / calibrated SVM
    if name.startswith("SVC_rbf_C"):
        raw = name.split("SVC_rbf_C", 1)[1].replace("_bal", "")
        C = {"01": 0.1, "1": 1.0, "10": 10.0, "100": 100.0}.get(raw, float(raw))
        return SVC(
            kernel="rbf",
            C=C,
            gamma="scale",
            probability=True,
            class_weight="balanced" if name.endswith("_bal") else None,
            random_state=42,
        )
    if name.startswith("NuSVC_"):
        nu = float("0." + name.split("_", 1)[1])
        return NuSVC(
            nu=nu, kernel="rbf", gamma="scale", probability=True, random_state=42
        )
    if name.startswith("LSVC_C"):
        raw = name.split("LSVC_C", 1)[1]
        C = {"001": 0.01, "01": 0.1, "1": 1.0, "10": 10.0}.get(raw, float(raw))
        return CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=5000, random_state=42), method="sigmoid", cv=3
        )
    if name.startswith("CalibLSVC_"):
        parts = name.split("_")
        method = "isotonic" if "iso" in parts[1] else "sigmoid"
        C = 1.0
        if parts[-1].startswith("C"):
            raw = parts[-1][1:]
            C = {"01": 0.1, "1": 1.0, "10": 10.0}.get(raw, float(raw))
        return CalibratedClassifierCV(
            LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=42),
            method=method,
            cv=3,
        )
    if name.startswith("CalibRidge_"):
        method = "isotonic" if "iso" in name else "sigmoid"
        alpha = 0.1 if name.endswith("a01") else 1.0
        return CalibratedClassifierCV(
            RidgeClassifier(alpha=alpha, random_state=42), method=method, cv=3
        )

    # Neighbours / centroid / semi-supervised
    if name == "kNN_default":
        return KNeighborsClassifier()
    if name.startswith("kNN_"):
        parts = name.split("_")
        n_neighbors = int(parts[1])
        metric = "minkowski"
        weights = "uniform"
        if "cos" in parts:
            metric = "cosine"
        if "man" in parts:
            metric = "manhattan"
        if "wt" in parts or "dist" in parts:
            weights = "distance"
        return KNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric, weights=weights, n_jobs=1
        )
    if name in {"NearestCentroid_raw", "NC_no_shrink"}:
        return NearestCentroidProba()
    if name in {"NCProba"}:
        return NearestCentroidProba()
    if name.startswith("NearestCentroid_shrink") or name.startswith("NC_shrink_"):
        val = name.rsplit("_", 1)[-1].replace("shrink", "")
        shrink = {"01": 0.1, "02": 0.2, "05": 0.5, "1": 1.0, "2": 2.0}.get(val, 0.2)
        return NearestCentroidProba(shrink_threshold=shrink)
    if name.startswith("RadNC"):
        return RadNCProbaCls()
    if name == "LabelSpread_rbf":
        return LabelSpreading(kernel="rbf", gamma=0.25, alpha=0.2, max_iter=1000)
    if name.startswith("SelfTrain_RF"):
        threshold = 0.9 if name.endswith("09") else 0.75
        return SelfTrainingClassifier(
            RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42),
            threshold=threshold,
        )
    if name.startswith("SelfTrain_LR"):
        threshold = 0.9 if name.endswith("09") else 0.75
        return SelfTrainingClassifier(
            LogisticRegression(C=0.1, max_iter=1000, solver="saga", random_state=42),
            threshold=threshold,
        )

    # MLP / bagging
    if name == "MLP_default":
        return MLPClassifier(random_state=42, max_iter=500)
    if name.startswith("MLP_"):
        if name == "MLP_100_50":
            return MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
            )
        if name == "MLP_200_100":
            return MLPClassifier(
                hidden_layer_sizes=(200, 100), random_state=42, max_iter=500
            )
        if name.endswith("_tanh"):
            return MLPClassifier(
                hidden_layer_sizes=(100,),
                activation="tanh",
                random_state=42,
                max_iter=500,
            )
        if name.endswith("_relu"):
            return MLPClassifier(
                hidden_layer_sizes=(100,),
                activation="relu",
                random_state=42,
                max_iter=500,
            )
        if name.endswith("_l2"):
            return MLPClassifier(
                hidden_layer_sizes=(100,), alpha=0.01, random_state=42, max_iter=500
            )
        if name.endswith("_l2_hi"):
            return MLPClassifier(
                hidden_layer_sizes=(100,), alpha=0.1, random_state=42, max_iter=500
            )
        size = int(name.split("_")[1])
        return MLPClassifier(hidden_layer_sizes=(size,), random_state=42, max_iter=500)
    if name == "Bag_BNB":
        return _bagging_classifier(
            BernoulliNB(alpha=0.1),
            n_estimators=100,
            max_features=0.7,
            random_state=42,
            n_jobs=1,
        )
    if name.startswith("Bag_LR_"):
        n = int(name.split("_")[-1])
        return _bagging_classifier(
            LogisticRegression(C=0.1, max_iter=1000, random_state=42),
            n_estimators=n,
            max_features=0.7,
            random_state=42,
            n_jobs=1,
        )

    # Factor/decomposition + LR
    if name.startswith("FA_") or name.startswith("FA_LR_"):
        n = int(name.rsplit("_", 1)[-1])
        return FAClfCls(n_components=n, random_state=42)
    if name.startswith("TruncSVD_n"):
        return DecompLR(
            kind="svd", n_components=int(name.split("n", 1)[1]), random_state=42
        )
    if name.startswith("FastICA_n"):
        import re

        m = re.match(r"FastICA_n(\d+)(?:_(cube|exp|deflation))?$", name)
        if m:
            kwargs = {}
            if m.group(2) in {"cube", "exp"}:
                kwargs["fun"] = m.group(2)
            if m.group(2) == "deflation":
                kwargs["algorithm"] = "deflation"
            return DecompLR(
                kind="ica", n_components=int(m.group(1)), random_state=42, **kwargs
            )

    # Generative mixture classifiers
    if name.startswith("GMM_") or name.startswith("BayesGMM_"):
        parts = name.split("_")
        bayes = name.startswith("BayesGMM")
        n_components = int(parts[1])
        cov = parts[2] if len(parts) > 2 else "full"
        if cov == "sph":
            cov = "spherical"
        return GMMClf(
            n_components=n_components,
            covariance_type=cov,
            bayesian=bayes,
            random_state=42,
        )

    # External gradient boosting packages, when installed.
    if name.startswith("LGB"):
        try:
            import lightgbm as lgb
        except Exception as exc:
            raise ImportError(f"Model {name} requires lightgbm") from exc
        if name == "LGB_tuned":
            return lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                n_jobs=1,
                random_state=42,
                verbose=-1,
            )
        if name == "LGB_d3":
            return lgb.LGBMClassifier(
                max_depth=3,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=7,
                min_child_samples=10,
                n_jobs=1,
                random_state=42,
                verbose=-1,
            )
        if name == "LGB_d5":
            return lgb.LGBMClassifier(
                max_depth=5,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                n_jobs=1,
                random_state=42,
                verbose=-1,
            )
        if name == "LGB_l1_reg":
            return lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_jobs=1,
                random_state=42,
                verbose=-1,
            )
        return lgb.LGBMClassifier(n_jobs=1, random_state=42, verbose=-1)
    if name.startswith("XGB"):
        try:
            import xgboost as xgb
        except Exception as exc:
            raise ImportError(f"Model {name} requires xgboost") from exc
        if name == "XGB_tuned":
            return xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                nthread=1,
            )
        if name == "XGB_d3":
            return xgb.XGBClassifier(
                max_depth=3,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                nthread=1,
            )
        if name == "XGB_d4":
            return xgb.XGBClassifier(
                max_depth=4,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                nthread=1,
            )
        return xgb.XGBClassifier(
            eval_metric="logloss", random_state=42, verbosity=0, nthread=1
        )
    if name.startswith("CB_"):
        try:
            import catboost as cb
        except Exception as exc:
            raise ImportError(f"Model {name} requires catboost") from exc
        params = dict(verbose=False, random_seed=42, thread_count=1)
        if name == "CB_depth4":
            params["depth"] = 4
        if name == "CB_depth6":
            params["depth"] = 6
        if name == "CB_l2_5":
            params["l2_leaf_reg"] = 5
        return cb.CatBoostClassifier(**params)

    # Sparse-positive NB variants from some manuscript sweeps: approximate with BernoulliNB.
    if name.startswith("SPNB_"):
        import re

        m = re.search(r"_a(\d+)", name)
        alpha = (float(m.group(1)) / 10.0) if m else 1.0
        return BernoulliNB(alpha=alpha)

    raise KeyError(
        f"No model factory for selected member model {model_name!r}. "
        "This should only happen if the completed configs registry contains a model name whose definition is absent from the supplied manuscript sweep script and from the manuscript registry in this exporter."
    )


def _make_model(model_name: str, module=None):
    from_module = _model_from_sweep_module(module, model_name)
    if from_module is not None:
        return from_module
    return _manuscript_model_factory(model_name, module=module)


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


class FittedTransform:
    def __init__(self, name: str, max_pairs: int = 500, random_state: int = 42):
        self.name = str(name)
        self.max_pairs = int(max_pairs)
        self.random_state = int(random_state)
        self.params: dict[str, Any] = {}

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        name = self.name
        self.n_features_in_ = X.shape[1]
        if name == "clr_std":
            Z = self._stateless("clr", X)
            mu = Z.mean(axis=0, keepdims=True)
            sd = Z.std(axis=0, keepdims=True)
            sd[sd < 1e-10] = 1.0
            self.params = {"mu": mu, "sd": sd}
        elif name == "ilr_std":
            Z = self._stateless("ilr", X)
            mu = Z.mean(axis=0, keepdims=True)
            sd = Z.std(axis=0, keepdims=True)
            sd[sd < 1e-10] = 1.0
            self.params = {"mu": mu, "sd": sd}
        elif name == "rank_col":
            self.params = {
                "sorted_cols": [
                    np.sort(X[:, j].astype(np.float64)) for j in range(X.shape[1])
                ],
                "n_train": X.shape[0],
            }
        elif name in {"prev_weighted", "prev_weigthed"}:
            self.params = {"prev": (X > 0).mean(axis=0).astype(np.float32)}
        elif name == "power":
            est = PowerTransformer(method="yeo-johnson", standardize=True).fit(X)
            self.params = {"estimator": est}
        elif name == "robust":
            est = RobustScaler().fit(X)
            self.params = {"estimator": est}
        elif name == "quantile":
            est = QuantileTransformer(
                output_distribution="normal",
                random_state=42,
                n_quantiles=max(1, min(100, X.shape[0])),
            ).fit(X)
            self.params = {"estimator": est}
        elif name == "pairwise_logratio":
            D = X.shape[1]
            all_pairs = np.array(np.triu_indices(D, k=1)).T
            if len(all_pairs) <= self.max_pairs:
                chosen = all_pairs
            else:
                rng = np.random.RandomState(self.random_state)
                chosen = all_pairs[
                    rng.choice(len(all_pairs), self.max_pairs, replace=False)
                ]
            self.params = {
                "i_idx": chosen[:, 0].astype(np.int32),
                "j_idx": chosen[:, 1].astype(np.int32),
            }
        else:
            # Validate stateless names now.
            self._stateless(name, X[: min(2, len(X))])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        name = self.name
        if (
            getattr(self, "n_features_in_", None) is not None
            and X.shape[1] != self.n_features_in_
        ):
            raise ValueError(
                f"Transform {name!r} expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        if name == "clr_std":
            Z = self._stateless("clr", X)
            return ((Z - self.params["mu"]) / self.params["sd"]).astype(np.float32)
        if name == "ilr_std":
            Z = self._stateless("ilr", X)
            return ((Z - self.params["mu"]) / self.params["sd"]).astype(np.float32)
        if name == "rank_col":
            n_train = float(self.params["n_train"])
            out = np.zeros_like(X, dtype=np.float32)
            for j, col_sorted in enumerate(self.params["sorted_cols"]):
                out[:, j] = np.searchsorted(col_sorted, X[:, j], side="right") / max(
                    n_train, 1.0
                )
            return out
        if name in {"prev_weighted", "prev_weigthed"}:
            prev = self.params["prev"]
            Z = X * prev
            s = Z.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return (Z / s).astype(np.float32)
        if name in {"power", "robust", "quantile"}:
            return self.params["estimator"].transform(X).astype(np.float32)
        if name == "pairwise_logratio":
            Z = _mr(X)
            return np.log(
                Z[:, self.params["i_idx"]] / Z[:, self.params["j_idx"]]
            ).astype(np.float32)
        return self._stateless(name, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _stateless(self, name: str, X):
        X = np.asarray(X, dtype=np.float32)
        if name == "none":
            return X.astype(np.float32)
        if name == "binary":
            return (X > 0).astype(np.float32)
        if name == "sqrt":
            return np.sqrt(np.clip(X, 0, None)).astype(np.float32)
        if name == "hellinger":
            return np.sqrt(_renormalize(X)).astype(np.float32)
        if name == "arcsin_sqrt":
            return np.arcsin(np.sqrt(np.clip(X, 0, 1))).astype(np.float32)
        if name == "log":
            return np.log1p(np.clip(X, 0, None)).astype(np.float32)
        if name == "log10":
            return np.log10(np.clip(X, 0, None) + 1).astype(np.float32)
        if name == "log2":
            return np.log2(np.clip(X, 0, None) + 1).astype(np.float32)
        if name == "log_tss_floor":
            return np.log(np.maximum(_renormalize(X), 1e-10)).astype(np.float32)
        if name == "zi_log":
            return np.where(X > 0, np.log(np.clip(X, 1e-300, None)), 0.0).astype(
                np.float32
            )
        if name == "symlog":
            return (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
        if name == "zscore":
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True)
            sd[sd < 1e-10] = 1.0
            return ((X - mu) / sd).astype(np.float32)
        if name == "log_std":
            Z = np.log1p(np.clip(X, 0, None))
            mu = Z.mean(axis=1, keepdims=True)
            sd = Z.std(axis=1, keepdims=True)
            sd[sd < 1e-10] = 1.0
            return ((Z - mu) / sd).astype(np.float32)
        if name == "log_unit":
            Z = np.log1p(np.clip(X, 0, None))
            n = np.sqrt((Z**2).sum(axis=1, keepdims=True))
            n[n < 1e-10] = 1.0
            return (Z / n).astype(np.float32)
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
        if name == "chi_square":
            rel = _renormalize(X)
            w = np.sqrt(rel + 1e-10)
            return (rel / w).astype(np.float32)
        if name in {"clr", "scikit-bio_clr"}:
            L = np.log(_mr(X))
            return (L - L.mean(axis=1, keepdims=True)).astype(np.float32)
        if name in {"alr", "scikit-bio_alr"}:
            Z = _mr(X)
            return np.log(Z[:, :-1] / Z[:, -1:]).astype(np.float32)
        if name in {"ilr", "scikit-bio_ilr"}:
            Z = _mr(X)
            D = Z.shape[1]
            if D < 2:
                return Z.astype(np.float32)
            return (np.log(Z) @ _ilr_basis(D).T).astype(np.float32)
        if name == "rclr":
            L = np.where(X > 0, np.log(np.clip(X, 1e-300, None)), 0.0)
            nz = (X > 0).astype(np.float64)
            denom = nz.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            gm = (L * nz).sum(axis=1, keepdims=True) / denom
            return np.where(X > 0, L - gm, 0.0).astype(np.float32)
        if name == "bclr":
            D = X.shape[1]
            Z = X + 0.5 / max(D, 1)
            Z = Z / np.maximum(Z.sum(axis=1, keepdims=True), 1e-300)
            L = np.log(Z)
            return (L - L.mean(axis=1, keepdims=True)).astype(np.float32)
        raise ValueError(f"Unsupported transform {name!r}")


def _proba_pos(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        P = estimator.predict_proba(X)
        if np.ndim(P) == 1:
            return np.asarray(P, dtype=np.float32)
        classes = list(getattr(estimator, "classes_", [0, 1]))
        col = (
            classes.index(POSITIVE_CLASS)
            if POSITIVE_CLASS in classes
            else min(1, P.shape[1] - 1)
        )
        return np.asarray(P[:, col], dtype=np.float32)
    if hasattr(estimator, "decision_function"):
        D = estimator.decision_function(X)
        if np.ndim(D) > 1:
            D = D[:, min(1, D.shape[1] - 1)]
        return expit(D).astype(np.float32)
    pred = estimator.predict(X)
    return np.asarray(pred == POSITIVE_CLASS, dtype=np.float32)


class SuperLearnerLR:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=0.1, max_iter=500, random_state=42, solver="lbfgs"
        )

    def fit(self, X, y):
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def predict_proba_pos(self, X):
        P = self.clf.predict_proba(self.scaler.transform(X))
        classes = list(self.clf.classes_)
        return P[
            :,
            classes.index(POSITIVE_CLASS)
            if POSITIVE_CLASS in classes
            else min(1, P.shape[1] - 1),
        ].astype(np.float32)


class SuperLearnerRidge:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mdl = Ridge(alpha=1.0)

    def fit(self, X, y):
        self.mdl.fit(self.scaler.fit_transform(X), np.asarray(y, dtype=float))
        return self

    def predict_proba_pos(self, X):
        return expit(self.mdl.predict(self.scaler.transform(X))).astype(np.float32)


class SuperLearnerRF:
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, n_jobs=1, random_state=42
        )

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba_pos(self, X):
        P = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        return P[
            :,
            classes.index(POSITIVE_CLASS)
            if POSITIVE_CLASS in classes
            else min(1, P.shape[1] - 1),
        ].astype(np.float32)


def _make_superlearner(method: str):
    if method == "superlearner__lr":
        return SuperLearnerLR()
    if method == "superlearner__ridge":
        return SuperLearnerRidge()
    if method == "superlearner__rf":
        return SuperLearnerRF()
    raise ValueError(f"Unknown superlearner aggregation {method!r}")


def _aggregate_predictions(
    probas: np.ndarray, weights: np.ndarray, method: str, meta_model=None
) -> np.ndarray:
    pf = np.where(np.isnan(np.asarray(probas, dtype=np.float32)), 0.5, probas)
    n_members, n_samples = pf.shape
    if method.startswith("superlearner__"):
        if meta_model is None:
            raise ValueError(f"Aggregation {method} requires a fitted meta_model")
        return meta_model.predict_proba_pos(pf.T)
    if method == "mean_proba":
        return pf.mean(axis=0)
    if method == "weighted_mean_proba":
        w = np.clip(weights - weights.min() + 1e-8, 0, None)
        return (pf * w[:, None]).sum(axis=0) / (w.sum() or 1.0)
    if method == "median_proba":
        return np.median(pf, axis=0)
    if method == "trimmed_mean":
        trim = max(1, int(0.10 * n_members))
        core = np.sort(pf, axis=0)[trim : n_members - trim]
        return core.mean(axis=0) if len(core) else pf.mean(axis=0)
    if method == "geometric_mean":
        lp = np.log(np.clip(pf, 1e-10, 1 - 1e-10)).mean(axis=0)
        ln = np.log(np.clip(1 - pf, 1e-10, 1 - 1e-10)).mean(axis=0)
        ep, en = np.exp(lp), np.exp(ln)
        return ep / (ep + en + 1e-30)
    if method == "log_odds_mean":
        lo = np.log(np.clip(pf, 1e-6, 1 - 1e-6) / np.clip(1 - pf, 1e-6, 1 - 1e-6))
        return 1 / (1 + np.exp(-lo.mean(axis=0)))
    if method == "harmonic_mean":
        return np.clip(
            n_members / (1 / np.clip(pf, 1e-10, 1 - 1e-10)).sum(axis=0), 0, 1
        )
    if method == "minmax":
        return (pf.min(axis=0) + pf.max(axis=0)) / 2
    if method == "rank_mean":
        return (np.apply_along_axis(rankdata, 1, pf) / max(n_samples, 1)).mean(axis=0)
    if method == "borda_count":
        b = np.apply_along_axis(lambda r: rankdata(r, method="average"), 1, pf).sum(
            axis=0
        )
        rng = b.max() - b.min()
        return (b - b.min()) / rng if rng > 1e-10 else np.full(n_samples, 0.5)
    if method == "majority_vote":
        return (pf >= 0.5).mean(axis=0)
    if method == "weighted_vote":
        w = np.clip(weights - weights.min() + 1e-8, 0, None)
        return ((pf >= 0.5) * w[:, None]).sum(axis=0) / (w.sum() or 1.0)
    if method == "confidence_weighted":
        conf = np.maximum(pf, 1 - pf)
        bw = np.exp(weights - weights.max())
        bw /= bw.sum() or 1.0
        wm = conf * bw[:, None]
        ws = wm.sum(axis=0)
        return (pf * wm).sum(axis=0) / np.where(ws > 0, ws, 1.0)
    if method == "softmax_mean":
        z = np.log(np.clip(pf, 1e-10, 1 - 1e-10)) / 0.5
        z -= z.max(axis=0, keepdims=True)
        sm = np.exp(z)
        sm /= sm.sum(axis=0, keepdims=True)
        return (pf * sm).sum(axis=0)
    if method == "bayesian_avg":
        w = np.clip(weights, 0, None)
        norm = (w - w.min()) / (w.max() - w.min() + 1e-10)
        post = np.exp(n_members * norm)
        post /= post.sum() or 1.0
        return (pf * post[:, None]).sum(axis=0)
    if method == "power_mean_p3":
        return np.clip((np.clip(pf, 1e-10, 1) ** 3).mean(axis=0) ** (1 / 3), 0, 1)
    if method == "power_mean_p05":
        return np.clip((np.clip(pf, 1e-10, 1) ** 0.5).mean(axis=0) ** 2, 0, 1)
    if method == "max_proba":
        return pf.max(axis=0)
    if method == "min_proba":
        return pf.min(axis=0)
    if method == "dempster_shafer":
        p = np.clip(pf, 1e-6, 1 - 1e-6)
        pp = p.prod(axis=0)
        pn = (1 - p).prod(axis=0)
        return pp / np.where(pp + pn < 1e-10, 1e-10, pp + pn)
    raise ValueError(f"Unsupported aggregation strategy {method!r}")


def _collect_parquet_shards(directory: Path, patterns: list[str]) -> list[Path]:
    """Return unique parquet shards matching any of the historical sweep layouts."""
    seen: dict[str, Path] = {}
    for pat in patterns:
        for path in directory.glob(pat):
            seen[str(path.resolve())] = path
    return sorted(seen.values())


def _read_inner_scores(
    experiment_dir: Path, member_ids: list[str], optimize_metric: str
) -> dict[str, float]:
    if pl is None:
        return {cid: 1.0 for cid in member_ids}
    files = _collect_parquet_shards(
        experiment_dir / "inner_results",
        [
            "outer_r*_o*_inner.parquet",  # SCZ nested-CV sweep layout
            "outer_*__inner_*.parquet",  # LODO / legacy layout
            "*.parquet",  # last-resort fallback
        ],
    )
    if not files:
        return {cid: 1.0 for cid in member_ids}
    df = pl.scan_parquet([str(p) for p in files]).collect()
    if "error" in df.columns:
        df = df.filter(pl.col("error").is_null())
    if optimize_metric not in df.columns or "config_id" not in df.columns:
        return {cid: 1.0 for cid in member_ids}
    out = (
        df.with_columns(pl.col("config_id").cast(pl.Utf8))
        .filter(pl.col("config_id").is_in(member_ids))
        .group_by("config_id")
        .agg(
            pl.col(optimize_metric).cast(pl.Float64, strict=False).mean().alias("score")
        )
        .to_pandas()
    )
    d = dict(zip(out["config_id"].astype(str), out["score"].astype(float)))
    return {cid: float(d.get(cid, 1.0)) for cid in member_ids}


def _fit_superlearner_from_inner_oof(
    experiment_dir: Path, member_ids: list[str], method: str
):
    if pl is None:
        raise RuntimeError(
            "polars is required to read inner_predictions for superlearner export"
        )
    files = _collect_parquet_shards(
        experiment_dir / "inner_predictions",
        [
            "outer_r*_o*_inner.parquet",  # SCZ nested-CV sweep layout
            "outer_*__inner_*.parquet",  # LODO / legacy layout
            "*.parquet",  # last-resort fallback
        ],
    )
    if not files:
        raise FileNotFoundError(
            f"Selected aggregation is {method}, but no inner prediction shards were found in {experiment_dir / 'inner_predictions'}"
        )
    df = pl.scan_parquet([str(p) for p in files]).collect()
    required = {"config_id", "sample_id", "y_true", "y_proba_pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"inner_predictions missing required columns: {sorted(missing)}"
        )
    if "split_kind" in df.columns:
        df = df.filter(pl.col("split_kind") == "inner")
    df = df.with_columns(pl.col("config_id").cast(pl.Utf8)).filter(
        pl.col("config_id").is_in(member_ids)
    )
    if df.height == 0:
        raise ValueError(
            "inner_predictions contain no rows for the selected superlearner members"
        )

    # Match SWEEP_ENSEMBLE__all_configs_SCZ.py exactly: sample_id is reused
    # across inner folds, so the meta-learner row key must be split_key+sample_id.
    if "split_key" in df.columns:
        df = df.with_columns(
            (
                pl.col("split_key").cast(pl.Utf8)
                + "||"
                + pl.col("sample_id").cast(pl.Utf8)
            ).alias("__row_key")
        )
        idx_cols = ["__row_key"]
    else:
        idx_cols = [
            c
            for c in ["outer_study_id", "inner_study_id", "sample_id"]
            if c in df.columns
        ]
        if "sample_id" not in idx_cols:
            idx_cols.append("sample_id")

    piv = df.pivot(
        on="config_id",
        index=idx_cols,
        values="y_proba_pos",
        aggregate_function="mean",
        maintain_order=True,
    ).fill_null(0.5)
    for cid in member_ids:
        if cid not in piv.columns:
            raise ValueError(f"inner_predictions do not contain selected member {cid}")
    ytrue = df.unique(idx_cols).select(idx_cols + ["y_true"])
    y_map = {
        tuple(r[c] for c in idx_cols): r["y_true"] for r in ytrue.iter_rows(named=True)
    }
    keys = [
        tuple(r[c] for c in idx_cols)
        for r in piv.select(idx_cols).iter_rows(named=True)
    ]
    X_meta = piv.select(member_ids).to_numpy().astype(np.float32)
    y = np.array([y_map[k] for k in keys], dtype=np.int8)
    return _make_superlearner(method).fit(X_meta, y)


@dataclass
class MemberModel:
    config_id: str
    transform_name: str
    resolution: str
    model_name: str
    levels: list[str]
    schema: dict[str, Any]
    transform: FittedTransform
    estimator: Any


class IBSMPMAEDeploymentModel:
    def __init__(
        self,
        members: list[MemberModel],
        aggregation_strategy: str,
        weights: np.ndarray,
        meta_model=None,
        manifest: dict[str, Any] | None = None,
    ):
        self.members = members
        self.aggregation_strategy = aggregation_strategy
        self.weights = np.asarray(weights, dtype=np.float32)
        self.meta_model = meta_model
        self.manifest = manifest or {}

    def _profile_matrix_from_dataframe(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, list[str], list[str]]:
        # Supports MetaPhlAn-style: rows=features, columns=samples, first column is feature index,
        # and wide style: rows=samples, columns=features.
        if df.empty:
            raise ValueError("Input profile table is empty")
        first_col = str(df.columns[0])
        looks_like_feature_first = (
            df.iloc[:, 0]
            .astype(str)
            .str.contains(
                r"(?:^d__|^k__|\|p__|\|c__|\|o__|\|f__|\|g__|\|s__|\|t__|___p__|___c__|___o__|___f__|___g__|___s__|___t__)",
                regex=True,
            )
            .mean()
            > 0.5
        )
        if looks_like_feature_first:
            tmp = df.copy()
            tmp.iloc[:, 0] = tmp.iloc[:, 0].astype(str)
            tmp = tmp.set_index(first_col)
            X = tmp.apply(pd.to_numeric, errors="coerce").fillna(0.0).T
            return (
                X.to_numpy(dtype=np.float32),
                list(X.index.astype(str)),
                list(X.columns.astype(str)),
            )
        # Wide sample x feature table. Ignore nonnumeric metadata columns.
        numeric = df.apply(pd.to_numeric, errors="coerce")
        feature_cols = [c for c in numeric.columns if numeric[c].notna().any()]
        Xdf = numeric[feature_cols].fillna(0.0)
        sample_ids = list(df.index.astype(str))
        return Xdf.to_numpy(dtype=np.float32), sample_ids, list(map(str, feature_cols))

    def _build_X_for_member(
        self, X_raw: np.ndarray, raw_names: list[str], member: MemberModel
    ) -> np.ndarray:
        blocks = []
        for level in member.levels:
            depth = LEVEL_DEPTH[level]
            groups: dict[str, list[int]] = {}
            for j, raw in enumerate(raw_names):
                key = _truncate_lineage(raw, depth)
                if _feature_name_ok(key, level):
                    groups.setdefault(key, []).append(j)
            union = list(member.schema["level_features"][level])
            out = np.zeros((X_raw.shape[0], len(union)), dtype=np.float32)
            for j, f in enumerate(union):
                if f in groups:
                    out[:, j] = X_raw[:, groups[f]].sum(axis=1)
            blocks.append(_renormalize(out))
        return np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]

    def predict_proba_from_profile_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        X_raw, sample_ids, raw_names = self._profile_matrix_from_dataframe(df)
        member_probas = []
        for m in self.members:
            X = self._build_X_for_member(X_raw, raw_names, m)
            Xt = m.transform.transform(X)
            member_probas.append(_proba_pos(m.estimator, Xt))
        Pm = np.vstack(member_probas)
        p = _aggregate_predictions(
            Pm, self.weights, self.aggregation_strategy, self.meta_model
        )
        return pd.DataFrame(
            {
                "sample_id": sample_ids,
                "proba_positive": p,
                "prediction": (p >= 0.5).astype(int),
            }
        )

    def predict_profile_tsv(self, path: str | Path) -> pd.DataFrame:
        return self.predict_proba_from_profile_dataframe(pd.read_csv(path, sep="\t"))


# Stable module names for joblib pickle references inside the deployment artifact.
for _cls in [
    RidgeProba,
    NearestCentroidProba,
    FLAMLClassifier,
    SGDProba,
    PAProba,
    PerceptronProba,
    RidgeCVProba,
    RadNCProba,
    FAClf,
    DecompLR,
    GMMClf,
    FittedTransform,
    SuperLearnerLR,
    SuperLearnerRidge,
    SuperLearnerRF,
    MemberModel,
    IBSMPMAEDeploymentModel,
]:
    try:
        _cls.__module__ = RUNTIME_MODULE
    except Exception:
        pass


def _write_zip(
    out_path: Path,
    model: IBSMPMAEDeploymentModel,
    manifest: dict[str, Any],
    selected_configs: pd.DataFrame,
    selected_unit: dict[str, Any],
    sweep_script: Path | None = None,
):
    """Write the clean app-facing model package.

    The output zip intentionally does NOT include this exporter script or the
    manuscript sweep script. mllabiome-ii provides the importable
    `ibs_mpmae_runtime` module needed by joblib to load the deployment object.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_buf = io.BytesIO()
    joblib.dump(model, model_buf, compress=3)
    model_bytes = model_buf.getvalue()
    configs_buf = io.StringIO()
    selected_configs.to_csv(configs_buf, sep="\t", index=False)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.joblib", model_bytes)
        z.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
        z.writestr(
            "selected_unit.json", json.dumps(selected_unit, indent=2, default=str)
        )
        z.writestr("selected_member_configs.tsv", configs_buf.getvalue())
        z.writestr(
            "README.txt",
            "Clean mllabiome-ii deployment package. Copy this zip into "
            "app/backend/production_models/. The exporter script and manuscript "
            "sweep script are not part of the hosted model package. The app must "
            "provide ibs_mpmae_runtime before loading model.joblib.\n",
        )


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("FINAL/experiments/DM-LAMPP"),
        help="Existing DM LAMPP experiment directory.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("data/lampp-newest/data/dmw/dmw_train.csv"),
        help="Full labelled DM wide CSV used by the benchmark sweep.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output deployment zip. Default: <model-id>_model_<artifact-version>.zip",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting --out if it already exists.",
    )
    p.add_argument(
        "--artifact-version",
        default="v0.1.0",
        help="Model package version embedded in the manifest and default filename.",
    )
    p.add_argument("--model-id", default="dm_mpma_e", help="Stable backend model id.")
    p.add_argument(
        "--display-name",
        default="DM MPMA-E",
        help="Human-readable model name shown in the UI.",
    )
    p.add_argument("--target", default="DM", help="Short target label for model lists.")
    p.add_argument(
        "--positive-value",
        default="1",
        help="Raw CSV value treated as the positive class.",
    )
    p.add_argument(
        "--positive-label",
        default="Class 1",
        help="Display label for positive predictions.",
    )
    p.add_argument(
        "--negative-label",
        default="Class 0",
        help="Display label for negative predictions.",
    )
    p.add_argument(
        "--sweep-script",
        type=Path,
        default=None,
        help="Optional original sweep script; used to reuse _build_models when possible.",
    )
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--target-col", default="label")
    p.add_argument(
        "--metadata-cols",
        default="label,sample_id,subject_id,study_id",
        help="Comma-separated non-feature CSV columns.",
    )
    p.add_argument(
        "--study-id-col",
        default="study_id",
        help="CSV column containing LODO study/cohort ids.",
    )
    p.add_argument(
        "--study-ids",
        default="0,9,12",
        help="Comma-separated DM study IDs used in the sweep. Empty string uses all rows in the CSV.",
    )
    p.add_argument("--optimize-metric", default=DEFAULT_OPTIMIZE_METRIC)
    args = p.parse_args()

    experiment_dir = args.experiment_dir.expanduser().resolve()
    csv_path = args.csv.expanduser().resolve()
    module = _load_module(args.sweep_script)

    selected_unit = _read_selected_unit(experiment_dir)
    ens = selected_unit["inner_val_best_ensemble"]
    member_ids = [str(x) for x in ens["members"]]
    aggregation_strategy = str(ens["aggregation_strategy"])

    configs = _read_configs(experiment_dir)
    selected_configs = configs[configs["config_id"].astype(str).isin(member_ids)].copy()
    missing = sorted(set(member_ids) - set(selected_configs["config_id"].astype(str)))
    if missing:
        raise ValueError(
            f"Selected member ids missing from configs registry: {missing}"
        )
    selected_configs["__order"] = (
        selected_configs["config_id"]
        .astype(str)
        .map({cid: i for i, cid in enumerate(member_ids)})
    )
    selected_configs = selected_configs.sort_values("__order").drop(columns="__order")

    all_levels = sorted(
        {lv for _, row in selected_configs.iterrows() for lv in _parse_levels(row)},
        key=lambda x: SINGLE_LEVELS.index(x),
    )
    print(
        f"Selected MPMA-E: {len(member_ids)} members, aggregation={aggregation_strategy}"
    )
    print(f"CSV: {csv_path}")
    print(f"Levels needed: {', '.join(all_levels)}")

    metadata_cols = {x.strip() for x in str(args.metadata_cols).split(",") if x.strip()}
    study_ids = _parse_study_ids_arg(args.study_ids)
    if study_ids:
        print(f"Study IDs: {study_ids}")
    arrays = _build_lampp_level_arrays(
        csv_path,
        all_levels,
        args.sample_id_col,
        args.target_col,
        args.positive_value,
        metadata_cols,
        study_id_col=args.study_id_col,
        study_ids=study_ids,
    )
    weights_by_member = _read_inner_scores(
        experiment_dir, member_ids, args.optimize_metric
    )
    weights = np.array(
        [weights_by_member.get(cid, 1.0) for cid in member_ids], dtype=np.float32
    )

    trained_members: list[MemberModel] = []
    for _, row in selected_configs.iterrows():
        cid = str(row["config_id"])
        transform_name = str(row["transform"])
        model_name = str(row["model"])
        resolution = str(row["resolution"])
        levels = _parse_levels(row)
        print(
            f"Training member {cid}: {resolution} / {transform_name} / {model_name}",
            flush=True,
        )
        X, y, sample_ids, schema = _build_lampp_member_training_matrix(arrays, levels)
        ft = FittedTransform(transform_name)
        Xt = ft.fit_transform(X)
        estimator = _make_model(model_name, module=module)
        estimator.fit(Xt, y)
        trained_members.append(
            MemberModel(
                config_id=cid,
                transform_name=transform_name,
                resolution=resolution,
                model_name=model_name,
                levels=list(levels),
                schema=schema,
                transform=ft,
                estimator=estimator,
            )
        )

    meta_model = None
    if aggregation_strategy.startswith("superlearner__"):
        print(
            f"Fitting deployment meta-learner from stored inner OOF predictions: {aggregation_strategy}",
            flush=True,
        )
        meta_model = _fit_superlearner_from_inner_oof(
            experiment_dir, member_ids, aggregation_strategy
        )

    manifest = {
        "artifact_schema": "mllabiome.mpmae_object.v1",
        "artifact_version": str(args.artifact_version),
        "model_id": str(args.model_id),
        "display_name": str(args.display_name),
        "description": f"{args.display_name} packaged microbiome inference model.",
        "target": str(args.target),
        "model_type": "MPMA-E",
        "source": "LAMPP benchmark experiment outputs",
        "experiment_dir": str(experiment_dir),
        "data_csv": str(csv_path),
        "trained_on_full_labelled_dataset": True,
        "performance_source": "existing benchmark sweep outputs, not this full-data refit",
        "selected_members": member_ids,
        "aggregation_strategy": aggregation_strategy,
        "member_weights": {
            cid: float(weights_by_member.get(cid, 1.0)) for cid in member_ids
        },
        "n_members": len(trained_members),
        "sample_id_col": args.sample_id_col,
        "target_col": args.target_col,
        "study_id_col": args.study_id_col,
        "study_ids": [str(x) for x in study_ids],
        "positive_value": str(args.positive_value),
        "positive_label": str(args.positive_label),
        "negative_label": str(args.negative_label),
        "input_format": "Wide sample-by-feature CSV/TSV with MetaPhlAn-style lineage columns, or MetaPhlAn-style profile table.",
    }
    model = IBSMPMAEDeploymentModel(
        trained_members,
        aggregation_strategy,
        weights,
        meta_model=meta_model,
        manifest=manifest,
    )
    out_path = _versioned_model_out_path(
        args.out, args.model_id, args.artifact_version, overwrite=args.overwrite
    )
    _write_zip(
        out_path,
        model,
        manifest,
        selected_configs,
        selected_unit,
        sweep_script=args.sweep_script,
    )
    print(f"\nWrote deployment package: {out_path}")
    print(
        "Copy this versioned zip into app/backend/production_models/ and restart the backend."
    )


if __name__ == "__main__":
    main()
