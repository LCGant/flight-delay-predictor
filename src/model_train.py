from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import warnings

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_recall_curve,
    confusion_matrix, classification_report
)
import joblib

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

SPLITS_DIR = Path("data/processed/splits")
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR  = Path("data/models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "atraso15"
CATEGORICAL_BASE = ["icao_empresa", "origem_icao", "destino_icao", "rota"]
NUMERICAL_BASE = [
    "mes","dia_semana","hora_bloco_30","hora_sin","hora_cos","periodo_dia_id",
    "is_weekend","is_feriado","is_vespera_feriado","em_ferias",
    "w_dep_precip_mm","w_dep_wind_ms","w_dep_gust_ms","w_dep_temp_c","w_dep_rh",
    "wx_chuva","wx_vento_forte","wx_rajada_forte","wx_calor","wx_frio","wx_umidade_alta",
    "w_dep_precip_mm_isna","w_dep_wind_ms_isna","w_dep_gust_ms_isna","w_dep_temp_c_isna","w_dep_rh_isna",
    "wx_missing_any",
    "visibilidade_ruim",
    "w_arr_precip_mm","w_arr_wind_ms","w_arr_gust_ms","w_arr_temp_c","w_arr_rh",
    "w_arr_precip_mm_isna","w_arr_wind_ms_isna","w_arr_gust_ms_isna","w_arr_temp_c_isna","w_arr_rh_isna",
    "wx_arr_chuva","wx_arr_vento_forte","wx_arr_rajada_forte","wx_arr_calor","wx_arr_frio","wx_arr_umidade_alta",
    "load_origem_15","dist_km","atrasos_mesmo_aeroporto_hora",
]

def load_split(name: str) -> pd.DataFrame:
    p = SPLITS_DIR / f"{name}.parquet"
    if not p.exists():
        raise SystemExit(f"Split '{name}' não encontrado em {p}")
    df = pd.read_parquet(p)
    if TARGET not in df.columns:
        raise SystemExit(f"Coluna alvo '{TARGET}' ausente em {p}")
    df[TARGET] = df[TARGET].astype(int)
    for c in CATEGORICAL_BASE:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    for c in NUMERICAL_BASE:
        if c in df.columns and df[c].dtype.name == "Int8":
            df[c] = df[c].astype("float")
    return df

def infer_feature_columns(df: pd.DataFrame):
    cat_cols = [c for c in CATEGORICAL_BASE if c in df.columns]
    num_cols = [c for c in NUMERICAL_BASE if c in df.columns]
    dyn_cols = [
        c for c in df.columns
        if c.startswith(("hist_", "lag_")) or c.endswith("_hist")
    ]
    for c in dyn_cols:
        if df[c].dtype.name in ("Int8","Int16","Int32","Int64","UInt8","UInt16","UInt32","UInt64","object"):
            with np.errstate(all="ignore"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
    seen = set()
    num_cols = [c for c in (num_cols + dyn_cols) if not (c in seen or seen.add(c))]
    return cat_cols, num_cols


def build_Xy(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]):
    cols = [c for c in cat_cols + num_cols if c in df.columns]
    X = df[cols].copy()
    y = df[TARGET].astype(int).values
    cat_idx = [i for i, c in enumerate(cols) if c in cat_cols]
    return X, y, cols, cat_idx

def fit_category_maps(X: pd.DataFrame, cat_cols: list[str]) -> dict[str, list[str]]:
    maps = {}
    for c in cat_cols:
        if c in X.columns:
            cats = pd.Series(X[c].astype(str).unique()).dropna().astype(str)
            maps[c] = sorted(cats.tolist())
    return maps

def apply_category_maps(X: pd.DataFrame, cat_maps: dict[str, list[str]], as_codes: bool = False) -> pd.DataFrame:
    X2 = X.copy()
    for c, cats in cat_maps.items():
        if c in X2.columns:
            cat_dtype = pd.api.types.CategoricalDtype(categories=cats)
            X2[c] = X2[c].astype("string").astype(cat_dtype)
            if as_codes:
                X2[c] = X2[c].cat.codes.astype("int32")
    return X2

def impute_dist_km_by_route(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame):
    if "dist_km" not in tr.columns:
        return tr, va, te
    def _rota_series(df):
        return df["rota"].astype(str) if "rota" in df.columns else pd.Series(index=df.index, dtype="string")
    tr_rota = _rota_series(tr)
    m_valid = tr["dist_km"].notna() & tr_rota.notna()
    if m_valid.any():
        rota_mean = (
            tr.loc[m_valid]
              .groupby(tr_rota[m_valid])["dist_km"]
              .mean()
        )
        glob_med = float(tr.loc[tr["dist_km"].notna(), "dist_km"].median())
    else:
        rota_mean = pd.Series(dtype="float64")
        glob_med = float("nan")
    def _fill(df):
        df = df.copy()
        if "dist_km" in df.columns:
            if "rota" in df.columns and not rota_mean.empty:
                df["dist_km"] = df["dist_km"].fillna(df["rota"].astype(str).map(rota_mean))
            if np.isnan(glob_med):
                df["dist_km"] = df["dist_km"].fillna(0.0)
            else:
                df["dist_km"] = df["dist_km"].fillna(glob_med)
        return df
    tr = _fill(tr); va = _fill(va); te = _fill(te)
    return tr, va, te

def pick_threshold_f1(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
    i = int(np.nanargmax(f1))
    return float(thr[i]), {"policy": "f1", "F1": float(f1[i]), "precision": float(prec[i]), "recall": float(rec[i])}

def pick_threshold_fbeta(y_true, proba, beta: float = 1.0):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    beta2 = beta * beta
    fbeta = (1 + beta2) * prec[:-1] * rec[:-1] / (beta2 * prec[:-1] + rec[:-1] + 1e-9)
    i = int(np.nanargmax(fbeta))
    return float(thr[i]), {"policy": f"f_beta({beta})", "F_beta": float(fbeta[i]), "precision": float(prec[i]), "recall": float(rec[i])}

def pick_threshold_prec_at(y_true, proba, min_precision: float = 0.5):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    mask = prec[:-1] >= min_precision
    if not np.any(mask):
        return pick_threshold_f1(y_true, proba)
    idx = np.where(mask)[0]
    best = idx[np.argmax(rec[:-1][idx])]
    return float(thr[best]), {"policy": f"prec_at({min_precision})", "precision": float(prec[best]), "recall": float(rec[best])}

def pick_threshold_recall_at(y_true, proba, min_recall: float = 0.7):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    mask = rec[:-1] >= min_recall
    if not np.any(mask):
        return pick_threshold_f1(y_true, proba)
    idx = np.where(mask)[0]
    best = idx[np.argmax(prec[:-1][idx])]
    return float(thr[best]), {"policy": f"recall_at({min_recall})", "precision": float(prec[best]), "recall": float(rec[best])}

def pick_threshold(y_true, proba, policy: str = "f1", beta: float = 1.0, min_precision: float | None = None, min_recall: float | None = None):
    policy = policy.lower()
    if policy == "f1":
        return pick_threshold_f1(y_true, proba)
    if policy == "f_beta":
        return pick_threshold_fbeta(y_true, proba, beta=beta)
    if policy == "prec_at":
        if min_precision is None:
            min_precision = 0.5
        return pick_threshold_prec_at(y_true, proba, min_precision=min_precision)
    if policy == "recall_at":
        if min_recall is None:
            min_recall = 0.7
        return pick_threshold_recall_at(y_true, proba, min_recall=min_recall)
    return pick_threshold_f1(y_true, proba)

def eval_at_threshold(y_true, proba, thr):
    y_pred = (proba >= thr).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, proba)),
        "PR_AUC": float(average_precision_score(y_true, proba)),
        "F1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True)
    }

def choose_groups_for_sampling(df: pd.DataFrame):
    if "rota" in df.columns and "mes" in df.columns:
        return ["rota", "mes"]
    if all(c in df.columns for c in ["origem_icao", "destino_icao", "mes"]):
        return ["origem_icao", "destino_icao", "mes"]
    if "partida_prevista" in df.columns:
        dt = pd.to_datetime(df["partida_prevista"], errors="coerce")
        return [dt.dt.to_period("M").astype(str).rename("ano_mes")]
    if "mes" in df.columns:
        return ["mes"]
    return []

def _ensure_series(x):
    return x if isinstance(x, pd.Series) else pd.Series(x)

def stratified_negative_undersample(
    df: pd.DataFrame,
    target_col: str,
    target_pos: float = 0.35,
    group_cols: list[str] | list[pd.Series] | None = None,
    neg_cap_per_group: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    work = df.copy()
    materialized_groups = []
    temp_cols = []
    if group_cols:
        for gc in group_cols:
            if isinstance(gc, pd.Series):
                name = gc.name or "grp_tmp"
                work[name] = _ensure_series(gc).values
                materialized_groups.append(name)
                temp_cols.append(name)
            else:
                materialized_groups.append(gc)
    pos = work[work[target_col] == 1]
    neg = work[work[target_col] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_neg_target = int(min(len(neg), np.ceil(len(pos) * (1.0 - target_pos) / max(target_pos, 1e-9))))
    if not materialized_groups:
        neg_samp = neg.sample(n=n_neg_target, random_state=seed, replace=False)
    else:
        parts = []
        for _, g in neg.groupby(materialized_groups, dropna=False):
            k = min(neg_cap_per_group, len(g))
            if k > 0:
                parts.append(g.sample(n=k, random_state=seed, replace=False))
        neg_pool = pd.concat(parts, ignore_index=False) if parts else neg
        if len(neg_pool) > n_neg_target:
            neg_samp = neg_pool.sample(n=n_neg_target, random_state=seed, replace=False)
        else:
            need = n_neg_target - len(neg_pool)
            if need > 0:
                remaining = neg.drop(index=neg_pool.index, errors="ignore")
                extra = remaining.sample(n=min(need, len(remaining)), random_state=seed, replace=False)
                neg_samp = pd.concat([neg_pool, extra], ignore_index=False)
            else:
                neg_samp = neg_pool
    out = pd.concat([pos, neg_samp], ignore_index=False).sample(frac=1.0, random_state=seed)
    if temp_cols:
        out = out.drop(columns=[c for c in temp_cols if c in out.columns], errors="ignore")
    return out.reset_index(drop=True)

def balance_train_df(
    train_df: pd.DataFrame,
    mode: str = "none",
    target_col: str = "atraso15",
    target_pos: float = 0.35,
    neg_cap_per_group: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    if mode == "none":
        return train_df
    if mode == "undersample_neg":
        groups = choose_groups_for_sampling(train_df)
        return stratified_negative_undersample(
            train_df, target_col, target_pos, groups, neg_cap_per_group, seed
        )
    if mode == "month_quota":
        if "partida_prevista" in train_df.columns:
            dt = pd.to_datetime(train_df["partida_prevista"], errors="coerce")
            groups = [dt.dt.to_period("M").astype(str).rename("ano_mes")]
        elif "mes" in train_df.columns:
            groups = ["mes"]
        else:
            groups = []
        return stratified_negative_undersample(
            train_df, target_col, target_pos, groups, neg_cap_per_group, seed
        )
    if mode == "route_quota":
        if "rota" in train_df.columns and "mes" in train_df.columns:
            groups = ["rota", "mes"]
        elif "rota" in train_df.columns:
            groups = ["rota"]
        elif all(c in train_df.columns for c in ["origem_icao", "destino_icao"]):
            groups = ["origem_icao", "destino_icao"]
        else:
            groups = []
        return stratified_negative_undersample(
            train_df, target_col, target_pos, groups, neg_cap_per_group, seed
        )
    raise ValueError(f"Modo de balanceamento desconhecido: {mode}")

def resolve_device_args(want_device: str):
    want_device = (want_device or "cpu").lower()
    use_gpu = (want_device == "gpu")
    cat_params = {"task_type": "GPU", "devices": "0"} if use_gpu else {}
    lgbm_params = {}
    if use_gpu:
        lgbm_params = {
            "device_type": "gpu",
            "max_bin": 255,
            "max_cat_threshold": 64,
            "max_cat_to_onehot": 16,
            "min_data_per_group": 200,
        }
    xgb_params = {}
    if use_gpu:
        xgb_params = {
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
        }
    return use_gpu, cat_params, lgbm_params, xgb_params

def train_catboost(Xtr, ytr, Xva, yva, cols, cat_idx, args, cat_device_params):
    pos_rate = float(np.mean(ytr))
    if 0 < pos_rate < 1:
        w_pos = (1.0 - pos_rate) / max(pos_rate, 1e-9)
        class_weights = [1.0, w_pos]
    else:
        class_weights = None
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=args.lr,
        depth=args.depth,
        l2_leaf_reg=args.l2,
        iterations=args.iters,
        od_type="Iter",
        od_wait=args.od_wait,
        random_seed=42,
        verbose=args.verbose,
        class_weights=class_weights,
        **cat_device_params
    )
    try:
        model.fit(
            Pool(Xtr, ytr, cat_features=cat_idx),
            eval_set=Pool(Xva, yva, cat_features=cat_idx),
            use_best_model=True
        )
    except Exception as e:
        if cat_device_params:
            warnings.warn(f"Falha no treino CatBoost GPU ({e}). Fazendo fallback para CPU…")
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                learning_rate=args.lr,
                depth=args.depth,
                l2_leaf_reg=args.l2,
                iterations=args.iters,
                od_type="Iter",
                od_wait=args.od_wait,
                random_seed=42,
                verbose=args.verbose,
                class_weights=class_weights
            )
            model.fit(
                Pool(Xtr, ytr, cat_features=cat_idx),
                eval_set=Pool(Xva, yva, cat_features=cat_idx),
                use_best_model=True
            )
        else:
            raise
    return model

def train_lgbm(Xtr, ytr, Xva, yva, cat_cols, args, lgbm_device_params):
    if LGBMClassifier is None:
        raise SystemExit("LightGBM não está instalado. Rode: pip install lightgbm")
    use_gpu = bool(lgbm_device_params)
    cat_maps = fit_category_maps(Xtr, cat_cols)
    if use_gpu:
        Xtr_c = apply_category_maps(Xtr, cat_maps, as_codes=True)
        Xva_c = apply_category_maps(Xva, cat_maps, as_codes=True)
        cat_feature_arg = None
    else:
        Xtr_c = apply_category_maps(Xtr, cat_maps, as_codes=False)
        Xva_c = apply_category_maps(Xva, cat_maps, as_codes=False)
        cat_feature_arg = cat_cols
    pos = float(np.sum(ytr)); neg = float(len(ytr) - pos)
    spw = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    base_params = dict(
        n_estimators=args.iters,
        learning_rate=args.lr,
        max_depth=-1 if args.depth <= 0 else args.depth,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=args.l2,
        random_state=42,
        class_weight=None,
        scale_pos_weight=spw,
        n_jobs=-1
    )
    base_params.update(lgbm_device_params)
    model = LGBMClassifier(**base_params)
    try:
        model.fit(
            Xtr_c, ytr,
            eval_set=[(Xva_c, yva)],
            eval_metric="auc",
            categorical_feature=cat_feature_arg,
            callbacks=[]
        )
    except Exception as e:
        if lgbm_device_params:
            warnings.warn(f"Falha no treino LightGBM GPU ({e}). Fazendo fallback para CPU…")
            base_params.pop("device_type", None)
            model = LGBMClassifier(**base_params)
            model.fit(
                Xtr_c, ytr,
                eval_set=[(Xva_c, yva)],
                eval_metric="auc",
                categorical_feature=cat_feature_arg,
                callbacks=[]
            )
        else:
            raise
    return model, cat_maps

def train_xgb(Xtr, ytr, Xva, yva, cat_cols, args, xgb_device_params):
    if XGBClassifier is None:
        raise SystemExit("XGBoost não está instalado. Rode: pip install xgboost")
    use_gpu = xgb_device_params.get("tree_method", "") == "gpu_hist"
    cat_maps = fit_category_maps(Xtr, cat_cols)
    Xtr_c = apply_category_maps(Xtr, cat_maps, as_codes=True)
    Xva_c = apply_category_maps(Xva, cat_maps, as_codes=True)
    pos = float(np.sum(ytr)); neg = float(len(ytr) - pos)
    spw = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    base_params = dict(
        n_estimators=args.iters,
        learning_rate=args.lr,
        max_depth=args.depth if args.depth > 0 else 8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=args.l2,
        random_state=42,
        eval_metric="auc",
        enable_categorical=False,
        scale_pos_weight=spw,
        n_jobs=-1,
        tree_method="hist",
        missing=-1,
        max_bin=256,
        early_stopping_rounds=args.od_wait
    )
    base_params.update(xgb_device_params)
    model = XGBClassifier(**base_params)
    try:
        model.fit(
            Xtr_c, ytr,
            eval_set=[(Xva_c, yva)],
            verbose=args.verbose > 0
        )
    except Exception as e:
        if xgb_device_params:
            warnings.warn(f"Falha no treino XGBoost GPU ({e}). Fazendo fallback para CPU…")
            for k in ["tree_method", "predictor", "device"]:
                base_params.pop(k, None)
            base_params["tree_method"] = "hist"
            model = XGBClassifier(**base_params)
            model.fit(
                Xtr_c, ytr,
                eval_set=[(Xva_c, yva)],
                verbose=args.verbose > 0
            )
        else:
            raise
    return model, cat_maps

def get_probs(model, X, model_type, cat_idx=None, cat_maps=None):
    if model_type == "catboost":
        return model.predict_proba(X)[:, 1]
    elif model_type == "lgbm":
        try:
            Xc = apply_category_maps(X, cat_maps, as_codes=False)
            return model.predict_proba(Xc)[:, 1]
        except Exception:
            Xc = apply_category_maps(X, cat_maps, as_codes=True)
            return model.predict_proba(Xc)[:, 1]
    elif model_type == "xgb":
        Xc = apply_category_maps(X, cat_maps, as_codes=True)
        return model.predict_proba(Xc)[:, 1]
    else:
        raise ValueError("model_type inválido")

def feature_importances(model, cols, model_type):
    if model_type == "catboost":
        imp = model.get_feature_importance()
    elif model_type in ("lgbm", "xgb"):
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            imp = np.zeros(len(cols))
    else:
        imp = np.zeros(len(cols))
    return dict(zip(cols, [float(x) for x in imp]))

def drop_features_from_df(df: pd.DataFrame, drops: list[str]) -> pd.DataFrame:
    if not drops:
        return df
    keep = [c for c in df.columns if c not in set(drops)]
    return df[keep].copy()

def drop_from_lists(cat_cols: list[str], num_cols: list[str], drops: list[str]) -> tuple[list[str], list[str]]:
    s = set(drops)
    return [c for c in cat_cols if c not in s], [c for c in num_cols if c not in s]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["catboost", "lgbm", "xgb"], default="catboost")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2", type=float, default=3.0)
    parser.add_argument("--od-wait", dest="od_wait", type=int, default=500)
    parser.add_argument("--verbose", type=int, default=200)
    parser.add_argument("--balance", choices=["none","undersample_neg","month_quota","route_quota"], default="none")
    parser.add_argument("--target-pos", dest="target_pos", type=float, default=0.35)
    parser.add_argument("--neg-cap-per-group", dest="neg_cap_per_group", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--th-policy", choices=["f1","f_beta","prec_at","recall_at"], default="f1")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--min-precision", dest="min_precision", type=float, default=None)
    parser.add_argument("--min-recall", dest="min_recall", type=float, default=None)
    parser.add_argument("--device", choices=["cpu","gpu"], default="cpu")
    parser.add_argument("--drop-features", nargs="*", default=[])
    parser.add_argument("--prune-below", type=float, default=None)
    args = parser.parse_args()

    tr = load_split("train")
    va = load_split("val")
    te = load_split("test")

    tr, va, te = impute_dist_km_by_route(tr, va, te)

    if args.balance != "none":
        before = len(tr)
        tr = balance_train_df(
            tr,
            mode=args.balance,
            target_col=TARGET,
            target_pos=args.target_pos,
            neg_cap_per_group=args.neg_cap_per_group,
            seed=args.seed,
        )
        after = len(tr)
        pos_rate = float(tr[TARGET].mean())
        print(f"✔ Balanceamento '{args.balance}': {before:,} → {after:,} linhas | pos_rate≈{pos_rate:.2%}")

    if args.drop_features:
        print("🔧 Removendo features:", ", ".join(args.drop_features))
        tr = drop_features_from_df(tr, args.drop_features)
        va = drop_features_from_df(va, args.drop_features)
        te = drop_features_from_df(te, args.drop_features)

    cat_cols, num_cols = infer_feature_columns(tr)
    cat_cols, num_cols = drop_from_lists(cat_cols, num_cols, args.drop_features)

    Xtr, ytr, cols, cat_idx = build_Xy(tr, cat_cols, num_cols)
    Xva, yva, _, _ = build_Xy(va, cat_cols, num_cols)
    Xte, yte, _, _ = build_Xy(te, cat_cols, num_cols)

    use_gpu, cat_dev, lgbm_dev, xgb_dev = resolve_device_args(args.device)
    if use_gpu:
        print("⚙️  Treinando em GPU.")
    else:
        print("⚙️  Treinando em CPU.")

    model_type = args.model
    cat_maps = None
    if model_type == "catboost":
        model = train_catboost(Xtr, ytr, Xva, yva, cols, cat_idx, args, cat_dev)
    elif model_type == "lgbm":
        model, cat_maps = train_lgbm(Xtr, ytr, Xva, yva, cat_cols, args, lgbm_dev)
    elif model_type == "xgb":
        model, cat_maps = train_xgb(Xtr, ytr, Xva, yva, cat_cols, args, xgb_dev)
    else:
        raise SystemExit("Modelo inválido.")

    p_tr = get_probs(model, Xtr, model_type, cat_idx=cat_idx, cat_maps=cat_maps)
    p_va = get_probs(model, Xva, model_type, cat_idx=cat_idx, cat_maps=cat_maps)
    p_te = get_probs(model, Xte, model_type, cat_idx=cat_idx, cat_maps=cat_maps)

    thr, thr_info = pick_threshold(
        yva, p_va,
        policy=args.th_policy,
        beta=args.beta,
        min_precision=args.min_precision,
        min_recall=args.min_recall
    )

    metrics = {
        "sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "class_balance": {
            "train_pos_rate": float(np.mean(ytr)),
            "val_pos_rate": float(np.mean(yva)),
            "test_pos_rate": float(np.mean(yte)),
        },
        "threshold": thr,
        "threshold_policy": thr_info,
        "train": eval_at_threshold(ytr, p_tr, thr),
        "val":   eval_at_threshold(yva, p_va, thr),
        "test":  eval_at_threshold(yte, p_te, thr),
        "features": cols,
        "categorical": cat_cols,
        "numerical": num_cols,
        "categorical_idx": [int(i) for i in range(len(cols)) if cols[i] in cat_cols],
        "feature_importance": feature_importances(model, cols, model_type),
        "model_type": model_type,
        "params": {
            "iters": args.iters, "lr": args.lr, "depth": args.depth,
            "l2": args.l2, "od_wait": args.od_wait,
            "balance": args.balance, "target_pos": args.target_pos,
            "neg_cap_per_group": args.neg_cap_per_group,
            "device": args.device,
            "drop_features": args.drop_features,
            "prune_below": args.prune_below
        }
    }

    if model_type == "catboost":
        out_path = MODELS_DIR / "catboost_delay.cbm"
        model.save_model(str(out_path))
    elif model_type == "lgbm":
        out_path = MODELS_DIR / "lgbm_delay.pkl"
        joblib.dump({"model": model, "cat_maps": cat_maps, "cols": cols, "cat_cols": cat_cols, "num_cols": num_cols}, out_path)
    elif model_type == "xgb":
        out_path = MODELS_DIR / "xgb_delay.pkl"
        joblib.dump({"model": model, "cat_maps": cat_maps, "cols": cols, "cat_cols": cat_cols, "num_cols": num_cols}, out_path)

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"feature": cols, "importance": [metrics["feature_importance"][c] for c in cols]}) \
      .sort_values("importance", ascending=False) \
      .to_csv(REPORTS_DIR / f"feature_importance_{model_type}.csv", index=False, encoding="utf-8")

    print(f"\n✔ Modelo ({model_type}) salvo em {out_path}")
    print(f"✔ Threshold ({metrics['threshold_policy']['policy']}): {thr:.4f}")
    print("✔ Métricas em reports/metrics.json")
    print(f"Val  → AUC: {metrics['val']['AUC']:.4f} | PR-AUC: {metrics['val']['PR_AUC']:.4f} | F1: {metrics['val']['F1']:.4f}")
    print(f"Test → AUC: {metrics['test']['AUC']:.4f} | PR-AUC: {metrics['test']['PR_AUC']:.4f} | F1: {metrics['test']['F1']:.4f}")
    top8 = list(pd.Series(metrics["feature_importance"]).sort_values(ascending=False).head(8).index)
    print("Top features:", top8)

    if args.prune_below is not None:
        imp_dict = feature_importances(model, cols, model_type)
        to_drop = sorted([c for c, v in imp_dict.items() if float(v) < args.prune_below])
        if to_drop:
            print(f"✂️  Prune < {args.prune_below:.4f}: removendo {len(to_drop)} features.")
            tr2 = drop_features_from_df(tr, to_drop)
            va2 = drop_features_from_df(va, to_drop)
            te2 = drop_features_from_df(te, to_drop)
            cat_cols2, num_cols2 = infer_feature_columns(tr2)
            cat_cols2, num_cols2 = drop_from_lists(cat_cols2, num_cols2, to_drop)
            Xtr2, ytr2, cols2, cat_idx2 = build_Xy(tr2, cat_cols2, num_cols2)
            Xva2, yva2, _, _ = build_Xy(va2, cat_cols2, num_cols2)
            Xte2, yte2, _, _ = build_Xy(te2, cat_cols2, num_cols2)
            if model_type == "catboost":
                model2 = train_catboost(Xtr2, ytr2, Xva2, yva2, cols2, cat_idx2, args, cat_dev)
                cat_maps2 = None
            elif model_type == "lgbm":
                model2, cat_maps2 = train_lgbm(Xtr2, ytr2, Xva2, yva2, cat_cols2, args, lgbm_dev)
            elif model_type == "xgb":
                model2, cat_maps2 = train_xgb(Xtr2, ytr2, Xva2, yva2, cat_cols2, args, xgb_dev)
            else:
                raise SystemExit("Modelo inválido no prune.")
            p_tr2 = get_probs(model2, Xtr2, model_type, cat_idx=cat_idx2, cat_maps=cat_maps2)
            p_va2 = get_probs(model2, Xva2, model_type, cat_idx=cat_idx2, cat_maps=cat_maps2)
            p_te2 = get_probs(model2, Xte2, model_type, cat_idx=cat_idx2, cat_maps=cat_maps2)
            thr2, thr_info2 = pick_threshold(yva2, p_va2, policy=args.th_policy, beta=args.beta,
                                             min_precision=args.min_precision, min_recall=args.min_recall)
            metrics_pruned = {
                "prune_below": args.prune_below,
                "dropped_features": to_drop,
                "sizes": {"train": int(len(tr2)), "val": int(len(va2)), "test": int(len(te2))},
                "threshold": thr2,
                "threshold_policy": thr_info2,
                "train": eval_at_threshold(ytr2, p_tr2, thr2),
                "val":   eval_at_threshold(yva2, p_va2, thr2),
                "test":  eval_at_threshold(yte2, p_te2, thr2),
                "features": cols2,
                "categorical": cat_cols2,
                "numerical": num_cols2,
                "feature_importance": feature_importances(model2, cols2, model_type),
                "model_type": model_type,
                "params": metrics["params"] | {"pruned": True}
            }
            (REPORTS_DIR / "metrics_pruned.json").write_text(json.dumps(metrics_pruned, indent=2, ensure_ascii=False), encoding="utf-8")
            pd.DataFrame({"feature": cols2, "importance": [metrics_pruned["feature_importance"][c] for c in cols2]}) \
              .sort_values("importance", ascending=False) \
              .to_csv(REPORTS_DIR / f"feature_importance_{model_type}_pruned.csv", index=False, encoding="utf-8")
            if model_type == "catboost":
                out_path2 = MODELS_DIR / "catboost_delay_pruned.cbm"
                model2.save_model(str(out_path2))
            elif model_type == "lgbm":
                out_path2 = MODELS_DIR / "lgbm_delay_pruned.pkl"
                joblib.dump({"model": model2, "cat_maps": cat_maps2, "cols": cols2, "cat_cols": cat_cols2, "num_cols": num_cols2}, out_path2)
            elif model_type == "xgb":
                out_path2 = MODELS_DIR / "xgb_delay_pruned.pkl"
                joblib.dump({"model": model2, "cat_maps": cat_maps2, "cols": cols2, "cat_cols": cat_cols2, "num_cols": num_cols2}, out_path2)
            print(f"✔ Modelo podado salvo em {out_path2}")
        else:
            print(f"✂️  Prune: nenhuma feature com importância < {args.prune_below:.4f}")

if __name__ == "__main__":
    main()
