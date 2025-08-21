from __future__ import annotations
import math, json
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# === Raiz do projeto (…/projeto_voos)
BASE_DIR = Path(__file__).resolve().parent.parent

# === Pastas
MODELS_DIR  = BASE_DIR / "data" / "models"
REPORTS_DIR = BASE_DIR / "reports"
SPLITS_DIR  = BASE_DIR / "data" / "processed" / "splits"

# ================== Load do modelo (1x) ==================
model = None
model_type = None
cat_maps: dict[str, list[str]] | None = None
cols: list[str] | None = None
cat_cols: list[str] | None = None
num_cols: list[str] | None = None

def _force_cpu_predict_xgb():
    """Se o modelo XGB foi treinado com GPU, força predição em CPU (evita warning)."""
    try:
        bst = model.get_booster()
        bst.set_param({'predictor': 'cpu_predictor', 'device': 'cpu'})
        # Alguns builds aceitam também:
        model.set_params(predictor='cpu_predictor')
    except Exception:
        pass

def _load_any_model():
    global model, model_type, cat_maps, cols, cat_cols, num_cols
    xgb_p = MODELS_DIR / "xgb_delay.pkl"
    lgb_p = MODELS_DIR / "lgbm_delay.pkl"
    cat_p = MODELS_DIR / "catboost_delay.cbm"

    if xgb_p.exists():
        import joblib
        b = joblib.load(xgb_p)
        model = b["model"]
        cat_maps = b.get("cat_maps")
        cols = b.get("cols"); cat_cols = b.get("cat_cols"); num_cols = b.get("num_cols")
        model_type = "xgb"
        _force_cpu_predict_xgb()
        print(f"✔ Modelo XGB carregado: {xgb_p}")
        return

    if lgb_p.exists():
        import joblib
        b = joblib.load(lgb_p)
        model = b["model"]
        cat_maps = b.get("cat_maps")
        cols = b.get("cols"); cat_cols = b.get("cat_cols"); num_cols = b.get("num_cols")
        model_type = "lgbm"
        print(f"✔ Modelo LGBM carregado: {lgb_p}")
        return

    if cat_p.exists():
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(str(cat_p))
        model = m
        model_type = "catboost"
        print(f"✔ Modelo CatBoost carregado: {cat_p}")
        return

    raise SystemExit(f"Nenhum modelo encontrado em {MODELS_DIR}/")

def _load_metrics():
    p = REPORTS_DIR / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

_load_any_model()
metrics = _load_metrics()

# ================== Utilitários de encoding ==================
def apply_category_maps(X: pd.DataFrame, cat_maps: dict[str, list[str]] | None, as_codes: bool) -> pd.DataFrame:
    if not cat_maps:
        return X.copy()
    X2 = X.copy()
    for c, cats in cat_maps.items():
        if c in X2.columns:
            dtype = pd.api.types.CategoricalDtype(categories=cats)
            X2[c] = X2[c].astype("string").astype(dtype)
            if as_codes:
                X2[c] = X2[c].cat.codes.astype("int32")  # -1 para NaN
    return X2

def get_probs(model, X: pd.DataFrame) -> np.ndarray:
    # Normaliza tipos numéricos para evitar “object”
    for c in X.columns:
        if c not in ("icao_empresa","origem_icao","destino_icao","rota"):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    if model_type == "xgb":
        Xc = apply_category_maps(X, cat_maps, as_codes=True)
        return model.predict_proba(Xc)[:, 1]
    if model_type == "lgbm":
        try:
            Xc = apply_category_maps(X, cat_maps, as_codes=False)
            return model.predict_proba(Xc)[:, 1]
        except Exception:
            Xc = apply_category_maps(X, cat_maps, as_codes=True)
            return model.predict_proba(Xc)[:, 1]
    if model_type == "catboost":
        return model.predict_proba(X)[:, 1]
    raise RuntimeError("model_type inválido")

# ================== Opções do formulário ==================
def _safe_upper(x): 
    return pd.Series(x, dtype="string").str.upper().str.strip()

def load_form_options():
    empresas = aeroportos = None
    if cat_maps:
        empresas = [e for e in cat_maps.get("icao_empresa", []) if isinstance(e, str) and e]
        origs = cat_maps.get("origem_icao", [])
        dests = cat_maps.get("destino_icao", [])
        if origs or dests:
            aeroportos = sorted({*origs, *dests})
    if (not empresas or not aeroportos) and (SPLITS_DIR / "train.parquet").exists():
        df = pd.read_parquet(SPLITS_DIR / "train.parquet")
        if empresas is None and "icao_empresa" in df.columns:
            empresas = _safe_upper(df["icao_empresa"]).value_counts().head(30).index.tolist()
        if aeroportos is None:
            opts = []
            for c in ("origem_icao","destino_icao"):
                if c in df.columns:
                    opts += _safe_upper(df[c]).value_counts().head(60).index.tolist()
            aeroportos = sorted(set(opts))
    if not empresas:   empresas = ["TAM","GLO","AZU","VOE"]  # ICAO típicos (ex.: LATAM=TAM, GOL=GLO)
    if not aeroportos: aeroportos = ["GRU","CGH","GIG","SDU","BSB","CNF","VCP","REC","POA","SSA"]
    return empresas, aeroportos

EMPRESAS, AEROPORTOS = load_form_options()

# ================== Montagem da linha de features ==================
def build_feature_row(form) -> pd.DataFrame:
    def as_int(name, default=0, lo=None, hi=None):
        try:
            v = int(form.get(name, default))
            if lo is not None: v = max(lo, v)
            if hi is not None: v = min(hi, v)
            return v
        except Exception:
            return default
    def as_float(name, default=np.nan, lo=None, hi=None):
        try:
            raw = form.get(name, "")
            if raw == "" or raw is None: return default
            v = float(raw)
            if lo is not None: v = max(lo, v)
            if hi is not None: v = min(hi, v)
            return v
        except Exception:
            return default

    cia  = str(form.get("icao_empresa", "")).upper().strip()
    orig = str(form.get("origem_icao", "")).upper().strip()
    dest = str(form.get("destino_icao", "")).upper().strip()
    mes  = as_int("mes", 1, 1, 12)
    dow  = as_int("dia_semana", 0, 0, 6)
    hora = as_int("hora", 12, 0, 23)
    chuva = as_float("chuva_mm", np.nan, 0)
    tempC = as_float("temp_c",   np.nan, -20, 55)
    vento = as_float("vento_kmh", np.nan, 0)

    rota = f"{orig}>{dest}" if orig and dest else ""
    blk = int(hora*2)
    sec = float(hora*3600)
    ang = 2*math.pi*(sec/86400.0)
    hora_sin, hora_cos = math.sin(ang), math.cos(ang)
    periodo = 0 if hora<=5 else (1 if hora<=11 else (2 if hora<=17 else 3))
    is_weekend = 1 if dow>=5 else 0
    wind_ms = (vento/3.6) if not np.isnan(vento) else np.nan

    base = {
        "icao_empresa": cia, "origem_icao": orig, "destino_icao": dest, "rota": rota,
        "mes": mes, "dia_semana": dow, "hora_bloco_30": blk,
        "hora_sin": hora_sin, "hora_cos": hora_cos, "periodo_dia_id": periodo,
        "is_weekend": is_weekend, "is_feriado": 0, "is_vespera_feriado": 0,
        "em_ferias": 1 if mes in (1,7) else 0,
        "w_dep_precip_mm": chuva, "w_dep_wind_ms": wind_ms, "w_dep_gust_ms": np.nan,
        "w_dep_temp_c": tempC, "w_dep_rh": np.nan,
    }
    for k in ["w_dep_precip_mm","w_dep_wind_ms","w_dep_gust_ms","w_dep_temp_c","w_dep_rh"]:
        base[f"{k}_isna"] = 1 if pd.isna(base[k]) else 0

    base["wx_chuva"]        = 0 if pd.isna(chuva) else int(chuva >= 1.0)
    base["wx_vento_forte"]  = 0 if pd.isna(wind_ms) else int(wind_ms >= 8.0)
    base["wx_rajada_forte"] = 0
    base["wx_calor"]        = 0 if pd.isna(tempC) else int(tempC >= 30.0)
    base["wx_frio"]         = 0 if pd.isna(tempC) else int(tempC <= 12.0)
    base["wx_umidade_alta"] = 0
    base["wx_missing_any"]  = int(any(base[f"{k}_isna"] for k in
                                 ["w_dep_precip_mm","w_dep_wind_ms","w_dep_gust_ms","w_dep_temp_c","w_dep_rh"]))

    # Features que não estão no formulário
    base.update({
        "load_origem_15": np.nan, "dist_km": np.nan,
        "w_arr_precip_mm": np.nan, "w_arr_wind_ms": np.nan, "w_arr_gust_ms": np.nan,
        "w_arr_temp_c": np.nan, "w_arr_rh": np.nan,
        "hist_atraso_rota_30": np.nan, "hist_atraso_empresa_50": np.nan,
        "hist_atraso_num_voo_10": np.nan, "hist_vol_rota_30": np.nan, "hist_std_rota_30": np.nan,
    })

    X = pd.DataFrame([base])

    # Garante ordem/colunas do treino
    global cols
    if cols:
        for c in cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[cols]
    return X

# Threshold e métricas
DEFAULT_THR = 0.5
THR   = float(metrics.get("threshold", DEFAULT_THR)) if metrics else DEFAULT_THR
AUC   = metrics.get("test", {}).get("AUC") if metrics else None
PRAUC = metrics.get("test", {}).get("PR_AUC") if metrics else None

# ================== Flask ==================
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "web" / "templates"),
    static_folder=str(BASE_DIR / "web" / "static"),
)

@app.route("/", methods=["GET","POST"])
def index():
    proba = None
    will_delay = None
    error_msg = None

    if request.method == "POST":
        try:
            X = build_feature_row(request.form)
            proba = float(get_probs(model, X)[0])
            will_delay = (proba >= THR)
        except Exception as e:
            error_msg = f"Erro ao calcular: {type(e).__name__}: {e}"
            proba = None
            will_delay = None

    return render_template(
        "index.html",
        empresas=EMPRESAS,
        aerop=AEROPORTOS,
        proba=proba,
        will_delay=will_delay,
        thr=THR,
        auc=AUC, prauc=PRAUC,
        model_type=model_type,
        error_msg=error_msg
    )

if __name__ == "__main__":
    # debug=False para não derrubar o worker do Plesk quando dá exceção
    app.run(host="0.0.0.0", port=5000, debug=False)
