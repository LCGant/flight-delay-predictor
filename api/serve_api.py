from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ==============================================================================
# Caminhos / artefatos
# ==============================================================================
ROOT = (
    Path(__file__).resolve().parents[1]
    if (Path(__file__).name == "serve_api.py")
    else Path(".")
)
MODEL_PATH = ROOT / "data/models/rf_cpu_delay.joblib"
METRICS_PATH = ROOT / "reports/metrics.json"
CLIMO_PATH = ROOT / "data/processed/wx_climo_hourly.parquet"

# Feature store opcional (Parquet/CSV). Se os arquivos não existirem, nada quebra.
FEATURES_DIR = ROOT / "data/feature_store"
FS_AIRLINE_PATH = FEATURES_DIR / "by_airline.parquet"        # cols: icao_empresa, hist_atraso_empresa_50
FS_ROUTE_PATH = FEATURES_DIR / "by_route.parquet"            # cols: rota, hist_atraso_rota_30, hist_vol_rota_30, hist_std_rota_30
FS_FLIGHTNUM_PATH = FEATURES_DIR / "by_flightnum.parquet"    # cols: airline_iata, flight_number_num, hist_atraso_num_voo_10
FS_ORIG_HOUR_PATH = FEATURES_DIR / "by_airport_hour.parquet" # cols: origem_icao, hora_bloco_30, hist_atraso_origem_hora_30

TARGET = "atraso15"
CATEGORICAL_BASE = ["icao_empresa", "origem_icao", "destino_icao", "rota"]

# ==============================================================================
# Carregamento de modelo + metadados (com hot-reload)
# ==============================================================================
rf = None
cols: list[str] = []
cat_maps: Dict[str, Dict[str, int]] = {}
medians: Dict[str, float] = {}
thresh: float = 0.5

def _load_bundle():
    global rf, cols, cat_maps, medians
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Modelo não encontrado em {MODEL_PATH}. Treine e salve em data/models/rf_cpu_delay.joblib."
        )
    bundle = joblib.load(MODEL_PATH)
    rf = bundle["rf"]
    cols = bundle["cols"]
    cat_maps = bundle["cat_maps"]
    medians = bundle["medians"]

def _load_threshold():
    global thresh
    if METRICS_PATH.exists():
        try:
            thresh = float(
                json.loads(METRICS_PATH.read_text(encoding="utf-8")).get(
                    "threshold", 0.5
                )
            )
        except Exception:
            thresh = 0.5
    else:
        thresh = 0.5

def _load_parquet_or_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return None

# Climo
def load_climo_df() -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(CLIMO_PATH)
        df = (
            df[["month", "hour", "p_rain", "p_bad"]]
            .dropna(subset=["month", "hour"])
            .copy()
        )
        df["month"] = df["month"].astype(int)
        df["hour"] = df["hour"].astype(int)
        br = (
            df.groupby(["month", "hour"], dropna=False)
            .agg(p_rain=("p_rain", "mean"), p_bad=("p_bad", "mean"))
            .reset_index()
        )
        return br
    except Exception:
        return None

# Feature store em memória
FS_AIRLINE = None
FS_ROUTE = None
FS_FLIGHTNUM = None
FS_ORIG_HOUR = None

def _load_feature_store():
    global FS_AIRLINE, FS_ROUTE, FS_FLIGHTNUM, FS_ORIG_HOUR
    FS_AIRLINE = _load_parquet_or_csv(FS_AIRLINE_PATH)
    FS_ROUTE = _load_parquet_or_csv(FS_ROUTE_PATH)
    FS_FLIGHTNUM = _load_parquet_or_csv(FS_FLIGHTNUM_PATH)
    FS_ORIG_HOUR = _load_parquet_or_csv(FS_ORIG_HOUR_PATH)

# Carrega tudo na inicialização
_load_bundle()
_load_threshold()
CLIMO_BR = load_climo_df()
_load_feature_store()

# ==============================================================================
# Tabelas auxiliares (IATA/ICAO)
# ==============================================================================
IATA_TO_ICAO = {
    # SP
    "GRU": "SBGR", "CGH": "SBSP", "VCP": "SBKP", "RAO": "SBRP", "SJK": "SBSJ", "SJP": "SBSR",
    "ARU": "SBAU", "UBT": "SDUB", "JUND": "SDJU",
    # RJ
    "GIG": "SBGL", "SDU": "SBRJ", "CFB": "SBAF", "MEA": "SBME",
    # MG
    "CNF": "SBCF", "PLU": "SBBH", "UDI": "SBUL", "UBA": "SBUR", "JDF": "SBJF",
    # DF
    "BSB": "SBBR",
    # PR
    "CWB": "SBCT", "IGU": "SBFI", "LDB": "SBLO", "MGF": "SBMG", "FBE": "SSFB",
    # RS
    "POA": "SBPA", "CXJ": "SBCX", "PEL": "SBPK",
    # SC
    "FLN": "SBFL", "JOI": "SBJV", "NVT": "SBNF", "XAP": "SBCH",
    # BA
    "SSA": "SBSV", "IOS": "SBIL", "BPS": "SBPS",
    # PE / PB / RN / AL
    "REC": "SBRF", "JPA": "SBJP", "CPV": "SBKG", "NAT": "SBSG", "MCZ": "SBMO",
    # CE / MA / PI
    "FOR": "SBFZ", "SLZ": "SBSL", "THE": "SBTE", "JDO": "SBJU",
    # PA / AP / AM / RR
    "BEL": "SBBE", "MCP": "SBMQ", "MAO": "SBEG", "BVB": "SBBV",
    # MT / MS / GO / TO
    "CGB": "SBCY", "CGR": "SBCG", "GYN": "SBGO", "PMW": "SBPJ",
    # ES
    "VIX": "SBVT",
    # RO / AC
    "PVH": "SBPV", "RBR": "SBRB",
    # Outros comuns
    "VIX": "SBVT", "BEL": "SBBE", "MAO": "SBEG", "NVT": "SBNF", "JOI": "SBJV",
}

AIRLINE_IATA_TO_ICAO = {"G3": "GLO", "LA": "TAM", "AD": "AZU", "2Z": "VOP"}

def iata_to_icao(code: Optional[str]) -> str:
    if not code:
        return ""
    code = code.strip().upper()
    return IATA_TO_ICAO.get(code, code)

def airline_to_icao(code: Optional[str]) -> str:
    if not code:
        return ""
    code = code.strip().upper()
    return AIRLINE_IATA_TO_ICAO.get(code, code)

# ==============================================================================
# Feriados / datas
# ==============================================================================
_holiday_cache: Dict[int, set[date]] = {}

def _easter_date(y: int) -> date:
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, month, day)

def _br_holidays(y: int) -> set[date]:
    if y in _holiday_cache:
        return _holiday_cache[y]
    fixed = {
        date(y, 1, 1),
        date(y, 4, 21),
        date(y, 5, 1),
        date(y, 9, 7),
        date(y, 10, 12),
        date(y, 11, 2),
        date(y, 11, 15),
        date(y, 12, 25),
    }
    easter = _easter_date(y)
    carnival_tue = easter - timedelta(days=47)
    good_friday = easter - timedelta(days=2)
    corpus = easter + timedelta(days=60)
    allh = fixed | {carnival_tue, good_friday, corpus}
    _holiday_cache[y] = allh
    return allh

def _is_holiday(d: date) -> bool:
    return d in _br_holidays(d.year)

def _is_eve_of_holiday(d: date) -> bool:
    return (d + timedelta(days=1)) in _br_holidays(d.year)

def build_time_features(departure_iso: str) -> dict:
    # Parse ISO; se vier com tz, converte para America/Sao_Paulo; se for naïve, assume São Paulo
    dt = pd.to_datetime(departure_iso, errors="coerce")
    if pd.isna(dt):
        return dict(
            mes=np.nan,
            dia_semana=np.nan,
            hora_partida_prevista=np.nan,
            hora_bloco_30=np.nan,
            hora_sin=np.nan,
            hora_cos=np.nan,
            periodo_dia_id=np.nan,
            is_weekend=np.nan,
            is_feriado=np.nan,
            is_vespera_feriado=np.nan,
            em_ferias=np.nan,
        )
    try:
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.tz_convert("America/Sao_Paulo")
        else:
            dt = dt.tz_localize("America/Sao_Paulo")
        dt = dt.tz_localize(None)
    except Exception:
        pass

    mes = int(dt.month)
    dia_semana = int(dt.weekday())
    minute_of_day = int(dt.hour) * 60 + int(dt.minute)
    hora_partida_prevista = float(minute_of_day)
    hora_bloco_30 = int(minute_of_day // 30)

    ang = 2 * np.pi * (minute_of_day / 1440.0)
    hora_sin = float(np.sin(ang))
    hora_cos = float(np.cos(ang))

    if 0 <= dt.hour < 6:
        periodo = 0
    elif 6 <= dt.hour < 12:
        periodo = 1
    elif 12 <= dt.hour < 18:
        periodo = 2
    else:
        periodo = 3

    is_weekend = 1 if dia_semana >= 5 else 0

    d = dt.date()
    is_feriado = 1 if _is_holiday(d) else 0
    is_vespera = 1 if _is_eve_of_holiday(d) else 0
    em_ferias = 1 if mes in (1, 7, 12) else 0

    return dict(
        mes=mes,
        dia_semana=dia_semana,
        hora_partida_prevista=hora_partida_prevista,
        hora_bloco_30=hora_bloco_30,
        hora_sin=hora_sin,
        hora_cos=hora_cos,
        periodo_dia_id=periodo,
        is_weekend=is_weekend,
        is_feriado=is_feriado,
        is_vespera_feriado=is_vespera,
        em_ferias=em_ferias,
    )

def compute_sched_block(
    departure_iso: Optional[str], arrival_iso: Optional[str], duration_min: Optional[float]
) -> Tuple[float, float]:
    mins = np.nan
    if arrival_iso and departure_iso:
        dt_dep = pd.to_datetime(departure_iso, errors="coerce")
        dt_arr = pd.to_datetime(arrival_iso, errors="coerce")
        if (not pd.isna(dt_dep)) and (not pd.isna(dt_arr)):
            delta = (dt_arr - dt_dep).total_seconds() / 60.0
            if delta >= 0:
                mins = float(delta)
    if (not np.isfinite(mins)) and (duration_min is not None):
        try:
            mins = float(duration_min)
        except Exception:
            pass
    if not np.isfinite(mins):
        return np.nan, np.nan
    bucket = int(mins // 30)
    bucket = float(np.clip(bucket, 0, 20))
    return float(mins), bucket

def climo_lookup(month: int, hour: int) -> Tuple[float, float]:
    if CLIMO_BR is None or not np.isfinite(month) or not np.isfinite(hour):
        return 0.0, 0.0
    m = int(month)
    h = int(hour) % 24
    hit = CLIMO_BR[(CLIMO_BR["month"] == m) & (CLIMO_BR["hour"] == h)]
    if len(hit) == 1:
        r = float(hit.at[hit.index[0], "p_rain"])
        b = float(hit.at[hit.index[0], "p_bad"])
        if np.isfinite(r) and np.isfinite(b):
            return r, b
    return 0.0, 0.0

def apply_scenario_overrides_clear(row: dict) -> None:
    for k in ("chuva_id", "tempo_ruim_id", "chuva_arr_id", "tempo_ruim_arr_id"):
        if k in row:
            row[k] = 0.0
    for k in ("wx_dep_all_missing", "wx_arr_all_missing"):
        if k in row:
            row[k] = 0.0
    for k in ("wx_dep_cov", "wx_arr_cov"):
        if k in row:
            row[k] = 1.0

# ==============================================================================
# Pydantic request
# ==============================================================================
class PredictReq(BaseModel):
    origin_iata: str
    dest_iata: str
    departure_iso: str
    arrival_iso: Optional[str] = None
    duration_min: Optional[float] = None
    stops: Optional[int] = None

    marketing_airline_iata: Optional[str] = None
    operating_airline_iata: Optional[str] = None
    operating_airline_icao: Optional[str] = None
    airline_iata: Optional[str] = None

    flight_number: Optional[str] = None
    scenario: Optional[str] = "climo"  # "climo" | "clear" | "manual"
    web_overrides: Optional[Dict[str, Any]] = None  # quando scenario == "manual"

# ==============================================================================
# App
# ==============================================================================
app = FastAPI(
    title="Flight Delay Risk API (RF)", description="Inferência RF", version="0.5.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": rf is not None,
        "n_features": len(cols),
        "threshold": thresh,
        "climo_loaded": CLIMO_BR is not None,
        "fs": {
            "airline": FS_AIRLINE is not None,
            "route": FS_ROUTE is not None,
            "flightnum": FS_FLIGHTNUM is not None,
            "orig_hour": FS_ORIG_HOUR is not None,
        },
    }

@app.post("/reload")
def reload_all():
    """Recarrega modelo/threshold/feature store sem reiniciar o servidor."""
    _load_bundle()
    _load_threshold()
    _load_feature_store()
    return {"reloaded": True, "threshold": thresh, "n_features": len(cols)}

# ==============================================================================
# Pré-process helpers
# ==============================================================================
def _apply_cat_maps(df, cat_cols, maps):
    X = df.copy()
    for c in cat_cols:
        m = maps.get(c, {})
        s = X[c].astype(str).fillna("__NA__")
        X[c] = s.map(m).fillna(0).astype("int32")
    return X

def _apply_medians(df, num_cols, med):
    X = df.copy()
    for c in num_cols:
        X[c] = (
            pd.to_numeric(X[c], errors="coerce")
            .fillna(med.get(c, 0.0))
            .astype("float32")
        )
    return X

def _debug_print_model_input(
    X_row_np: np.ndarray,
    col_order: list[str],
    proba: float,
    thr: float,
    alert: bool,
    scen: str,
    overrides_applied: bool,
):
    pairs = []
    row_list = X_row_np.tolist()
    for i, val in enumerate(row_list):
        try:
            v = float(val)
            if not np.isfinite(v):
                v = 0.0
        except Exception:
            v = val
        pairs.append([col_order[i], v])
    print(f"— scenario: {scen} | overrides_applied: {overrides_applied}")
    print("— model_input_ordered_pairs:", json.dumps(pairs, ensure_ascii=False))
    print(f"— proba: {proba:.4f} | thr: {thr:.4f} | alert: {alert}")

def _clean_for_json(obj):
    """Converte NaN/Inf → None e numpy types → nativos (recursivo)."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj

# ==============================================================================
# Feature store lookups
# ==============================================================================
def _parse_flightnum(req: PredictReq) -> tuple[Optional[str], Optional[int]]:
    # tenta extrair IATA + número: ex "LA3158" -> ("LA", 3158)
    if req.flight_number:
        m = re.match(r"^([A-Z]{2})\s*0*?(\d+)$", req.flight_number.strip().upper())
        if m:
            return m.group(1), int(m.group(2))
    # fallback: usa operating_airline_iata + somente números do flight_number
    code = (
        (req.operating_airline_iata or req.marketing_airline_iata or req.airline_iata or "")
        .strip()
        .upper()
        or None
    )
    num = None
    if req.flight_number:
        m = re.search(r"(\d+)", req.flight_number)
        if m:
            num = int(m.group(1))
    return code, num

def _fs_lookup_apply(row: dict, req: PredictReq) -> dict:
    """Preenche as features históricas a partir dos dataframes do feature store (se existirem)."""
    # by_airline
    if (
        FS_AIRLINE is not None
        and "icao_empresa" in row
        and isinstance(row["icao_empresa"], str)
        and row["icao_empresa"]
    ):
        hit = FS_AIRLINE.loc[FS_AIRLINE["icao_empresa"] == row["icao_empresa"]]
        if len(hit) == 1 and "hist_atraso_empresa_50" in row:
            row["hist_atraso_empresa_50"] = float(hit["hist_atraso_empresa_50"].values[0])

    # by_route
    if FS_ROUTE is not None and "rota" in row and isinstance(row["rota"], str) and row["rota"]:
        hit = FS_ROUTE.loc[FS_ROUTE["rota"] == row["rota"]]
        if len(hit) == 1:
            for k in ("hist_atraso_rota_30", "hist_vol_rota_30", "hist_std_rota_30"):
                if k in row and k in hit.columns:
                    row[k] = float(hit[k].values[0])

    # by_flightnum
    if FS_FLIGHTNUM is not None and "hist_atraso_num_voo_10" in row:
        iata, num = _parse_flightnum(req)
        if iata and num is not None:
            hit = FS_FLIGHTNUM.loc[
                (FS_FLIGHTNUM["airline_iata"] == iata)
                & (FS_FLIGHTNUM["flight_number_num"] == num)
            ]
            if len(hit) == 1 and "hist_atraso_num_voo_10" in hit.columns:
                row["hist_atraso_num_voo_10"] = float(hit["hist_atraso_num_voo_10"].values[0])

    # by_airport_hour
    if FS_ORIG_HOUR is not None and "origem_icao" in row and "hora_bloco_30" in row:
        try:
            hb = int(row["hora_bloco_30"])
        except Exception:
            hb = None
        if row["origem_icao"] and hb is not None:
            hit = FS_ORIG_HOUR.loc[
                (FS_ORIG_HOUR["origem_icao"] == row["origem_icao"])
                & (FS_ORIG_HOUR["hora_bloco_30"] == hb)
            ]
            if (
                len(hit) == 1
                and "hist_atraso_origem_hora_30" in row
                and "hist_atraso_origem_hora_30" in hit.columns
            ):
                row["hist_atraso_origem_hora_30"] = float(
                    hit["hist_atraso_origem_hora_30"].values[0]
                )

    return row

# ==============================================================================
# Endpoint principal
# ==============================================================================
@app.post("/predict")
def predict(req: PredictReq):
    # ICAO da rota/cia
    origem_icao = iata_to_icao(req.origin_iata)
    destino_icao = iata_to_icao(req.dest_iata)
    # IMPORTANTE: use o MESMO separador do treino ("origem>destino"):
    rota = f"{origem_icao}>{destino_icao}" if origem_icao and destino_icao else ""

    cia_icao = (
        req.operating_airline_icao
        or airline_to_icao(req.operating_airline_iata)
        or airline_to_icao(req.airline_iata)
        or airline_to_icao(req.marketing_airline_iata)
        or ""
    )

    # tempo / duração
    t_dep = build_time_features(req.departure_iso)
    sched_min, sched_bucket = compute_sched_block(
        req.departure_iso, req.arrival_iso, req.duration_min
    )

    # linha base
    row = {c: np.nan for c in cols}
    row["icao_empresa"] = cia_icao or ""
    row["origem_icao"] = origem_icao or ""
    row["destino_icao"] = destino_icao or ""
    row["rota"] = rota or ""
    for k, v in t_dep.items():
        if k in row:
            row[k] = v
    if "sched_block_min" in row:
        row["sched_block_min"] = sched_min
    if "sched_block_bucket" in row:
        row["sched_block_bucket"] = sched_bucket

    # >>> Feature store lookups (se disponível) <<<
    row = _fs_lookup_apply(row, req)

    scen = (req.scenario or "climo").lower().strip()
    applied_overrides: Dict[str, Any] = {}

    if scen == "clear":
        apply_scenario_overrides_clear(row)

    elif scen in ("climo", "wx_climo"):
        dt_dep = pd.to_datetime(req.departure_iso, errors="coerce")
        dep_m = int(dt_dep.month) if not pd.isna(dt_dep) else int(t_dep.get("mes") or 1)
        dep_h = int(dt_dep.hour) if not pd.isna(dt_dep) else int(
            (t_dep.get("hora_partida_prevista") or 0) // 60
        )
        pr_dep, pb_dep = climo_lookup(dep_m, dep_h)

        dt_arr = pd.to_datetime(req.arrival_iso, errors="coerce") if req.arrival_iso else pd.NaT
        arr_m = int(dt_arr.month) if not pd.isna(dt_arr) else dep_m
        arr_h = int(dt_arr.hour) if not pd.isna(dt_arr) else dep_h
        pr_arr, pb_arr = climo_lookup(arr_m, arr_h)

        if "chuva_id" in row:
            row["chuva_id"] = pr_dep
        if "tempo_ruim_id" in row:
            row["tempo_ruim_id"] = pb_dep
        if "chuva_arr_id" in row:
            row["chuva_arr_id"] = pr_arr
        if "tempo_ruim_arr_id" in row:
            row["tempo_ruim_arr_id"] = pb_arr
        if "wx_dep_cov" in row:
            row["wx_dep_cov"] = 1.0
        if "wx_arr_cov" in row:
            row["wx_arr_cov"] = 1.0
        if "wx_dep_all_missing" in row:
            row["wx_dep_all_missing"] = 0.0
        if "wx_arr_all_missing" in row:
            row["wx_arr_all_missing"] = 0.0

    elif scen == "manual":
        # normalização de overrides vindos do front
        ov = dict(req.web_overrides or {})

        # aceita "clima_origem"/"clima_destino" (0..4) e mapeia para 0..1 se usu não preencher p* diretamente
        if "clima_origem" in ov and ("chuva_id" not in ov and "tempo_ruim_id" not in ov):
            try:
                lvl = float(ov["clima_origem"])
                p = max(0.0, min(1.0, lvl / 4.0))
                ov["chuva_id"] = p
                ov["tempo_ruim_id"] = p
            except Exception:
                pass
        if "clima_destino" in ov and ("chuva_arr_id" not in ov and "tempo_ruim_arr_id" not in ov):
            try:
                lvl = float(ov["clima_destino"])
                p = max(0.0, min(1.0, lvl / 4.0))
                ov["chuva_arr_id"] = p
                ov["tempo_ruim_arr_id"] = p
            except Exception:
                pass

        # aplica overrides com clipping “amigável”
        for k, v in ov.items():
            if k not in row or v is None:
                continue
            try:
                if k in {"wx_dep_all_missing", "wx_arr_all_missing"}:
                    row[k] = 1.0 if bool(v) else 0.0
                elif k in {
                    "chuva_id",
                    "tempo_ruim_id",
                    "chuva_arr_id",
                    "tempo_ruim_arr_id",
                    "wx_dep_cov",
                    "wx_arr_cov",
                }:
                    vv = float(v)
                    vv = 0.0 if not np.isfinite(vv) else max(0.0, min(1.0, vv))
                    row[k] = vv
                elif k in {
                    "congestion_bucket",
                    "sched_block_bucket",
                    "load_origem_15",
                    "airport_size_id",
                }:
                    vv = int(float(v))
                    vv = max(0, min(20, vv))
                    row[k] = float(vv)
                elif k in {"congestion_ratio"}:
                    vv = float(v)
                    row[k] = float(max(0.0, min(5.0, vv)))
                elif k in {"sched_block_min", "dist_km", "hist_vol_rota_30", "hist_std_rota_30"}:
                    vv = float(v)
                    row[k] = float(max(0.0, min(20000.0, vv)))
                else:
                    vv = float(v)
                    if not np.isfinite(vv):
                        continue
                    row[k] = vv
                applied_overrides[k] = row[k]
            except Exception:
                continue

    else:
        apply_scenario_overrides_clear(row)

    # monta X e aplica encoders
    X = pd.DataFrame([row], columns=cols)
    cat_in = [c for c in cols if c in CATEGORICAL_BASE]
    num_in = [c for c in cols if c not in CATEGORICAL_BASE]
    X_enc = _apply_cat_maps(X, cat_in, cat_maps)
    X_enc = _apply_medians(X_enc, num_in, medians)
    X_np = X_enc[cols].to_numpy(dtype=np.float32)

    # inferência
    proba = float(rf.predict_proba(X_np)[0, 1])
    alert = bool(proba >= float(thresh))

    # debug
    _debug_print_model_input(
        X_np[0], cols, proba, float(thresh), alert, scen, bool(applied_overrides)
    )

    # resposta
    resp = {
        "probability": proba,
        "threshold": float(thresh),
        "alert": alert,
        "inputs": {
            "origin_iata": (req.origin_iata or "").upper(),
            "dest_iata": (req.dest_iata or "").upper(),
            "departure_iso": req.departure_iso,
            "arrival_iso": req.arrival_iso,
            "duration_min": req.duration_min,
            "stops": req.stops,
            "marketing_airline_iata": (req.marketing_airline_iata or None),
            "operating_airline_iata": (req.operating_airline_iata or None),
            "operating_airline_icao": (req.operating_airline_icao or None),
            "airline_iata": (req.airline_iata or None),
            "flight_number": req.flight_number or "",
            "scenario": scen,
        },
        "derived": {
            "icao_empresa": row.get("icao_empresa"),
            "origem_icao": row.get("origem_icao"),
            "destino_icao": row.get("destino_icao"),
            "rota": row.get("rota"),
            "tempo_feats": {
                "mes": row.get("mes"),
                "dia_semana": row.get("dia_semana"),
                "hora_partida_prevista": row.get("hora_partida_prevista"),
                "hora_bloco_30": row.get("hora_bloco_30"),
                "hora_sin": row.get("hora_sin"),
                "hora_cos": row.get("hora_cos"),
                "periodo_dia_id": row.get("periodo_dia_id"),
                "is_weekend": row.get("is_weekend"),
                "is_feriado": row.get("is_feriado"),
                "is_vespera_feriado": row.get("is_vespera_feriado"),
                "em_ferias": row.get("em_ferias"),
            },
            "sched_block_min": row.get("sched_block_min"),
            "sched_block_bucket": row.get("sched_block_bucket"),
            "wx_dep": {
                "p_rain": row.get("chuva_id"),
                "p_bad": row.get("tempo_ruim_id"),
                "cov": row.get("wx_dep_cov"),
            },
            "wx_arr": {
                "p_rain": row.get("chuva_arr_id"),
                "p_bad": row.get("tempo_ruim_arr_id"),
                "cov": row.get("wx_arr_cov"),
            },
            "applied_overrides": applied_overrides if scen == "manual" else {},
            # Histórico resolvido (se existir FS)
            "hist": {
                "hist_atraso_empresa_50": row.get("hist_atraso_empresa_50"),
                "hist_atraso_rota_30": row.get("hist_atraso_rota_30"),
                "hist_atraso_num_voo_10": row.get("hist_atraso_num_voo_10"),
                "hist_vol_rota_30": row.get("hist_vol_rota_30"),
                "hist_std_rota_30": row.get("hist_std_rota_30"),
                "hist_atraso_origem_hora_30": row.get("hist_atraso_origem_hora_30"),
            },
        },
    }

    safe = _clean_for_json(resp)
    return JSONResponse(content=jsonable_encoder(safe, exclude_none=False))

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.serve_api:app", host="0.0.0.0", port=8000, reload=True)
