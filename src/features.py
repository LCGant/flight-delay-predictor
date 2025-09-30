# features.py
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from pandas.api.types import is_categorical_dtype, is_interval_dtype

def _ensure_series_datetime(x, col: str = "partida_prevista") -> pd.Series:
    if isinstance(x, pd.DataFrame):
        s = x[col] if col in x.columns else pd.Series(pd.NaT, index=x.index)
    else:
        s = pd.Series(x)
    s = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        s = s.dt.tz_convert(None)
    return s

def _safe_upper_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()

def _brazil_holidays_2023_2024() -> set:
    fixed = ["2023-01-01","2023-04-21","2023-05-01","2023-09-07","2023-10-12","2023-11-02","2023-11-15",
             "2023-12-25","2024-01-01","2024-04-21","2024-05-01","2024-09-07","2024-10-12","2024-11-02",
             "2024-11-15","2024-12-25"]
    moveable = ["2023-02-20","2023-02-21","2023-04-07","2023-06-08","2024-02-12","2024-02-13","2024-03-29","2024-05-30"]
    all_days = pd.to_datetime(fixed + moveable).date
    return set(all_days)

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("icao_empresa","origem_icao","destino_icao"):
        if c in out.columns:
            out[c] = _safe_upper_strip(out[c])
    if "rota" not in out.columns and {"origem_icao","destino_icao"} <= set(out.columns):
        out["rota"] = _safe_upper_strip(out["origem_icao"]) + ">" + _safe_upper_strip(out["destino_icao"])
    dt = _ensure_series_datetime(out, "partida_prevista")
    if "mes" not in out.columns:
        out["mes"] = dt.dt.month.astype("Int32")
    if "dia_semana" not in out.columns:
        out["dia_semana"] = dt.dt.dayofweek.astype("Int32")
    if "hora_partida_prevista" not in out.columns:
        out["hora_partida_prevista"] = dt.dt.hour.astype("Int32")
    blk = (dt.dt.hour * 2 + (dt.dt.minute >= 30).astype("int")).astype("Int16")
    out["hora_bloco_30"] = blk
    seconds = (dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second).fillna(0).astype(float)
    angle = 2 * math.pi * (seconds / 86400.0)
    out["hora_sin"] = np.sin(angle)
    out["hora_cos"] = np.cos(angle)
    h = dt.dt.hour.fillna(-1)
    periodo = pd.Series(np.where(
        (0 <= h) & (h <= 5), 0,
        np.where((6 <= h) & (h <= 11), 1,
        np.where((12 <= h) & (h <= 17), 2,
        np.where((18 <= h) & (h <= 23), 3, np.nan)))))
    out["periodo_dia_id"] = periodo.astype("Int8")
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype("Int8")
    feriados = _brazil_holidays_2023_2024()
    ddate = dt.dt.date
    is_feriado = ddate.map(lambda d: d in feriados if pd.notna(d) else False)
    is_vespera = ddate.map(lambda d: (pd.Timestamp(d) + pd.Timedelta(days=1)).date() in feriados if pd.notna(d) else False)
    out["is_feriado"] = is_feriado.astype("Int8")
    out["is_vespera_feriado"] = is_vespera.astype("Int8")
    out["em_ferias"] = dt.dt.month.isin([1,7,12]).astype("Int8")
    return out

def _key_rota(df: pd.DataFrame) -> pd.Series:
    if "rota" in df.columns:
        return _safe_upper_strip(df["rota"])
    if {"origem_icao","destino_icao"} <= set(df.columns):
        return _safe_upper_strip(df["origem_icao"]) + ">" + _safe_upper_strip(df["destino_icao"])
    return pd.Series([""], index=df.index, dtype="string")

def _grp_roll_mean(y: pd.Series, key: pd.Series, window: int, minp: int) -> pd.Series:
    return y.groupby(key).apply(lambda s: s.shift().rolling(window, min_periods=minp).mean()).reset_index(level=0, drop=True)

def _grp_roll_std(y: pd.Series, key: pd.Series, window: int, minp: int) -> pd.Series:
    return y.groupby(key).apply(lambda s: s.shift().rolling(window, min_periods=minp).std()).reset_index(level=0, drop=True)

def _grp_roll_count(y: pd.Series, key: pd.Series, window: int, minp: int) -> pd.Series:
    return y.groupby(key).apply(lambda s: s.shift().rolling(window, min_periods=minp).count()).reset_index(level=0, drop=True)

def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = _ensure_series_datetime(out, "partida_prevista")
    order = dt.argsort(kind="mergesort")
    out = out.iloc[order].copy()
    y = out.get("atraso15", pd.Series(index=out.index, dtype="float")).astype(float)
    rota_key = _key_rota(out)
    cia_key = _safe_upper_strip(out["icao_empresa"]) if "icao_empresa" in out.columns else pd.Series([""], index=out.index)
    nvoo_key = _safe_upper_strip(out["numero_voo"]) if "numero_voo" in out.columns else pd.Series([""], index=out.index)
    out["hist_atraso_rota_30"] = _grp_roll_mean(y, rota_key, 30, 5)
    out["hist_atraso_empresa_50"] = _grp_roll_mean(y, cia_key, 50, 10)
    out["hist_atraso_num_voo_10"] = _grp_roll_mean(y, nvoo_key, 10, 3)
    out["hist_vol_rota_30"] = _grp_roll_count(y, rota_key, 30, 1)
    out["hist_std_rota_30"] = _grp_roll_std(y, rota_key, 30, 5)
    if {"icao_empresa","numero_voo"} <= set(out.columns):
        grp = _safe_upper_strip(out["icao_empresa"]) + "|" + _safe_upper_strip(out["numero_voo"])
        out["hist_voo_prev_atrasou"] = out["atraso15"].groupby(grp).shift(1).fillna(0).astype(float)
    mean_global = float(pd.to_numeric(y, errors="coerce").mean()) if len(out) else 0.0
    for c in ["hist_atraso_rota_30","hist_atraso_empresa_50","hist_atraso_num_voo_10"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(mean_global)
    out["hist_vol_rota_30"] = pd.to_numeric(out["hist_vol_rota_30"], errors="coerce").fillna(0.0)
    out["hist_std_rota_30"] = pd.to_numeric(out["hist_std_rota_30"], errors="coerce").fillna(0.0)
    out = out.sort_index()
    return out

def add_history_origem_hora(df: pd.DataFrame, window: int = 30, minp: int = 5) -> pd.DataFrame:
    out = df.copy()
    need = {"origem_icao","partida_prevista","atraso15"}
    if not need <= set(out.columns):
        out["hist_atraso_origem_hora_30"] = np.nan
        return out

    out["origem_icao"] = _safe_upper_strip(out["origem_icao"])
    dt = _ensure_series_datetime(out, "partida_prevista")

    order = dt.argsort(kind="mergesort")
    work = out.iloc[order].copy()
    dt_sorted = dt.iloc[order]

    hour = dt_sorted.dt.hour.fillna(-1).astype("Int16")
    key = work["origem_icao"].astype(str) + "|" + hour.astype(str)
    y = pd.to_numeric(work["atraso15"], errors="coerce").astype(float)

    work["hist_atraso_origem_hora_30"] = _grp_roll_mean(y, key, window=window, minp=minp)

    s = work["hist_atraso_origem_hora_30"].reindex(out.index)
    mean_global = float(pd.to_numeric(out["atraso15"], errors="coerce").astype(float).mean()) if len(out) else 0.0
    out["hist_atraso_origem_hora_30"] = pd.to_numeric(s, errors="coerce").fillna(mean_global)
    return out

def add_airport_load_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "origem_icao" not in out.columns or "partida_prevista" not in out.columns:
        out["load_origem_15"] = np.nan
        return out
    out["origem_icao"] = _safe_upper_strip(out["origem_icao"])
    dt = _ensure_series_datetime(out, "partida_prevista")
    out["_bucket15"] = dt.dt.floor("15min")
    counts = out[["origem_icao","_bucket15"]].value_counts().rename("load_origem_15").reset_index()
    out = out.merge(counts, how="left", left_on=["origem_icao","_bucket15"], right_on=["origem_icao","_bucket15"])
    out.drop(columns=["_bucket15"], inplace=True)
    out["load_origem_15"] = pd.to_numeric(out["load_origem_15"], errors="coerce").fillna(0).astype("Int32")
    return out

def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"origem_icao","partida_prevista","atraso15"}
    if not required <= set(out.columns):
        out["atrasos_mesmo_aeroporto_1h_hist"] = np.nan
        return out
    out["origem_icao"] = _safe_upper_strip(out["origem_icao"])
    dt = _ensure_series_datetime(out, "partida_prevista")
    tmp = pd.DataFrame({
        "orig_idx": out.index,
        "origem_icao": out["origem_icao"],
        "partida_prevista": dt,
        "atraso15": pd.to_numeric(out["atraso15"], errors="coerce")
    }).sort_values(["origem_icao","partida_prevista","orig_idx"])
    parts = []
    for _, g in tmp.groupby("origem_icao", sort=False):
        gg = g.copy()
        gg["prev"] = gg["atraso15"].shift(1)
        roll = gg.set_index("partida_prevista")["prev"].rolling("60min", min_periods=5).mean()
        gg["atrasos_mesmo_aeroporto_1h_hist"] = roll.values
        parts.append(gg[["orig_idx","atrasos_mesmo_aeroporto_1h_hist"]])
    feat = pd.concat(parts, ignore_index=False)
    out = out.join(feat.set_index("orig_idx"), how="left")
    return out

def add_airport_size_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "origem_icao" not in out.columns:
        out["airport_size_id"] = np.nan
        out["airport_size_label"] = np.nan
        return out
    out["origem_icao"] = _safe_upper_strip(out["origem_icao"])
    vol = out.groupby("origem_icao").size().rename("vol_total").reset_index()
    q = vol["vol_total"].quantile([0.25,0.50,0.85]).to_list()
    def _size_bucket(v):
        if v <= q[0]: return 0
        if v <= q[1]: return 1
        if v <= q[2]: return 2
        return 3
    vol["airport_size_id"] = vol["vol_total"].apply(_size_bucket).astype("Int8")
    label_map = {0:"pequeno",1:"medio",2:"grande",3:"hub"}
    vol["airport_size_label"] = vol["airport_size_id"].map(label_map)
    out = out.merge(vol[["origem_icao","airport_size_id","airport_size_label"]], on="origem_icao", how="left")
    return out

def add_congestion_from_size(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "load_origem_15" not in out.columns:
        out = add_airport_load_features(out)
    if "origem_icao" not in out.columns or "load_origem_15" not in out.columns or "partida_prevista" not in out.columns:
        out["congestion_ratio"] = np.nan
        out["congestion_bucket"] = np.nan
        return out
    out["origem_icao"] = _safe_upper_strip(out["origem_icao"])
    dt = _ensure_series_datetime(out, "partida_prevista")
    out["_bucket15"] = dt.dt.floor("15min")
    p95 = out.groupby("origem_icao")["load_origem_15"].quantile(0.95).rename("p95_load").reset_index()
    out = out.merge(p95, on="origem_icao", how="left")
    out["congestion_ratio"] = (out["load_origem_15"] / out["p95_load"].replace({0: np.nan})).clip(lower=0, upper=2)
    out["congestion_bucket"] = pd.cut(out["congestion_ratio"], bins=[-np.inf,0.5,1.0,1.5,np.inf], labels=[0,1,2,3], right=True).astype("Int8")
    out.drop(columns=[c for c in ["_bucket15","p95_load"] if c in out.columns], inplace=True)
    return out

def add_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = ["w_dep_precip_mm","w_dep_wind_ms","w_dep_gust_ms","w_dep_temp_c","w_dep_rh"]
    for c in cols:
        if c in out.columns:
            out[f"{c}_isna"] = out[c].isna().astype("Int8")
        else:
            out[c] = np.nan
            out[f"{c}_isna"] = 1
    precip = pd.to_numeric(out["w_dep_precip_mm"], errors="coerce")
    wind = pd.to_numeric(out["w_dep_wind_ms"], errors="coerce")
    gust = pd.to_numeric(out["w_dep_gust_ms"], errors="coerce")
    temp = pd.to_numeric(out["w_dep_temp_c"], errors="coerce")
    rh = pd.to_numeric(out["w_dep_rh"], errors="coerce")
    out["wx_chuva"] = (precip >= 2.0).astype("Int8")
    out["wx_chuva_forte"] = (precip >= 5.0).astype("Int8")
    out["wx_vento_forte"] = (wind >= 8.0).astype("Int8")
    out["wx_rajada_forte"] = (gust >= 14.0).astype("Int8")
    out["wx_calor"] = (temp >= 30.0).astype("Int8")
    out["wx_frio"] = (temp <= 12.0).astype("Int8")
    out["wx_umidade_alta"] = (rh >= 90.0).astype("Int8")
    out["wx_missing_any"] = (out[[f"{c}_isna" for c in cols]].sum(axis=1).gt(0)).astype("Int8")
    def qstr(s):
        try:
            b = pd.qcut(s, q=[0,.5,.8,.9,1.0], duplicates="drop")
            return b.astype("string")
        except Exception:
            return pd.Series([pd.NA]*len(out), dtype="string")
    out["precip_q"] = qstr(precip)
    out["wind_q"] = qstr(wind)
    out["gust_q"] = qstr(gust)
    out["temp_q"] = qstr(temp)
    out["rh_q"] = qstr(rh)
    return out

def add_visibility_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rh = pd.to_numeric(out.get("w_dep_rh", np.nan), errors="coerce")
    temp = pd.to_numeric(out.get("w_dep_temp_c", np.nan), errors="coerce")
    wind = pd.to_numeric(out.get("w_dep_wind_ms", np.nan), errors="coerce")
    precip = pd.to_numeric(out.get("w_dep_precip_mm", np.nan), errors="coerce")
    out["visibilidade_ruim"] = ((precip >= 5.0) | ((rh >= 93.0) & (wind < 2.5)) | ((rh >= 96.0) & (temp < 5.0))).astype("Int8")
    return out

def add_weather_destino(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = ["w_arr_precip_mm","w_arr_wind_ms","w_arr_gust_ms","w_arr_temp_c","w_arr_rh"]
    for c in cols:
        if c in out.columns:
            out[f"{c}_isna"] = out[c].isna().astype("Int8")
        else:
            out[c] = np.nan
            out[f"{c}_isna"] = 1
    precip = pd.to_numeric(out["w_arr_precip_mm"], errors="coerce")
    wind = pd.to_numeric(out["w_arr_wind_ms"], errors="coerce")
    gust = pd.to_numeric(out["w_arr_gust_ms"], errors="coerce")
    temp = pd.to_numeric(out["w_arr_temp_c"], errors="coerce")
    rh = pd.to_numeric(out["w_arr_rh"], errors="coerce")
    out["wx_arr_chuva"] = (precip >= 2.0).astype("Int8")
    out["wx_arr_chuva_forte"] = (precip >= 5.0).astype("Int8")
    out["wx_arr_vento_forte"] = (wind >= 8.0).astype("Int8")
    out["wx_arr_rajada_forte"] = (gust >= 14.0).astype("Int8")
    out["wx_arr_calor"] = (temp >= 30.0).astype("Int8")
    out["wx_arr_frio"] = (temp <= 12.0).astype("Int8")
    out["wx_arr_umidade_alta"] = (rh >= 90.0).astype("Int8")
    out["wx_arr_missing_any"] = (out[[f"{c}_isna" for c in cols]].sum(axis=1).gt(0)).astype("Int8")
    def qstr(s):
        try:
            b = pd.qcut(s, q=[0,.5,.8,.9,1.0], duplicates="drop")
            return b.astype("string")
        except Exception:
            return pd.Series([pd.NA]*len(out), dtype="string")
    out["arr_precip_q"] = qstr(precip)
    out["arr_wind_q"] = qstr(wind)
    out["arr_gust_q"] = qstr(gust)
    out["arr_temp_q"] = qstr(temp)
    out["arr_rh_q"] = qstr(rh)
    return out

def add_distance_km(df: pd.DataFrame, airports: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    if "dist_km" in df.columns:
        if not df["dist_km"].isna().all():
            return df
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0088
        lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
        lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
        lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
        lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))
        dlat = lat2 - lat1; dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return (2 * R * np.arcsin(np.sqrt(a))).astype("float32")
    def _try_load_airports() -> pd.DataFrame | None:
        for p in [Path("data/external/airports_br.parquet"), Path("data/external/airports_br.csv"),
                  Path("data/raw/airports_br.parquet"), Path("data/raw/airports.csv")]:
            if p.exists():
                try:
                    ap = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                except Exception:
                    continue
                icao_cols = [c for c in ap.columns if c.lower() in ("icao","icao_code","icao_4","aeroporto_icao","codigo_oaci","oaci")]
                lat_cols = [c for c in ap.columns if "lat" in c.lower()]
                lon_cols = [c for c in ap.columns if ("lon" in c.lower()) or ("long" in c.lower())]
                if not (icao_cols and lat_cols and lon_cols):
                    continue
                ap = ap.rename(columns={icao_cols[0]:"icao", lat_cols[0]:"lat", lon_cols[0]:"lon"})
                ap["icao"] = ap["icao"].astype(str).str.upper().str.strip()
                return ap[["icao","lat","lon"]]
        return None
    if not {"origem_icao","destino_icao"} <= set(df.columns):
        warnings.warn("add_distance_km: sem origem_icao/destino_icao; pulando.")
        df["dist_km"] = np.nan
        return df
    left = df.copy()
    left["origem_icao"] = left["origem_icao"].astype(str).str.upper().str.strip()
    left["destino_icao"] = left["destino_icao"].astype(str).str.upper().str.strip()
    ap = airports.rename(columns={"icao":"icao","lat":"lat","lon":"lon"})[["icao","lat","lon"]].copy() if airports is not None else _try_load_airports()
    if ap is None or ap.empty:
        warnings.warn("add_distance_km: não encontrei base de aeroportos; 'dist_km' ficará NaN.")
        df["dist_km"] = np.nan
        return df
    m = left.merge(ap.rename(columns={"icao":"origem_icao","lat":"origem_lat","lon":"origem_lon"}), on="origem_icao", how="left")
    m = m.merge(ap.rename(columns={"icao":"destino_icao","lat":"destino_lat","lon":"destino_lon"}), on="destino_icao", how="left")
    mask = m["origem_lat"].notna() & m["origem_lon"].notna() & m["destino_lat"].notna() & m["destino_lon"].notna()
    dist = pd.Series(np.nan, index=m.index, dtype="float32")
    dist.loc[mask] = _haversine(m.loc[mask,"origem_lat"], m.loc[mask,"origem_lon"], m.loc[mask,"destino_lat"], m.loc[mask,"destino_lon"])
    df["dist_km"] = dist.values
    return df

def add_sched_block_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dep = _ensure_series_datetime(out, "partida_prevista")
    arr = _ensure_series_datetime(out, "chegada_prevista") if "chegada_prevista" in out.columns else pd.Series(pd.NaT, index=out.index)
    blk = (arr - dep).dt.total_seconds() / 60.0
    out["sched_block_min"] = blk
    out["sched_block_bucket"] = pd.cut(
        blk, bins=[-np.inf, 60, 120, 180, 300, np.inf], labels=[0,1,2,3,4], right=False
    ).astype("Int8")
    return out

def add_weather_coverage_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dep = [c for c in ["w_dep_precip_mm","w_dep_wind_ms","w_dep_gust_ms","w_dep_temp_c","w_dep_rh"] if c in out.columns]
    arr = [c for c in ["w_arr_precip_mm","w_arr_wind_ms","w_arr_gust_ms","w_arr_temp_c","w_arr_rh"] if c in out.columns]
    if dep:
        cov = out[dep].notna().sum(axis=1)
        out["wx_dep_cov"] = cov.astype("Int8")          # 0..5
        out["wx_dep_all_missing"] = (cov == 0).astype("Int8")
    else:
        out["wx_dep_cov"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
        out["wx_dep_all_missing"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    if arr:
        cova = out[arr].notna().sum(axis=1)
        out["wx_arr_cov"] = cova.astype("Int8")        # 0..5
        out["wx_arr_all_missing"] = (cova == 0).astype("Int8")
    else:
        out["wx_arr_cov"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
        out["wx_arr_all_missing"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    return out

_COMPACT_LABELS = {0: "nada", 1: "fraco", 2: "medio", 3: "forte", 4: "extremo"}

def _sev_from_bins(series, bins):
    v = pd.to_numeric(series, errors="coerce")
    sev = pd.cut(v, bins=[-np.inf] + list(bins) + [np.inf], labels=False)
    sev = pd.Series(sev, index=v.index)
    return sev.astype("Int8")

def _sev_adaptive(series, quantiles=(0.5, 0.8, 0.95, 0.995), fallback_bins=None):
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().sum() >= 50:
        qs = np.nanquantile(v, quantiles)
        qs = np.unique(qs)
        if len(qs) < 4 and fallback_bins is not None:
            return _sev_from_bins(v, fallback_bins)
        while len(qs) < 4:
            qs = np.append(qs, qs[-1] + 1e-6)
        return _sev_from_bins(v, qs[:4])
    fb = fallback_bins if fallback_bins is not None else (0.2, 2, 5, 15)
    return _sev_from_bins(v, fb)

def _sev_precip(precip, mode="rule"):
    bins = (0.2, 2.0, 5.0, 15.0)
    if precip is None:
        return None
    return _sev_from_bins(precip, bins) if mode == "rule" else _sev_adaptive(precip, fallback_bins=bins)

def _sev_wind(wind_ms, mode="rule"):
    bins = (2.0, 5.0, 8.0, 14.0)
    if wind_ms is None:
        return None
    return _sev_from_bins(wind_ms, bins) if mode == "rule" else _sev_adaptive(wind_ms, fallback_bins=bins)

def _sev_gust(gust_ms, mode="rule"):
    bins = (6.0, 10.0, 14.0, 20.0)
    if gust_ms is None:
        return None
    return _sev_from_bins(gust_ms, bins) if mode == "rule" else _sev_adaptive(gust_ms, fallback_bins=bins)

def _sev_rh(rh, mode="rule"):
    bins = (85.0, 90.0, 95.0, 98.0)
    if rh is None:
        return None
    return _sev_from_bins(rh, bins) if mode == "rule" else _sev_adaptive(rh, fallback_bins=bins)

def _sev_temp(temp_c, mode="rule"):
    if temp_c is None:
        return None
    t = pd.to_numeric(temp_c, errors="coerce")
    sev_hot = _sev_from_bins(t, (28.0, 32.0, 35.0, 38.0)).astype("float64")
    cold_bins = [-np.inf, 4.0, 8.0, 12.0, 18.0, np.inf]
    sev_cold = pd.cut(t, bins=cold_bins, labels=[4,3,2,1,0], include_lowest=True)
    sev_cold = pd.Series(sev_cold, index=t.index).astype("float64")
    sev = pd.concat([sev_hot, sev_cold], axis=1).max(axis=1, skipna=True)
    return sev.round(0).astype("Int8")

def _sev_vis_flag(vis_flag):
    if vis_flag is None:
        return None
    v = pd.to_numeric(vis_flag, errors="coerce")
    out = pd.Series(pd.NA, index=v.index, dtype="Int8")
    out[v == 0] = 0
    out[v == 1] = 3
    return out

def _compose_tempo_ruim(sev_parts, weights=None):
    df = pd.concat(sev_parts, axis=1)
    df = df.astype("float64")
    if weights is None:
        weights = np.array([1.0, 0.8, 1.0, 0.3, 0.4, 1.0])[:df.shape[1]]
    else:
        weights = np.asarray(weights)[:df.shape[1]]
    present = df.notna().to_numpy()
    sev = df.to_numpy(dtype="float64")
    w = np.where(present, weights, 0.0)
    num = np.nansum(sev * w, axis=1)
    den = 4.0 * np.sum(w, axis=1)
    norm = np.divide(num, den, out=np.full_like(num, np.nan, dtype="float64"), where=(den > 0))
    bins = np.array([0.0, 0.15, 0.35, 0.60, 0.85, 1.01])
    avg_id = np.digitize(np.nan_to_num(norm, nan=-1.0), bins) - 1
    avg_id = pd.Series(avg_id, index=df.index)
    avg_id = avg_id.mask(np.isnan(norm), other=pd.NA).astype("Int8")
    max_id = df.max(axis=1, skipna=True).round(0).astype("Int8")
    out = pd.Series(np.maximum(avg_id.fillna(-1), max_id.fillna(-1)), index=df.index)
    out = out.mask(out < 0, other=pd.NA).astype("Int8")
    return out

def add_compact_weather(
    df: pd.DataFrame,
    mode: str = "rule",
    include_destination: bool = True,
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    Cria features compactas de clima (0..4 -> nada, fraco, medio, forte, extremo).

    Saídas (partida):
      - chuva_id, chuva
      - tempo_ruim_id, tempo_ruim

    Saídas (chegada, se include_destination=True):
      - chuva_arr_id, chuva_arr
      - tempo_ruim_arr_id, tempo_ruim_arr

    Entrada usa o que houver:
      w_dep_* (precip_mm, wind_ms, gust_ms, temp_c, rh), visibilidade_ruim
      w_arr_* (precip_mm, wind_ms, gust_ms, temp_c, rh)
    """
    out = df.copy()

    p = out.get("w_dep_precip_mm")
    w = out.get("w_dep_wind_ms")
    g = out.get("w_dep_gust_ms")
    t = out.get("w_dep_temp_c")
    rh = out.get("w_dep_rh")
    vis = out.get("visibilidade_ruim")

    chuva_id = _sev_precip(p, mode=mode) if p is not None else None
    if chuva_id is None:
        weak = pd.to_numeric(out.get("wx_chuva", None), errors="coerce") if "wx_chuva" in out.columns else None
        strong = pd.to_numeric(out.get("wx_chuva_forte", None), errors="coerce") if "wx_chuva_forte" in out.columns else None
        if weak is not None or strong is not None:
            chuva_id = pd.Series(0, index=out.index, dtype="Int8")
            if weak is not None:
                chuva_id = chuva_id.mask(weak == 1, other=2)
            if strong is not None:
                chuva_id = chuva_id.mask(strong == 1, other=3)
        else:
            chuva_id = pd.Series(pd.NA, index=out.index, dtype="Int8")

    wind_id = _sev_wind(w, mode=mode)   if w   is not None else None
    gust_id = _sev_gust(g, mode=mode)   if g   is not None else None
    rh_id   = _sev_rh(rh, mode=mode)    if rh  is not None else None
    temp_id = _sev_temp(t, mode=mode)   if t   is not None else None
    vis_id  = _sev_vis_flag(vis)        if vis is not None else None

    parts = [s for s in [chuva_id, wind_id, gust_id, rh_id, temp_id, vis_id] if s is not None]
    tempo_ruim_id = _compose_tempo_ruim(parts, weights=weights) if parts else pd.Series(pd.NA, index=out.index, dtype="Int8")

    out["chuva_id"] = chuva_id
    out["chuva"] = out["chuva_id"].map(_COMPACT_LABELS).astype("string")
    out["tempo_ruim_id"] = tempo_ruim_id
    out["tempo_ruim"] = out["tempo_ruim_id"].map(_COMPACT_LABELS).astype("string")

    if include_destination:
        pa = out.get("w_arr_precip_mm")
        wa = out.get("w_arr_wind_ms")
        ga = out.get("w_arr_gust_ms")
        ta = out.get("w_arr_temp_c")
        rha = out.get("w_arr_rh")
        vis_arr = None
        if any(col in out.columns for col in ["w_arr_rh","w_arr_temp_c","w_arr_wind_ms","w_arr_precip_mm"]):
            rha_n = pd.to_numeric(out.get("w_arr_rh", np.nan), errors="coerce")
            ta_n  = pd.to_numeric(out.get("w_arr_temp_c", np.nan), errors="coerce")
            wa_n  = pd.to_numeric(out.get("w_arr_wind_ms", np.nan), errors="coerce")
            pa_n  = pd.to_numeric(out.get("w_arr_precip_mm", np.nan), errors="coerce")
            vis_arr = (((pa_n >= 5.0) | ((rha_n >= 93.0) & (wa_n < 2.5)) | ((rha_n >= 96.0) & (ta_n < 5.0))).astype("Int8"))
        vis_arr_id = _sev_vis_flag(vis_arr) if vis_arr is not None else None

        chuva_arr_id = _sev_precip(pa, mode=mode) if pa is not None else pd.Series(pd.NA, index=out.index, dtype="Int8")
        wind_arr_id  = _sev_wind(wa, mode=mode)   if wa is not None else None
        gust_arr_id  = _sev_gust(ga, mode=mode)   if ga is not None else None
        rh_arr_id    = _sev_rh(rha, mode=mode)    if rha is not None else None
        temp_arr_id  = _sev_temp(ta, mode=mode)   if ta is not None else None

        parts_arr = [s for s in [chuva_arr_id, wind_arr_id, gust_arr_id, rh_arr_id, temp_arr_id, vis_arr_id] if s is not None]
        tempo_ruim_arr_id = _compose_tempo_ruim(parts_arr, weights=weights) if parts_arr else pd.Series(pd.NA, index=out.index, dtype="Int8")

        out["chuva_arr_id"] = chuva_arr_id
        out["chuva_arr"] = out["chuva_arr_id"].map(_COMPACT_LABELS).astype("string")
        out["tempo_ruim_arr_id"] = tempo_ruim_arr_id
        out["tempo_ruim_arr"] = out["tempo_ruim_arr_id"].map(_COMPACT_LABELS).astype("string")

    return out

def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if is_interval_dtype(s.dtype):
            out[c] = s.astype(str)
            continue
        if is_categorical_dtype(s.dtype):
            try:
                cats = s.cat.categories
                if hasattr(cats, "dtype") and is_interval_dtype(cats.dtype):
                    out[c] = s.astype(str)
            except Exception:
                out[c] = s.astype(str)
    return out
