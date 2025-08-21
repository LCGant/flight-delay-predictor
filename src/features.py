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
    fixed = ["2023-01-01","2023-04-21","2023-05-01","2023-09-07","2023-10-12","2023-11-02","2023-11-15","2023-12-25","2024-01-01","2024-04-21","2024-05-01","2024-09-07","2024-10-12","2024-11-02","2024-11-15","2024-12-25"]
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
    periodo = pd.Series(np.where((0 <= h) & (h <= 5), 0, np.where((6 <= h) & (h <= 11), 1, np.where((12 <= h) & (h <= 17), 2, np.where((18 <= h) & (h <= 23), 3, np.nan)))))
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
    tmp = pd.DataFrame({"orig_idx": out.index, "origem_icao": out["origem_icao"], "partida_prevista": dt, "atraso15": pd.to_numeric(out["atraso15"], errors="coerce")}).sort_values(["origem_icao","partida_prevista","orig_idx"])
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
        for p in [Path("data/external/airports_br.parquet"), Path("data/external/airports_br.csv"), Path("data/raw/airports_br.parquet"), Path("data/raw/airports.csv")]:
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
