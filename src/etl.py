from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass
import json
import zipfile
from io import StringIO
import numpy as np
import pandas as pd
from pandas.api.types import DatetimeTZDtype

from features import (
    add_basic_features,
    add_airport_load_features,
    add_weather_flags,
    add_distance_km,
    add_visibility_feature,
    add_weather_destino,
    add_congestion_features,
    add_airport_size_feature,
    add_congestion_from_size,
)

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
for d in (INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

VRA_GLOB = "VRA_*.csv"
AIRPORTS_FILE_CANDIDATES = [
    "aerodromospublicosv1.csv",
    "aerodromospublicos.csv",
    "airports.csv",
]
AIRLINES_FILE_CANDIDATES = [
    "pda_empresas_aereas_nacionais.csv",
    "empresas_aereas.csv",
    "airlines.csv",
]
ENCODINGS = ("utf-8", "utf-8-sig", "latin1", "cp1252")


def normalize_col(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.strip().lower()
    s = re.sub(r"[^\w\s/|-]+", "", s)
    s = s.replace("/", "_").replace("|", "_").replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s


def try_open_first_lines(path: Path, max_lines: int = 500):
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                lines = []
                for _ in range(max_lines):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line.rstrip("\n"))
            return lines, enc
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="latin1", errors="ignore") as f:
        lines = []
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return lines, "latin1"


def detect_sep(lines):
    cand = [";", ","]
    best_sep, best_count = ";", -1
    for sep in cand:
        max_count = max((ln.count(sep) for ln in lines if ln), default=0)
        if max_count > best_count:
            best_sep, best_count = sep, max_count
    return best_sep


def detect_header_index(lines, sep):
    counts = [ln.count(sep) for ln in lines]
    if not counts:
        return 0
    max_count = max(counts)
    idx = counts.index(max_count)
    return idx


def read_head_smart(path: Path, nrows=50):
    lines, enc = try_open_first_lines(path, max_lines=500)
    sep = detect_sep(lines)
    header_idx = detect_header_index(lines, sep)
    df = pd.read_csv(
        path, sep=sep, header=header_idx, nrows=nrows, encoding=enc, engine="python"
    )
    return df, sep, enc, header_idx


def inspect_file(path: Path) -> dict:
    df, sep, enc, header_idx = read_head_smart(path, nrows=50)
    raw_cols = list(df.columns)
    norm_cols = [normalize_col(c) for c in raw_cols]
    return {
        "file": path.name,
        "sep": sep,
        "encoding": enc,
        "header_index": header_idx,
        "n_preview_rows": len(df),
        "raw_columns": raw_cols,
        "normalized_columns": norm_cols,
    }


def find_first_existing(candidates):
    for name in candidates:
        p_root = RAW_DIR / name
        if p_root.exists():
            return p_root
        for p in RAW_DIR.rglob(name):
            if p.is_file():
                return p
    return None


def get_vra_paths(year: int | None = None):
    patterns = ["VRA_*.csv"] if year is None else [f"VRA_{year}*.csv"]
    found = []
    for pat in patterns:
        found += list((RAW_DIR).glob(pat))
        found += list((RAW_DIR).rglob(pat))
    uniq = sorted({p.resolve() for p in found}, key=lambda p: p.name)
    return uniq


def inspect_all():
    rows = []
    for p in get_vra_paths(year=None):
        rows.append(inspect_file(Path(p)))
    airports = find_first_existing(AIRPORTS_FILE_CANDIDATES)
    if airports:
        rows.append(inspect_file(airports))
    airlines = find_first_existing(AIRLINES_FILE_CANDIDATES)
    if airlines:
        rows.append(inspect_file(airlines))

    if not rows:
        (REPORTS_DIR / "columns_summary.md").write_text(
            "# Resumo de colunas detectadas (smart header)\n\n"
            "Nenhum arquivo encontrado sob data/raw/**.\n"
            "Verifique se os CSVs estão em pastas como:\n"
            "- data/raw/vra_2023/VRA_2023*.csv\n"
            "- data/raw/vra_2024/VRA_2024*.csv\n"
            "- data/raw/aerodromos/aerodromospublicosv1.csv\n"
            "- data/raw/pda_empresas/pda_empresas_aereas_nacionais.csv\n",
            encoding="utf-8",
        )
        print("⚠ Nenhum arquivo encontrado para inspeção. Confirme os caminhos em data/raw/.")
        return

    md_lines = ["# Resumo de colunas detectadas (smart header)\n"]
    for r in rows:
        md_lines.append(f"## {r['file']}")
        md_lines.append(
            f"- separador: `{r['sep']}` | encoding: `{r['encoding']}` | header_index: {r['header_index']} | preview: {r['n_preview_rows']} linhas"
        )
        md_lines.append(f"- **Colunas (originais)**: {', '.join(map(str, r['raw_columns']))}")
        md_lines.append(f"- **Colunas (normalizadas)**: {', '.join(r['normalized_columns'])}\n")
    (REPORTS_DIR / "columns_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    all_norm_cols = sorted({c for r in rows for c in r["normalized_columns"]})
    table = []
    for r in rows:
        presence = {c: (1 if c in r["normalized_columns"] else 0) for c in all_norm_cols}
        presence.update(
            file=r["file"], sep=r["sep"], encoding=r["encoding"], header_index=r["header_index"]
        )
        table.append(presence)
    if table:
        presence_df = pd.DataFrame(table)
        if "file" in presence_df.columns:
            presence_df = presence_df.set_index("file")
        presence_df.to_csv(REPORTS_DIR / "columns_presence_matrix.csv", encoding="utf-8")
    else:
        pd.DataFrame(columns=["file"]).to_csv(
            REPORTS_DIR / "columns_presence_matrix.csv", index=False, encoding="utf-8"
        )

    expected_keys = [
        "icao_aerodromo_origem",
        "icao_aerodromo_destino",
        "iata_aerodromo_origem",
        "iata_aerodromo_destino",
        "sigla_empresa",
        "icao_empresa",
        "iata_empresa",
        "numero_voo",
        "data_partida_prevista",
        "hora_partida_prevista",
        "data_partida_real",
        "hora_partida_real",
        "data_chegada_prevista",
        "hora_chegada_prevista",
        "data_chegada_real",
        "hora_chegada_real",
        "situacao",
        "justificativa",
        "etapa",
    ]
    (REPORTS_DIR / "expected_keys.txt").write_text("\n".join(expected_keys), encoding="utf-8")
    print("✔ Inspeção concluída (smart header).")
    print("  - reports/columns_summary.md")
    print("  - reports/columns_presence_matrix.csv")
    print("  - reports/expected_keys.txt")


REN_VRA = {
    "icao_empresa_aerea": "icao_empresa",
    "numero_voo": "numero_voo",
    "icao_aerodromo_origem": "origem_icao",
    "icao_aerodromo_destino": "destino_icao",
    "partida_prevista": "partida_prevista",
    "partida_real": "partida_real",
    "chegada_prevista": "chegada_prevista",
    "chegada_real": "chegada_real",
    "situacao_voo": "situacao",
    "codigo_justificativa": "justificativa",
    "codigo_autorizacao_di": "di",
    "codigo_tipo_linha": "tipo_linha",
}


def _strip_tz_to_naive(s: pd.Series) -> pd.Series:
    if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
        try:
            return s.dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None)
        except Exception:
            return s.dt.tz_convert(None)
    return s


def _parse_dt_flex_from_str(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"T": " "}, regex=False)
    s = s.str.replace(r"\s*UTC\s*$", "", regex=True, case=False)
    s = s.str.replace(r"Z$", "", regex=True)
    s = s.str.replace(r"([+\-]\d{2}:?\d{2})$", "", regex=True)
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in fmts:
        m = out.isna()
        if m.any():
            try:
                parsed = pd.to_datetime(s[m], format=fmt, errors="coerce")
                out.loc[m] = parsed
            except Exception:
                pass
    m = out.isna()
    if m.any():
        try:
            out.loc[m] = pd.to_datetime(s[m], errors="coerce", dayfirst=True, utc=False)
        except Exception:
            pass
    return _strip_tz_to_naive(out)


def _combine_date_time_cols(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    if date_col in df.columns and time_col in df.columns:
        combo = (
            df[date_col].astype(str).str.strip()
            + " "
            + df[time_col].astype(str).str.strip()
        ).str.strip()
        return _parse_dt_flex_from_str(combo)
    return pd.Series(pd.NaT, index=df.index)


def _to_brt_naive_flex(df: pd.DataFrame, combined_col: str, date_col: str, time_col: str) -> pd.Series:
    if combined_col in df.columns:
        return _parse_dt_flex_from_str(df[combined_col])
    return _combine_date_time_cols(df, date_col, time_col)


def read_vra_month(path: Path) -> pd.DataFrame:
    df, sep, enc, header_idx = read_head_smart(path, nrows=None)
    df.columns = [normalize_col(c) for c in df.columns]
    df = df.rename(columns=REN_VRA)

    keep = list(REN_VRA.values())
    df = df[[c for c in keep if c in df.columns]].copy()

    df["partida_prevista"] = _to_brt_naive_flex(df, "partida_prevista", "data_partida_prevista", "hora_partida_prevista")
    df["partida_real"]     = _to_brt_naive_flex(df, "partida_real",     "data_partida_real",     "hora_partida_real")
    df["chegada_prevista"] = _to_brt_naive_flex(df, "chegada_prevista", "data_chegada_prevista", "hora_chegada_prevista")
    df["chegada_real"]     = _to_brt_naive_flex(df, "chegada_real",     "data_chegada_real",     "hora_chegada_real")

    if "partida_prevista" in df.columns:
        df = df[pd.to_datetime(df["partida_prevista"], errors="coerce").dt.year <= 2024].copy()

    if "situacao" in df.columns:
        df["situacao"] = (
            df["situacao"]
            .astype(str)
            .str.upper()
            .str.normalize("NFKD")
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )

    if {"chegada_real", "chegada_prevista"} <= set(df.columns):
        df["delay_arrival_min"] = (df["chegada_real"] - df["chegada_prevista"]).dt.total_seconds() / 60.0
    else:
        df["delay_arrival_min"] = np.nan

    if {"partida_real", "partida_prevista"} <= set(df.columns):
        df["delay_departure_min"] = (df["partida_real"] - df["partida_prevista"]).dt.total_seconds() / 60.0
    else:
        df["delay_departure_min"] = np.nan

    df["atraso15"] = (df["delay_arrival_min"] > 15).astype("Int8")
    if "situacao" in df.columns:
        df.loc[~df["situacao"].str.contains("REALIZADO", na=False), "atraso15"] = pd.NA

    if "partida_prevista" in df.columns:
        dt = pd.to_datetime(df["partida_prevista"], errors="coerce")
        df["mes"] = dt.dt.month
        df["dia_semana"] = dt.dt.dayofweek
        df["hora_partida_prevista"] = dt.dt.hour
    else:
        df["mes"] = df["dia_semana"] = df["hora_partida_prevista"] = pd.NA

    if {"origem_icao", "destino_icao"} <= set(df.columns):
        df["rota"] = (
            df["origem_icao"].astype(str).str.upper().str.strip()
            + ">"
            + df["destino_icao"].astype(str).str.upper().str.strip()
        )
    return df


def unify_vra_2024():
    paths = get_vra_paths(year=2024)
    if not paths:
        raise SystemExit("Nenhum VRA_2024*.csv encontrado em data/raw/")
    dfs = [read_vra_month(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    df.to_parquet(INTERIM_DIR / "vra_2024.parquet", index=False)

    cols_model = [
        "atraso15",
        "delay_arrival_min",
        "delay_departure_min",
        "icao_empresa",
        "origem_icao",
        "destino_icao",
        "rota",
        "mes",
        "dia_semana",
        "hora_partida_prevista",
        "partida_prevista",
        "chegada_prevista",
        "partida_real",
        "chegada_real",
        "situacao",
        "justificativa",
        "numero_voo",
        "di",
        "tipo_linha",
    ]
    cols_model = [c for c in cols_model if c in df.columns]
    df[cols_model].to_parquet(PROCESSED_DIR / "dataset_modelagem.parquet", index=False)

    print("✔ Unificado:")
    print(f"  - {len(df):,} linhas")
    print("  - data/interim/vra_2024.parquet")
    print("  - data/processed/dataset_modelagem.parquet")


def unify_vra_all():
    paths = get_vra_paths(year=None)
    if not paths:
        raise SystemExit("Nenhum VRA_*.csv encontrado em data/raw/**")
    dfs = [read_vra_month(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    if "partida_prevista" in df.columns:
        ts = pd.to_datetime(df["partida_prevista"], errors="coerce")
        df = df.loc[ts.dt.year <= 2024].copy()

    df.to_parquet(INTERIM_DIR / "vra_all.parquet", index=False)

    cols_model = [
        "atraso15",
        "delay_arrival_min",
        "delay_departure_min",
        "icao_empresa",
        "origem_icao",
        "destino_icao",
        "rota",
        "mes",
        "dia_semana",
        "hora_partida_prevista",
        "partida_prevista",
        "chegada_prevista",
        "partida_real",
        "chegada_real",
        "situacao",
        "justificativa",
        "numero_voo",
        "di",
        "tipo_linha",
    ]
    cols_model = [c for c in cols_model if c in df.columns]
    df[cols_model].to_parquet(PROCESSED_DIR / "dataset_modelagem.parquet", index=False)

    try:
        ts = pd.to_datetime(df["partida_prevista"], errors="coerce")
        by_year = ts.dt.year.value_counts(dropna=False).sort_index()
        by_year_month = ts.dt.to_period("M").value_counts().sort_index()
        print("✔ Unificado TODOS os anos. Linhas por ano:\n", by_year.to_string())
        print("\nLinhas por ano/mês:\n", by_year_month.to_string())
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        by_year.to_csv(REPORTS_DIR / "rows_by_year.csv", header=["count"])
        by_year_month.index = by_year_month.index.astype(str)
        by_year_month.to_csv(REPORTS_DIR / "rows_by_year_month.csv", header=["count"])
    except Exception as e:
        print("✔ Unificado TODOS os anos. (Resumo ano/mês indisponível)", e)

    print("  - data/interim/vra_all.parquet")
    print("  - data/processed/dataset_modelagem.parquet")
    print("  - reports/rows_by_year.csv")
    print("  - reports/rows_by_year_month.csv")


def profile_vra():
    p = INTERIM_DIR / "vra_2024.parquet"
    if not p.exists():
        unify_vra_2024()
    df = pd.read_parquet(p)

    def df_to_md(dfx):
        try:
            return dfx.to_markdown()
        except Exception:
            return "```\n" + dfx.to_string() + "\n```"

    keys = [
        "partida_prevista",
        "chegada_prevista",
        "partida_real",
        "chegada_real",
        "origem_icao",
        "destino_icao",
        "icao_empresa",
        "atraso15",
    ]
    keys_exist = [k for k in keys if k in df.columns]
    miss = df[keys_exist].isna().mean().sort_values(ascending=False).to_frame("pct_na")
    situ = (
        df["situacao"].value_counts(dropna=False).to_frame("contagem")
        if "situacao" in df.columns
        else pd.DataFrame({"contagem": []})
    )
    time_cols = [c for c in ["partida_prevista", "partida_real", "chegada_prevista", "chegada_real"] if c in df.columns]
    head_times = df[time_cols].dropna().head(5) if time_cols else pd.DataFrame()
    total = len(df)
    label_notna = df["atraso15"].notna().sum() if "atraso15" in df.columns else 0
    pos_rate = (
        df.loc[df["atraso15"].notna(), "atraso15"].astype("float").mean()
        if "atraso15" in df.columns and df["atraso15"].notna().any()
        else float("nan")
    )
    realized = df["situacao"].str.contains("REALIZADO", na=False).sum() if "situacao" in df.columns else 0

    md = []
    md.append("# Data Quality Report\n")
    md.append(f"- Linhas totais: **{total:,}**")
    md.append(
        f"- Linhas com label (`atraso15`) não nula: **{label_notna:,}**"
    )
    md.append(
        f"- Taxa de atraso (>15 min) nas linhas com label: **{pos_rate:.2%}**"
        if pd.notna(pos_rate)
        else "- Taxa de atraso: n/d"
    )
    md.append(f"- Voos com `situacao` contendo 'REALIZADO': **{realized:,}**\n")
    md.append("## Faltantes nas colunas-chave")
    md.append(df_to_md(miss))
    md.append("\n## Situação do voo (contagem)")
    md.append(df_to_md(situ))
    md.append("\n## Amostra de datetimes")
    md.append(df_to_md(head_times))
    (REPORTS_DIR / "data_quality.md").write_text("\n\n".join(md), encoding="utf-8")
    print("✔ Relatório de qualidade salvo em reports/data_quality.md")


WEATHER_TOLERANCE_MIN = 60
WEATHER_TIME_SHIFT_HOURS = -3


def _extract_inmet_metadata(lines):
    meta = {}
    for ln in lines[:60]:
        parts = ln.split(";")
        if len(parts) >= 2:
            key = normalize_col(parts[0].replace(":", ""))
            val = parts[1].strip()
            meta[key] = val

    def _to_float_pt(v):
        try:
            return float(str(v).replace(",", "."))
        except Exception:
            return np.nan

    return {
        "lat": _to_float_pt(meta.get("latitude", "")),
        "lon": _to_float_pt(meta.get("longitude", "")),
        "estacao": meta.get("estacao", ""),
        "uf": meta.get("uf", ""),
        "regiao": meta.get("regiao", ""),
    }


def _parse_inmet_text(raw_text: str) -> pd.DataFrame:
    lines = raw_text.splitlines()
    meta = _extract_inmet_metadata(lines)
    header_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("data;hora utc") or low.startswith("data;hora"):
            header_idx = i
            break
    if header_idx is None:
        sep_counts = [ln.count(";") for ln in lines]
        header_idx = int(np.argmax(sep_counts)) if sep_counts else 0
    buf = StringIO("\n".join(lines))
    df = pd.read_csv(
        buf,
        sep=";",
        header=header_idx,
        engine="python",
        decimal=",",
        na_values=["", "NA", "nan", "NN", "///", "////", " ", "-"],
    )
    df.columns = [normalize_col(c) for c in df.columns]
    col_data = next((c for c in df.columns if c.startswith("data")), None)
    col_hora = next((c for c in df.columns if "hora" in c), None)
    if col_data is None or col_hora is None:
        raise ValueError("Não encontrei colunas 'Data' e 'Hora'")

    hora = (
        df[col_hora]
        .astype(str)
        .str.replace("UTC", "", case=False, regex=False)
        .str.strip()
        .str.zfill(4)
        .str.replace(r"^(\d{2})(\d{2})$", r"\1:\2", regex=True)
    )
    dt_utc = pd.to_datetime(
        df[col_data].astype(str).str.strip() + " " + hora,
        errors="coerce",
        format="%Y/%m/%d %H:%M",
    )
    dt_local = dt_utc + pd.to_timedelta(WEATHER_TIME_SHIFT_HOURS, unit="h")

    out = pd.DataFrame({"wx_dt": dt_local})
    out["lat"] = meta["lat"]
    out["lon"] = meta["lon"]
    out["station_name"] = meta["estacao"]
    out["uf"] = meta["uf"]
    out["regiao"] = meta["regiao"]

    col_prec = next((c for c in df.columns if "precip" in c), None)
    if col_prec:
        out["precip_mm"] = pd.to_numeric(df[col_prec], errors="coerce")
    col_temp = next((c for c in df.columns if "temperatura" in c and "hora" in c), None)
    if col_temp:
        out["temp_c"] = pd.to_numeric(df[col_temp], errors="coerce")
    col_rh = next((c for c in df.columns if "umidade" in c and "rel" in c), None)
    if col_rh:
        out["rh"] = pd.to_numeric(df[col_rh], errors="coerce")
    col_wspd = next((c for c in df.columns if "vento" in c and "velocidade" in c), None)
    if col_wspd:
        out["wind_ms"] = pd.to_numeric(df[col_wspd], errors="coerce")
    col_gust = next((c for c in df.columns if "rajada" in c), None)
    if col_gust:
        out["gust_ms"] = pd.to_numeric(df[col_gust], errors="coerce")

    out = out[out["wx_dt"].notna()].copy()
    return out


def _read_inmet_zip_entry(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    raw_bytes = zf.read(name)
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin1", errors="ignore")
    return _parse_inmet_text(raw_text)


def _read_inmet_csv_path(path: Path) -> pd.DataFrame:
    raw_bytes = path.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin1", errors="ignore")
    return _parse_inmet_text(raw_text)


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def _load_airports_latlon() -> pd.DataFrame:
    cand = find_first_existing(AIRPORTS_FILE_CANDIDATES)
    if not cand:
        raise SystemExit(
            "Arquivo de aeródromos não encontrado em data/raw/. Coloque um CSV da ANAC com código OACI/ICAO e latitude/longitude."
        )
    raw_bytes = Path(cand).read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin1", errors="ignore")
    lines = raw_text.splitlines()

    def _norm(s):
        return normalize_col(s)

    header_idx = None
    for i, ln in enumerate(lines):
        low = _norm(ln)
        if ("oaci" in low or "icao" in low) and ("latitude" in low) and ("longitude" in low):
            header_idx = i
            break
    if header_idx is None:
        sep_counts = [ln.count(";") for ln in lines]
        idx_max = int(np.argmax(sep_counts)) if sep_counts else 0
        low = _norm(lines[idx_max]) if lines else ""
        header_idx = idx_max if ("oaci" in low or "icao" in low) else 0

    buf = StringIO("\n".join(lines))
    df = pd.read_csv(
        buf,
        sep=";",
        header=header_idx,
        engine="python",
        decimal=",",
        na_values=["", "NA", "nan", "///", "////", "-", " "],
    )
    df.columns = [normalize_col(c) for c in df.columns]
    print("DEBUG aeródromos (header OK) – colunas:", ", ".join(df.columns))

    icao_cols = [c for c in df.columns if ("oaci" in c or "icao" in c)]
    if not icao_cols:
        raise SystemExit("Ainda não encontrei coluna OACI/ICAO após detectar o header. Verifique o arquivo ANAC.")
    col_icao = icao_cols[0]

    lat_candidates = [c for c in df.columns if "lat" in c]
    lon_candidates = [c for c in df.columns if ("lon" in c or "long" in c)]
    if not lat_candidates or not lon_candidates:
        raise SystemExit("Não encontrei colunas de latitude/longitude no arquivo de aeródromos.")

    def _pick_best(cands):
        prio = [c for c in cands if ("decimal" in c or re.search(r"\bdec", c))]
        return prio[0] if prio else cands[0]

    col_lat = _pick_best(lat_candidates)
    col_lon = _pick_best(lon_candidates)

    hemi_re = re.compile(r"([NnSsEeWw])")
    nums_re = re.compile(r"[-+]?\d+(?:[\,\.]\d+)?")

    def _parse_latlon(s):
        if pd.isna(s):
            return np.nan
        t = str(s).strip()
        if not t:
            return np.nan
        t = t.replace("_", " ").replace("º", "°").replace("o", "°").replace(",", ".")
        hemi_m = hemi_re.search(t)
        hemi = hemi_m.group(1).upper() if hemi_m else None
        nums = [x.group(0) for x in nums_re.finditer(t)]
        nums = [float(x) for x in nums]
        if not nums:
            return np.nan
        if len(nums) == 1:
            val = float(nums[0])
        else:
            deg = float(nums[0])
            minutes = float(nums[1]) if len(nums) >= 2 else 0.0
            seconds = float(nums[2]) if len(nums) >= 3 else 0.0
            val = abs(deg) + minutes / 60.0 + seconds / 3600.0
            if deg < 0:
                val = -val
        if hemi in {"S", "W"}:
            val = -abs(val)
        elif hemi in {"N", "E"}:
            val = abs(val)
        return val

    lat = df[col_lat].map(_parse_latlon)
    lon = df[col_lon].map(_parse_latlon)

    out = (
        pd.DataFrame(
            {
                "icao": df[col_icao].astype(str).str.upper().str.strip(),
                "lat": pd.to_numeric(lat, errors="coerce"),
                "lon": pd.to_numeric(lon, errors="coerce"),
            }
        )
        .dropna(subset=["icao", "lat", "lon"])
        .drop_duplicates(subset=["icao"])
    )
    if out.empty:
        print(
            "DEBUG exemplo LAT/LON brutos:",
            df[col_lat].astype(str).head(5).tolist(),
            df[col_lon].astype(str).head(5).tolist(),
        )
        raise SystemExit("Li o arquivo ANAC, mas não consegui extrair ICAO+lat/lon válidos (tudo NaN).")
    return out


def ingest_weather():
    wz_dir = RAW_DIR / "weather"
    frames = []
    for path in sorted(wz_dir.rglob("*.csv")):
        try:
            frames.append(_read_inmet_csv_path(path))
        except Exception as e:
            print(f"Aviso: pulando {path.name} ({e})")
    for zpath in sorted(wz_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                for name in zf.namelist():
                    if name.lower().endswith((".csv", ".txt")):
                        try:
                            frames.append(_read_inmet_zip_entry(zf, name))
                        except Exception as e:
                            print(f"Aviso: pulando {name} ({e})")
        except Exception as e:
            print(f"Aviso: não foi possível abrir {zpath.name} ({e})")

    if not frames:
        raise SystemExit("Nenhum CSV válido de clima encontrado (verifique data/raw/weather/2023 e 2024).")

    wx = pd.concat(frames, ignore_index=True)
    wx = wx.dropna(subset=["wx_dt", "lat", "lon"]).copy()

    airports = _load_airports_latlon()
    stations = wx[["lat", "lon", "station_name"]].drop_duplicates().reset_index(drop=True)

    st_lat = stations["lat"].to_numpy()[:, None]
    st_lon = stations["lon"].to_numpy()[:, None]
    ap_lat = airports["lat"].to_numpy()[None, :]
    ap_lon = airports["lon"].to_numpy()[None, :]
    dist = _haversine_km(st_lat, st_lon, ap_lat, ap_lon)
    idx_min = np.argmin(dist, axis=1)
    min_dist = dist[np.arange(dist.shape[0]), idx_min]
    stations["icao"] = airports["icao"].to_numpy()[idx_min]
    stations["icao_dist_km"] = min_dist
    stations.loc[stations["icao_dist_km"] > 60, "icao"] = np.nan

    wx = wx.merge(stations, on=["lat", "lon", "station_name"], how="left")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    wx.to_parquet(INTERIM_DIR / "weather_2023_2024.parquet", index=False)
    print(f"✔ Clima ingerido: {len(wx):,} linhas -> data/interim/weather_2023_2024.parquet")
    miss_icao = wx["icao"].isna().mean()
    print(f"  Proporção sem ICAO (dist > 60km): {miss_icao:.2%}")


def _merge_asof_by_airport(flights: pd.DataFrame, wx: pd.DataFrame, ap_col: str, time_col: str, prefix: str):
    cols_keep = ["icao", "wx_dt", "precip_mm", "wind_ms", "gust_ms", "temp_c", "rh"]
    cols_keep = [c for c in cols_keep if c in wx.columns]
    wx_small = wx[cols_keep].dropna(subset=["wx_dt"]).copy()
    out = flights.copy()

    if time_col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[time_col]):
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(wx_small["wx_dt"]):
        wx_small["wx_dt"] = pd.to_datetime(wx_small["wx_dt"], errors="coerce")

    tol = pd.Timedelta(minutes=WEATHER_TOLERANCE_MIN)

    if ap_col not in out.columns:
        for c in ["precip_mm", "wind_ms", "gust_ms", "temp_c", "rh"]:
            newc = f"{prefix}{c}"
            if newc not in out.columns:
                out[newc] = np.nan
        return out

    m_has_ap = out[ap_col].notna()
    out_no_ap = out[~m_has_ap]
    work = out[m_has_ap].copy()
    work[ap_col] = work[ap_col].astype(str).str.upper().str.strip()
    airports = work[ap_col].unique().tolist()
    wx_air = wx_small[wx_small["icao"].isin(airports)].sort_values(["icao", "wx_dt"])

    if wx_air.empty:
        out = pd.concat([work, out_no_ap], ignore_index=True, sort=False)
        for c in ["precip_mm", "wind_ms", "gust_ms", "temp_c", "rh"]:
            newc = f"{prefix}{c}"
            if newc not in out.columns:
                out[newc] = np.nan
        return out

    merged_parts = []
    for ac in airports:
        fpart_all = work[work[ap_col] == ac]
        if time_col in fpart_all.columns:
            fpart = fpart_all[fpart_all[time_col].notna()].sort_values(time_col)
            fpart_notime = fpart_all[fpart_all[time_col].isna()]
        else:
            fpart = pd.DataFrame(columns=fpart_all.columns)
            fpart_notime = fpart_all

        wpart = wx_air[wx_air["icao"] == ac]
        if wpart.empty or fpart.empty:
            merged_parts.append(fpart_all)
            continue

        merged = pd.merge_asof(
            fpart, wpart, left_on=time_col, right_on="wx_dt", direction="nearest", tolerance=tol
        )
        if not fpart_notime.empty:
            merged = pd.concat([merged, fpart_notime], ignore_index=True, sort=False)
        merged_parts.append(merged)

    work_merged = pd.concat(merged_parts, ignore_index=True, sort=False)
    out = pd.concat([work_merged, out_no_ap], ignore_index=True, sort=False)

    rename = {}
    for c in ["precip_mm", "wind_ms", "gust_ms", "temp_c", "rh"]:
        if c in out.columns:
            rename[c] = f"{prefix}{c}"
    out = out.rename(columns=rename)
    if "wx_dt" in out.columns:
        out = out.drop(columns=["wx_dt"])
    for c in ["precip_mm", "wind_ms", "gust_ms", "temp_c", "rh"]:
        newc = f"{prefix}{c}"
        if newc not in out.columns:
            out[newc] = np.nan
    return out


def join_weather_to_dataset(include_destination=False):
    base_p = PROCESSED_DIR / "dataset_modelagem.parquet"
    if not base_p.exists():
        raise SystemExit("Execute primeiro: python src/etl.py unify_all (ou unify)")

    df = pd.read_parquet(base_p)

    wx_p = INTERIM_DIR / "weather_2023_2024.parquet"
    if not wx_p.exists():
        ingest_weather()
    wx = pd.read_parquet(wx_p)

    for c in ("origem_icao", "destino_icao"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()

    for tcol in ["partida_prevista", "chegada_prevista"]:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce", dayfirst=True)
            if isinstance(df[tcol].dtype, DatetimeTZDtype):
                df[tcol] = df[tcol].dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None)

    if {"origem_icao", "partida_prevista"} <= set(df.columns):
        df = _merge_asof_by_airport(df, wx, ap_col="origem_icao", time_col="partida_prevista", prefix="w_dep_")
    else:
        print("Aviso: não encontrei origem_icao/partida_prevista para join de clima na origem.")

    if include_destination and {"destino_icao", "chegada_prevista"} <= set(df.columns):
        df = _merge_asof_by_airport(df, wx, ap_col="destino_icao", time_col="chegada_prevista", prefix="w_arr_")

    df.to_parquet(PROCESSED_DIR / "dataset_modelagem_weather.parquet", index=False)
    print("✔ Clima associado ao dataset de modelagem:")
    print("  - data/processed/dataset_modelagem_weather.parquet")


def weather_pipeline():
    ingest_weather()
    join_weather_to_dataset(include_destination=True)
    print("✔ Weather pipeline concluído.")


def make_trainset():
    from features import (
        add_basic_features,
        add_history_features,
        add_airport_load_features,
        add_weather_flags,
        add_visibility_feature,
        add_weather_destino,
        add_congestion_features,
        add_airport_size_feature,
        add_congestion_from_size,
    )

    p_weather = PROCESSED_DIR / "dataset_modelagem_weather.parquet"
    p_base = PROCESSED_DIR / "dataset_modelagem.parquet"
    p = p_weather if p_weather.exists() else p_base
    if not p.exists():
        unify_vra_2024()
        p = PROCESSED_DIR / "dataset_modelagem.parquet"

    df = pd.read_parquet(p)
    if "situacao" not in df.columns:
        raise SystemExit("Coluna ausente: situacao")

    m_realizado = df["situacao"].str.contains("REALIZADO", na=False)
    for c in ("chegada_prevista", "chegada_real"):
        if c not in df.columns:
            raise SystemExit(f"Coluna ausente: {c}")

    base = df.loc[m_realizado & df["chegada_prevista"].notna() & df["chegada_real"].notna()].copy()

    base = add_basic_features(base)
    base = add_history_features(base)
    base = add_airport_load_features(base)
    base = add_airport_size_feature(base)
    base = add_congestion_from_size(base)
    base = add_weather_flags(base)
    base = add_visibility_feature(base)
    base = add_weather_destino(base)
    base = add_congestion_features(base)

    if "delay_arrival_min" in base.columns and base["delay_arrival_min"].notna().any():
        q1, q99 = base["delay_arrival_min"].quantile([0.01, 0.99])
        base = base[(base["delay_arrival_min"] >= q1) & (base["delay_arrival_min"] <= q99)]

    feat_cols = [
        "icao_empresa",
        "origem_icao",
        "destino_icao",
        "rota",
        "mes",
        "dia_semana",
        "hora_partida_prevista",
        "hora_bloco_30",
        "hora_sin",
        "hora_cos",
        "periodo_dia_id",
        "is_weekend",
        "is_feriado",
        "is_vespera_feriado",
        "em_ferias",
        "w_dep_precip_mm",
        "w_dep_wind_ms",
        "w_dep_gust_ms",
        "w_dep_temp_c",
        "w_dep_rh",
        "wx_chuva",
        "wx_vento_forte",
        "wx_rajada_forte",
        "wx_calor",
        "wx_frio",
        "wx_umidade_alta",
        "w_dep_precip_mm_isna",
        "w_dep_wind_ms_isna",
        "w_dep_gust_ms_isna",
        "w_dep_temp_c_isna",
        "w_dep_rh_isna",
        "wx_missing_any",
        "visibilidade_ruim",
        "w_arr_precip_mm",
        "w_arr_wind_ms",
        "w_arr_gust_ms",
        "w_arr_temp_c",
        "w_arr_rh",
        "w_arr_precip_mm_isna",
        "w_arr_wind_ms_isna",
        "w_arr_gust_ms_isna",
        "w_arr_temp_c_isna",
        "w_arr_rh_isna",
        "wx_arr_chuva",
        "wx_arr_vento_forte",
        "wx_arr_rajada_forte",
        "wx_arr_calor",
        "wx_arr_frio",
        "wx_arr_umidade_alta",
        "load_origem_15",
        "dist_km",
        "hist_atraso_rota_30",
        "hist_std_rota_30",
        "hist_vol_rota_30",
        "hist_atraso_empresa_50",
        "hist_atraso_num_voo_10",
        "atrasos_mesmo_aeroporto_1h_hist",
        "airport_size_id",
        "congestion_ratio",
        "congestion_bucket",
    ]
    feat_cols = [c for c in feat_cols if c in base.columns]
    target_col = "atraso15"
    if target_col not in base.columns:
        raise SystemExit("Coluna alvo 'atraso15' ausente após enriquecimento.")

    train = base[feat_cols + [target_col]].dropna(subset=[target_col]).copy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train_ready.parquet", index=False)

    pos_rate = float(train[target_col].astype("float").mean()) if len(train) else float("nan")
    print("✔ Trainset salvo em data/processed/train_ready.parquet (com clima e features enriquecidas)")
    print(f"  Linhas: {len(train):,} | Positivos (>15min): {pos_rate:.2%}" if len(train) else "  (Sem linhas no trainset)")
    try:
        print("  Features usadas:", ", ".join(feat_cols))
    except Exception:
        pass


SAFE_CATEGORICAL = ["icao_empresa", "origem_icao", "destino_icao", "rota"]
SAFE_NUMERICAL = [
    "mes",
    "dia_semana",
    "hora_partida_prevista",
    "hora_bloco_30",
    "hora_sin",
    "hora_cos",
    "periodo_dia_id",
    "is_weekend",
    "is_feriado",
    "is_vespera_feriado",
    "em_ferias",
    "w_dep_precip_mm",
    "w_dep_wind_ms",
    "w_dep_gust_ms",
    "w_dep_temp_c",
    "w_dep_rh",
    "wx_chuva",
    "wx_vento_forte",
    "wx_rajada_forte",
    "wx_calor",
    "wx_frio",
    "wx_umidade_alta",
    "w_dep_precip_mm_isna",
    "w_dep_wind_ms_isna",
    "w_dep_gust_ms_isna",
    "w_dep_temp_c_isna",
    "w_dep_rh_isna",
    "wx_missing_any",
    "visibilidade_ruim",
    "w_arr_precip_mm",
    "w_arr_wind_ms",
    "w_arr_gust_ms",
    "w_arr_temp_c",
    "w_arr_rh",
    "w_arr_precip_mm_isna",
    "w_arr_wind_ms_isna",
    "w_arr_gust_ms_isna",
    "w_arr_temp_c_isna",
    "w_arr_rh_isna",
    "wx_arr_chuva",
    "wx_arr_vento_forte",
    "wx_arr_rajada_forte",
    "wx_arr_calor",
    "wx_arr_frio",
    "wx_arr_umidade_alta",
    "load_origem_15",
    "dist_km",
    "hist_atraso_rota_30",
    "hist_atraso_empresa_50",
    "hist_atraso_num_voo_10",
    "hist_vol_rota_30",
    "hist_std_rota_30",
    "atrasos_mesmo_aeroporto_1h_hist",
    "airport_size_id",
    "congestion_ratio",
    "congestion_bucket",
]
SAFE_TIME_KEYS = ["partida_prevista"]
TARGET_COL = "atraso15"
LEAKY_COL_PREFIXES = ("delay_",)
LEAKY_COLS_EXACT = {"chegada_prevista", "chegada_real", "partida_real"}


def build_feature_view(df: pd.DataFrame) -> pd.DataFrame:
    cols_keep = [c for c in SAFE_CATEGORICAL + SAFE_NUMERICAL + SAFE_TIME_KEYS + [TARGET_COL] if c in df.columns]
    cols_keep = [c for c in cols_keep if c not in LEAKY_COLS_EXACT]
    cols_keep = [c for c in cols_keep if not any(c.startswith(p) for p in LEAKY_COL_PREFIXES)]
    out = df[cols_keep].copy()
    if "partida_prevista" in out.columns:
        out["partida_prevista"] = pd.to_datetime(out["partida_prevista"], errors="coerce")
    if TARGET_COL in out.columns and out[TARGET_COL].dtype.name != "Int8":
        out[TARGET_COL] = out[TARGET_COL].astype("Int8")
    for c in SAFE_CATEGORICAL:
        if c in out.columns:
            out[c] = out[c].astype(str).str.upper().str.strip()
    return out


SPLITS_DIR = PROCESSED_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SplitConfig:
    train_end: str = "2024-08-31"
    val_end: str = "2024-10-31"


def _temporal_cutoffs(ts: pd.Series, cfg: SplitConfig):
    ts = pd.to_datetime(ts, errors="coerce")
    ts = ts[ts.notna()]
    if ts.empty:
        raise SystemExit("Sem timestamps válidos em partida_prevista para gerar splits.")
    tmin, tmax = ts.min(), ts.max()
    fixed_train_end = pd.to_datetime(cfg.train_end)
    fixed_val_end = pd.to_datetime(cfg.val_end)
    if tmin <= fixed_train_end < fixed_val_end <= tmax:
        return dict(mode="fixed", train_end=str(fixed_train_end.date()), val_end=str(fixed_val_end.date()))
    q70 = ts.quantile(0.70)
    q85 = ts.quantile(0.85)
    return dict(mode="quantiles", train_end=str(q70.date()), val_end=str(q85.date()))


def _make_hist_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["partida_prevista"] = pd.to_datetime(work["partida_prevista"], errors="coerce")
    work = work.sort_values("partida_prevista").reset_index(drop=True)
    if "atraso15" not in work.columns:
        raise SystemExit("Para criar features históricas, 'atraso15' precisa existir.")

    y = work["atraso15"].astype("float")
    if {"origem_icao", "destino_icao"} <= set(work.columns):
        rota = (
            work["origem_icao"].astype(str).str.upper().str.strip()
            + ">"
            + work["destino_icao"].astype(str).str.upper().str.strip()
        )
    else:
        rota = pd.Series([""] * len(work))
    cia = work["icao_empresa"].astype(str).str.upper().str.strip() if "icao_empresa" in work.columns else pd.Series([""] * len(work))
    num_voo = work["numero_voo"].astype(str).str.upper().str.strip() if "numero_voo" in work.columns else pd.Series([""] * len(work))

    work["__rota_key"] = rota
    work["__cia_key"] = cia
    work["__nvoo_key"] = num_voo

    def grp_roll_mean(key, window, minp):
        return y.groupby(key).apply(lambda s: s.shift().rolling(window, min_periods=minp).mean()).reset_index(level=0, drop=True)

    def grp_roll_std(key, window, minp):
        return y.groupby(key).apply(lambda s: s.shift().rolling(window, min_periods=minp).std()).reset_index(level=0, drop=True)

    def grp_roll_count(key, window, minp):
        return y.groupby(key).apply(lambda s: s.shift().rolling(window, min_periods=minp).count()).reset_index(level=0, drop=True)

    work["hist_atraso_rota_30"] = grp_roll_mean(work["__rota_key"], window=30, minp=5)
    work["hist_atraso_empresa_50"] = grp_roll_mean(work["__cia_key"], window=50, minp=10)
    work["hist_atraso_num_voo_10"] = grp_roll_mean(work["__nvoo_key"], window=10, minp=3)
    work["hist_vol_rota_30"] = grp_roll_count(work["__rota_key"], window=30, minp=5)
    work["hist_std_rota_30"] = grp_roll_std(work["__rota_key"], window=30, minp=5)

    work = work.drop(columns=["__rota_key", "__cia_key", "__nvoo_key"])

    for c in ["hist_atraso_rota_30", "hist_atraso_empresa_50", "hist_atraso_num_voo_10", "hist_std_rota_30"]:
        if c in work.columns:
            work[c] = work[c].astype("float")

    mean_global = float(y.mean()) if len(work) else 0.0
    for c in ["hist_atraso_rota_30", "hist_atraso_empresa_50", "hist_atraso_num_voo_10"]:
        if c in work.columns:
            work[c] = work[c].fillna(mean_global)
    if "hist_vol_rota_30" in work.columns:
        work["hist_vol_rota_30"] = work["hist_vol_rota_30"].fillna(0.0)
    if "hist_std_rota_30" in work.columns:
        work["hist_std_rota_30"] = work["hist_std_rota_30"].fillna(0.0)

    return work


def add_historical_features():
    p_weather = PROCESSED_DIR / "dataset_modelagem_weather.parquet"
    p_base = PROCESSED_DIR / "dataset_modelagem.parquet"
    src = p_weather if p_weather.exists() else p_base
    if not src.exists():
        raise SystemExit("Base não encontrada. Rode: unify_all (e weather opcional).")

    df = pd.read_parquet(src)
    if "situacao" not in df.columns:
        raise SystemExit("Coluna 'situacao' ausente.")
    base = df[df["situacao"].str.contains("REALIZADO", na=False)].copy()
    if "atraso15" not in base.columns:
        raise SystemExit("Coluna 'atraso15' ausente.")
    base = base[base["atraso15"].notna()].copy()
    base["partida_prevista"] = pd.to_datetime(base["partida_prevista"], errors="coerce")
    base = base.sort_values("partida_prevista").reset_index(drop=True)

    out = _make_hist_features(base)
    out = add_basic_features(out)
    out = add_airport_load_features(out)
    out = add_airport_size_feature(out)
    out = add_congestion_from_size(out)
    out = add_weather_flags(out)
    out = add_visibility_feature(out)
    out = add_weather_destino(out)
    out = add_congestion_features(out)

    try:
        ap = _load_airports_latlon()
        out = add_distance_km(out, ap)
    except Exception as e:
        print("Aviso: não foi possível calcular dist_km:", e)

    out.to_parquet(PROCESSED_DIR / "dataset_modelagem_hist.parquet", index=False)
    print("✔ Features históricas salvas em data/processed/dataset_modelagem_hist.parquet (com dist_km, visibilidade, clima destino e congestion_hist)")


def _temporal_splits_base():
    p_hist = PROCESSED_DIR / "dataset_modelagem_hist.parquet"
    p_weather = PROCESSED_DIR / "dataset_modelagem_weather.parquet"
    p_base = PROCESSED_DIR / "dataset_modelagem.parquet"
    if p_hist.exists():
        return p_hist
    elif p_weather.exists():
        return p_weather
    else:
        return p_base


def make_temporal_splits():
    p = _temporal_splits_base()
    if not p.exists():
        raise SystemExit("Execute antes: unify_all (ou unify) e weather (opcional) / features (opcional).")

    df_raw = pd.read_parquet(p)
    if "situacao" in df_raw.columns:
        base = df_raw[df_raw["situacao"].str.contains("REALIZADO", na=False)].copy()
    else:
        base = df_raw.copy()

    if TARGET_COL not in base.columns:
        raise SystemExit(f"Coluna alvo '{TARGET_COL}' ausente.")
    base = base[base[TARGET_COL].notna()].copy()
    if base.empty:
        raise SystemExit("Sem linhas com label (atraso15).")

    fv = build_feature_view(base)
    if "partida_prevista" not in fv.columns:
        raise SystemExit("Coluna 'partida_prevista' é necessária para cortes temporais.")

    cfg = SplitConfig()
    cut = _temporal_cutoffs(fv["partida_prevista"], cfg)
    train_end = pd.to_datetime(cut["train_end"])
    val_end = pd.to_datetime(cut["val_end"])

    t = pd.to_datetime(fv["partida_prevista"], errors="coerce")
    train = fv[t <= train_end].copy()
    val = fv[(t > train_end) & (t <= val_end)].copy()
    test = fv[t > val_end].copy()

    if train.empty or val.empty or test.empty:
        q80 = t.quantile(0.80)
        q95 = t.quantile(0.95)
        train = fv[t <= q80].copy()
        val = fv[(t > q80) & (t <= q95)].copy()
        test = fv[t > q95].copy()
        if val.empty or test.empty:
            q75 = t.quantile(0.75)
            train = fv[t <= q75].copy()
            test = fv[t > q75].copy()
            val = test.sample(0)

    train_path = SPLITS_DIR / "train.parquet"
    val_path = SPLITS_DIR / "val.parquet"
    test_path = SPLITS_DIR / "test.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    manifest = {
        "source": str(p),
        "rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "cutoffs": cut,
        "target": TARGET_COL,
        "columns": list(fv.columns),
        "categorical": [c for c in SAFE_CATEGORICAL if c in fv.columns],
        "numerical": [c for c in SAFE_NUMERICAL if c in fv.columns],
        "note": "Feature view livre de vazamento: não inclui delay_*, *_real.",
    }
    (SPLITS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("✔ Splits temporais salvos em data/processed/splits/")
    print(f"  - train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    print(f"  - manifest: {SPLITS_DIR / 'manifest.json'}")


def pretrain_check():
    p_hist = PROCESSED_DIR / "dataset_modelagem_hist.parquet"
    p_weather = PROCESSED_DIR / "dataset_modelagem_weather.parquet"
    p_base = PROCESSED_DIR / "dataset_modelagem.parquet"
    p = p_hist if p_hist.exists() else (p_weather if p_weather.exists() else p_base)
    if not p.exists():
        raise SystemExit("Base não encontrada. Rode: unify_all (e weather/features opcional).")

    df = pd.read_parquet(p)
    if "situacao" in df.columns:
        df = df[df["situacao"].str.contains("REALIZADO", na=False)].copy()
    if "atraso15" not in df.columns:
        raise SystemExit("Coluna 'atraso15' ausente.")
    df = df[df["atraso15"].notna()].copy()
    if "partida_prevista" not in df.columns:
        raise SystemExit("Coluna 'partida_prevista' ausente.")

    df["partida_prevista"] = pd.to_datetime(df["partida_prevista"], errors="coerce")
    tmin, tmax = df["partida_prevista"].min(), df["partida_prevista"].max()
    df["ano_mes"] = df["partida_prevista"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("ano_mes")["atraso15"]
        .agg(n="count", pos="sum", pos_rate=lambda s: float(s.mean()))
        .reset_index()
        .sort_values("ano_mes")
    )

    feat_keys = [
        "icao_empresa",
        "origem_icao",
        "destino_icao",
        "rota",
        "mes",
        "dia_semana",
        "hora_partida_prevista",
    ]
    feat_keys = [c for c in feat_keys if c in df.columns]
    nulls = df[feat_keys].isna().mean().sort_values(ascending=False).rename("pct_nulls").to_frame()

    wx_cols = [c for c in ["w_dep_precip_mm", "w_dep_wind_ms", "w_dep_gust_ms", "w_dep_temp_c", "w_dep_rh"] if c in df.columns]
    wx_cov = None
    if wx_cols:
        wx_cov = (1 - df[wx_cols].isna().mean()).rename("coverage").to_frame()

    dup_cols = [c for c in ["icao_empresa", "numero_voo", "partida_prevista"] if c in df.columns]
    dups = 0
    if len(dup_cols) == 3:
        dups = int(df.duplicated(subset=dup_cols).sum())

    over_year = int((df["partida_prevista"].dt.year > 2024).sum())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(REPORTS_DIR / "pretrain_monthly.csv", index=False, encoding="utf-8")

    md = []
    md.append("# Pré-treino — Diagnóstico\n")
    md.append(f"- Base: `{p.name}`")
    md.append(f"- Linhas (REALIZADO + label não nula): **{len(df):,}**")
    md.append(f"- Janela temporal: **{tmin}** → **{tmax}**")
    md.append(f"- Duplicidades potenciais (icao_empresa, numero_voo, partida_prevista): **{dups}**")
    md.append(f"- Registros com ano > 2024: **{over_year}**")
    md.append("\n## Nulos nas features-chave")
    md.append(nulls.to_markdown() if hasattr(nulls, "to_markdown") else nulls.to_string())
    if wx_cov is not None:
        md.append("\n## Cobertura de clima (proporção de não-nulos)")
        md.append(wx_cov.to_markdown() if hasattr(wx_cov, "to_markdown") else wx_cov.to_string())
    md.append("\n## Classe positiva por mês (pretrain_monthly.csv salvo)")
    (REPORTS_DIR / "pretrain_check.md").write_text("\n\n".join(md), encoding="utf-8")

    print("✔ Pré-treino OK")
    print("  - reports/pretrain_check.md")
    print("  - reports/pretrain_monthly.csv")


def run_all():
    unify_vra_all()
    weather_pipeline()
    add_historical_features()
    pretrain_check()
    make_temporal_splits()
    make_trainset()
    print("✔ Pipeline completo concluído.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cmd",
        choices=["inspect","unify","unify_all","profile","weather","trainset","features","splits","pretrain_check","all"],
        help="comando"
    )
    args = parser.parse_args()
    if args.cmd == "inspect":
        inspect_all()
    elif args.cmd == "unify":
        unify_vra_2024()
    elif args.cmd == "unify_all":
        unify_vra_all()
    elif args.cmd == "profile":
        profile_vra()
    elif args.cmd == "weather":
        weather_pipeline()
    elif args.cmd == "trainset":
        make_trainset()
    elif args.cmd == "features":
        add_historical_features()
    elif args.cmd == "pretrain_check":
        pretrain_check()
    elif args.cmd == "splits":
        make_temporal_splits()
    elif args.cmd == "all":
        run_all()

if __name__ == "__main__":
    main()
