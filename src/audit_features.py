from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

PROCESSED=Path("data/processed"); REPORTS=Path("reports"); REPORTS.mkdir(parents=True,exist_ok=True)

NUM_WX_DEP=["w_dep_precip_mm","w_dep_wind_ms","w_dep_gust_ms","w_dep_temp_c","w_dep_rh"]
NUM_WX_ARR=["w_arr_precip_mm","w_arr_wind_ms","w_arr_gust_ms","w_arr_temp_c","w_arr_rh"]
WX_FLAGS=["wx_chuva","wx_chuva_forte","wx_vento_forte","wx_rajada_forte","wx_calor","wx_frio","wx_umidade_alta","wx_missing_any","wx_arr_chuva","wx_arr_chuva_forte","wx_arr_vento_forte","wx_arr_rajada_forte","wx_arr_calor","wx_arr_frio","wx_arr_umidade_alta","wx_arr_missing_any","visibilidade_ruim"]
WX_ISNA=[f"{c}_isna" for c in NUM_WX_DEP+NUM_WX_ARR]
BUCKETS=["precip_q","wind_q","gust_q","temp_q","rh_q","arr_precip_q","arr_wind_q","arr_gust_q","arr_temp_q","arr_rh_q"]
HIST=["hist_atraso_rota_30","hist_atraso_empresa_50","hist_atraso_num_voo_10","hist_vol_rota_30","hist_std_rota_30","atrasos_mesmo_aeroporto_1h_hist"]
BASIC=["icao_empresa","origem_icao","destino_icao","rota","mes","dia_semana","hora_partida_prevista","hora_bloco_30","hora_sin","hora_cos","periodo_dia_id","is_weekend","is_feriado","is_vespera_feriado","em_ferias","load_origem_15","dist_km","airport_size_id","congestion_ratio","congestion_bucket","partida_prevista","atraso15"]

def _pick_base():
    for n in ["dataset_modelagem_hist.parquet","dataset_modelagem_weather.parquet","dataset_modelagem.parquet"]:
        p=PROCESSED/n
        if p.exists(): return p
    raise SystemExit("nenhuma base encontrada em data/processed/")

def _summ_num(df, cols):
    out=[]
    for c in cols:
        if c not in df.columns: continue
        s=pd.to_numeric(df[c],errors="coerce")
        n=len(s); nn=s.notna().sum(); nz=(s==0).sum() if nn>0 else 0
        out.append([c,n,nn,nn/n if n else np.nan,nz,nz/n if n else np.nan,s.nunique(dropna=True),s.min(),s.quantile(.5),s.mean(),s.quantile(.9),s.quantile(.99),s.max()])
    return pd.DataFrame(out,columns=["col","n","non_null","non_null_rate","zeros","zero_rate","n_unique","min","p50","mean","p90","p99","max"]).sort_values(["zero_rate","n_unique"],ascending=[False,True])

def _summ_cat(df, cols):
    out=[]
    for c in cols:
        if c not in df.columns: continue
        s=df[c].astype("string")
        vc=s.value_counts(dropna=True).head(5)
        top=" | ".join([f"{k}:{int(v)}" for k,v in vc.items()])
        out.append([c,len(s),s.notna().mean(),s.nunique(dropna=True),top])
    return pd.DataFrame(out,columns=["col","n","non_null_rate","n_unique","top5"]).sort_values(["n_unique"],ascending=[True])

def _by_airport_cov(df):
    if "origem_icao" not in df.columns: return pd.DataFrame()
    g=df.groupby(df["origem_icao"].astype(str).str.upper().str.strip())
    metas=[]
    cols=[c for c in NUM_WX_DEP+NUM_WX_ARR if c in df.columns]
    for ap,part in g:
        if len(part)<100: continue
        cov={f"cov_{c}":float(part[c].notna().mean()) for c in cols}
        metas.append({"origem_icao":ap,"n":len(part),**cov})
    if not metas: return pd.DataFrame()
    return pd.DataFrame(metas).sort_values("n",ascending=False)

def _target_lift(df, cols):
    if "atraso15" not in df.columns: return pd.DataFrame()
    y=pd.to_numeric(df["atraso15"],errors="coerce")
    out=[]
    for c in cols:
        if c not in df.columns: continue
        s=df[c]
        if pd.api.types.is_numeric_dtype(s):
            q=pd.qcut(s.rank(method="first"),q=5,duplicates="drop")
            lift=y.groupby(q).mean()
            out.append([c,"num",float(y.mean()),";".join([f"{i}:{v:.3f}" for i,v in enumerate(lift.values)])])
        else:
            vc=s.astype("string").value_counts().head(6).index
            lift=y.groupby(s.astype("string").where(s.isin(vc),"__other__")).mean()
            out.append([c,"cat",float(y.mean()),";".join([f"{k}:{v:.3f}" for k,v in lift.items()])])
    return pd.DataFrame(out,columns=["col","type","base_pos","lift"]).sort_values("col")

def audit(args):
    p=_pick_base(); df=pd.read_parquet(p)
    cols_exist=[c for c in BASIC+NUM_WX_DEP+NUM_WX_ARR+WX_FLAGS+WX_ISNA+BUCKETS+HIST if c in df.columns]
    missing=sorted(set(BASIC+NUM_WX_DEP+NUM_WX_ARR+WX_FLAGS+WX_ISNA+BUCKETS+HIST)-set(df.columns))
    num_cols=[c for c in cols_exist if pd.api.types.is_numeric_dtype(df[c]) and c!="atraso15"]
    cat_cols=[c for c in cols_exist if c not in num_cols and c!="atraso15"]
    num= _summ_num(df,num_cols)
    cat= _summ_cat(df,cat_cols)
    zeros=num[num["zero_rate"]>=0.95]
    const=pd.concat([num[num["n_unique"]<=1][["col"]],cat[cat["n_unique"]<=1][["col"]]],ignore_index=True)
    lift=_target_lift(df,[c for c in cols_exist if c!="atraso15"])
    apcov=_by_airport_cov(df)
    num.to_csv(REPORTS/"numeric_summary.csv",index=False)
    cat.to_csv(REPORTS/"categorical_summary.csv",index=False)
    zeros.to_csv(REPORTS/"mostly_zeros.csv",index=False)
    const.to_csv(REPORTS/"constant_columns.csv",index=False)
    lift.to_csv(REPORTS/"target_lift.csv",index=False)
    if not apcov.empty: apcov.to_csv(REPORTS/"airport_weather_coverage.csv",index=False)
    md=[]
    md.append(f"# Audit {p.name}")
    md.append(f"- linhas: {len(df):,}")
    md.append(f"- alvo presente: {'atraso15' in df.columns}")
    md.append(f"- features encontradas: {len(cols_exist)}")
    md.append(f"- ausentes: {', '.join(missing) if missing else '(nenhuma)'}")
    if "atraso15" in df.columns:
        pr=float(pd.to_numeric(df["atraso15"],errors="coerce").mean()); md.append(f"- taxa de atraso (>15): {pr:.2%}")
    if not zeros.empty: md.append(f"- colunas com ≥95% zeros: {', '.join(zeros['col'].head(30))} {'...' if len(zeros)>30 else ''}")
    if not const.empty: md.append(f"- colunas constantes: {', '.join(const['col'].head(30))} {'...' if len(const)>30 else ''}")
    if any(c in df.columns for c in BUCKETS): md.append("- buckets de clima: presentes"); 
    else: md.append("- buckets de clima: ausentes (verifique features.py e make_trainset)")
    (REPORTS/"feature_audit.md").write_text("\n".join(md),encoding="utf-8")
    summary={"base":p.name,"rows":len(df),"present":cols_exist,"missing":missing}
    (REPORTS/"feature_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n".join(md))
    print("csvs salvos em reports/: numeric_summary.csv, categorical_summary.csv, mostly_zeros.csv, constant_columns.csv, target_lift.csv", "e airport_weather_coverage.csv" if not apcov.empty else "")

def head(args):
    p=_pick_base(); df=pd.read_parquet(p)
    cols=args.cols.split(",") if args.cols else None
    print(df[cols].head(int(args.n)) if cols else df.head(int(args.n)))

def stats(args):
    p=_pick_base(); df=pd.read_parquet(p)
    cols=[c for c in args.cols.split(",") if c in df.columns]
    print(_summ_num(df,cols))

def value_counts(args):
    p=_pick_base(); df=pd.read_parquet(p)
    col=args.col
    if col not in df.columns: raise SystemExit(f"coluna {col} não existe")
    vc=df[col].value_counts(dropna=False).head(int(args.k))
    print(vc.to_string())

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("audit"); s.set_defaults(func=audit)
    s=sub.add_parser("head"); s.add_argument("-n",default="5"); s.add_argument("--cols",default=""); s.set_defaults(func=head)
    s=sub.add_parser("stats"); s.add_argument("--cols",required=True); s.set_defaults(func=stats)
    s=sub.add_parser("vc"); s.add_argument("--col",required=True); s.add_argument("-k",default="20"); s.set_defaults(func=value_counts)
    args=ap.parse_args(); args.func(args)

if __name__=="__main__":
    main()
