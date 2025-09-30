# src/model_train.py
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix, classification_report
from catboost import CatBoostClassifier, Pool

SPLITS_DIR = Path("data/processed/splits")
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR  = Path("data/models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "atraso15"

CATEGORICAL_BASE = ["icao_empresa","origem_icao","destino_icao","rota"]
NUMERICAL_BASE = [
    "mes","dia_semana","hora_partida_prevista","hora_bloco_30","hora_sin","hora_cos","periodo_dia_id",
    "is_weekend","is_feriado","is_vespera_feriado","em_ferias",
    "chuva_id","tempo_ruim_id","chuva_arr_id","tempo_ruim_arr_id",
    "load_origem_15","dist_km","airport_size_id","congestion_ratio","congestion_bucket",
    "sched_block_min","sched_block_bucket",
    "wx_dep_cov","wx_dep_all_missing","wx_arr_cov","wx_arr_all_missing",
    "hist_atraso_rota_30","hist_atraso_empresa_50","hist_atraso_num_voo_10","hist_vol_rota_30","hist_std_rota_30",
    "hist_atraso_origem_hora_30","atrasos_mesmo_aeroporto_1h_hist","atrasos_mesmo_aeroporto_hora"
]

def load_split(name:str)->pd.DataFrame:
    p=SPLITS_DIR/f"{name}.parquet"
    if not p.exists(): raise SystemExit(f"Split '{name}' não encontrado em {p}")
    df=pd.read_parquet(p)
    if TARGET not in df.columns: raise SystemExit(f"Coluna alvo '{TARGET}' ausente em {p}")
    df[TARGET]=df[TARGET].astype(int)
    for c in CATEGORICAL_BASE:
        if c in df.columns: df[c]=df[c].astype(str).str.upper().str.strip()
    for c in NUMERICAL_BASE:
        if c in df.columns and df[c].dtype.name.startswith("Int"): df[c]=df[c].astype("float")
    return df

def infer_feature_columns(df:pd.DataFrame):
    cat_cols=[c for c in CATEGORICAL_BASE if c in df.columns]
    num_cols=[c for c in NUMERICAL_BASE if c in df.columns]
    dyn_cols=[c for c in df.columns if c.startswith(("hist_","lag_")) or c.endswith("_hist")]
    for c in dyn_cols:
        if df[c].dtype.name in ("Int8","Int16","Int32","Int64","UInt8","UInt16","UInt32","UInt64","object","string"):
            with np.errstate(all="ignore"): df[c]=pd.to_numeric(df[c],errors="coerce")
    seen=set(); num_cols=[c for c in (num_cols+dyn_cols) if not (c in seen or seen.add(c))]
    if "atrasos_mesmo_aeroporto_1h_hist" in num_cols and "atrasos_mesmo_aeroporto_hora" in num_cols:
        num_cols=[c for c in num_cols if c!="atrasos_mesmo_aeroporto_hora"]
    return cat_cols,num_cols

def build_Xy(df:pd.DataFrame,cat_cols:list[str],num_cols:list[str]):
    cols=[c for c in cat_cols+num_cols if c in df.columns]
    X=df[cols].copy(); y=df[TARGET].astype(int).values
    cat_idx=[i for i,c in enumerate(cols) if c in cat_cols]
    return X,y,cols,cat_idx

def impute_dist_km_by_route(tr,va,te):
    if "dist_km" not in tr.columns: return tr,va,te
    def _rota_series(df): return df["rota"].astype(str) if "rota" in df.columns else pd.Series(index=df.index,dtype="string")
    tr_rota=_rota_series(tr); m_valid=tr["dist_km"].notna() & tr_rota.notna()
    if m_valid.any():
        rota_mean=tr.loc[m_valid].groupby(tr_rota[m_valid])["dist_km"].mean()
        glob_med=float(tr.loc[tr["dist_km"].notna(),"dist_km"].median())
    else:
        rota_mean=pd.Series(dtype="float64"); glob_med=float("nan")
    def _fill(df):
        df=df.copy()
        if "dist_km" in df.columns:
            if "rota" in df.columns and not rota_mean.empty: df["dist_km"]=df["dist_km"].fillna(df["rota"].astype(str).map(rota_mean))
            df["dist_km"]=df["dist_km"].fillna(0.0 if np.isnan(glob_med) else glob_med)
        return df
    return _fill(tr),_fill(va),_fill(te)

def pick_threshold_f1(y_true,proba):
    prec,rec,thr=precision_recall_curve(y_true,proba); f1=(2*prec[:-1]*rec[:-1])/(prec[:-1]+rec[:-1]+1e-9)
    i=int(np.nanargmax(f1)); return float(thr[i]),{"policy":"f1","F1":float(f1[i]),"precision":float(prec[i]),"recall":float(rec[i])}

def pick_threshold_fbeta(y_true,proba,beta:float=1.0):
    prec,rec,thr=precision_recall_curve(y_true,proba); b2=beta*beta
    fbeta=(1+b2)*prec[:-1]*rec[:-1]/(b2*prec[:-1]+rec[:-1]+1e-9)
    i=int(np.nanargmax(fbeta)); return float(thr[i]),{"policy":f"f_beta({beta})","F_beta":float(fbeta[i]),"precision":float(prec[i]),"recall":float(rec[i])}

def pick_threshold_prec_at(y_true,proba,min_precision:float=0.5):
    prec,rec,thr=precision_recall_curve(y_true,proba); mask=prec[:-1]>=min_precision
    if not np.any(mask): return pick_threshold_f1(y_true,proba)
    idx=np.where(mask)[0]; best=idx[np.argmax(rec[:-1][idx])]
    return float(thr[best]),{"policy":f"prec_at({min_precision})","precision":float(prec[best]),"recall":float(rec[best])}

def pick_threshold_recall_at(y_true,proba,min_recall:float=0.7):
    prec,rec,thr=precision_recall_curve(y_true,proba); mask=rec[:-1]>=min_recall
    if not np.any(mask): return pick_threshold_f1(y_true,proba)
    idx=np.where(mask)[0]; best=idx[np.argmax(prec[:-1][idx])]
    return float(thr[best]),{"policy":f"recall_at({min_recall})","precision":float(prec[best]),"recall":float(rec[best])}

def pick_threshold(y_true,proba,policy:str="f1",beta:float=1.0,min_precision:float|None=None,min_recall:float|None=None):
    p=policy.lower()
    if p=="f1": return pick_threshold_f1(y_true,proba)
    if p=="f_beta": return pick_threshold_fbeta(y_true,proba,beta=beta)
    if p=="prec_at": return pick_threshold_prec_at(y_true,proba,min_precision=(0.5 if min_precision is None else min_precision))
    if p=="recall_at": return pick_threshold_recall_at(y_true,proba,min_recall=(0.7 if min_recall is None else min_recall))
    return pick_threshold_f1(y_true,proba)

def eval_at_threshold(y_true,proba,thr):
    y_pred=(proba>=thr).astype(int)
    return {"AUC":float(roc_auc_score(y_true,proba)),"PR_AUC":float(average_precision_score(y_true,proba)),"F1":float(f1_score(y_true,y_pred)),"confusion_matrix":confusion_matrix(y_true,y_pred).tolist(),"report":classification_report(y_true,y_pred,output_dict=True)}

def choose_groups_for_sampling(df):
    if "rota" in df.columns and "mes" in df.columns: return ["rota","mes"]
    if all(c in df.columns for c in ["origem_icao","destino_icao","mes"]): return ["origem_icao","destino_icao","mes"]
    if "partida_prevista" in df.columns: dt=pd.to_datetime(df["partida_prevista"],errors="coerce"); return [dt.dt.to_period("M").astype(str).rename("ano_mes")]
    if "mes" in df.columns: return ["mes"]
    return []

def stratified_negative_undersample(df,target_col:str,target_pos:float=0.35,group_cols:list[str]|list[pd.Series]|None=None,neg_cap_per_group:int=500,seed:int=42)->pd.DataFrame:
    def _ensure_series(x): return x if isinstance(x,pd.Series) else pd.Series(x)
    work=df.copy(); materialized=[]; temp=[]
    if group_cols:
        for gc in group_cols:
            if isinstance(gc,pd.Series):
                name=gc.name or "grp_tmp"; work[name]=_ensure_series(gc).values; materialized.append(name); temp.append(name)
            else: materialized.append(gc)
    pos=work[work[target_col]==1]; neg=work[work[target_col]==0]
    if len(pos)==0 or len(neg)==0: return df.sample(frac=1.0,random_state=seed).reset_index(drop=True)
    n_neg=int(min(len(neg),np.ceil(len(pos)*(1.0-target_pos)/max(target_pos,1e-9))))
    if not materialized: neg_samp=neg.sample(n=n_neg,random_state=seed,replace=False)
    else:
        parts=[]
        for _,g in neg.groupby(materialized,dropna=False):
            k=min(neg_cap_per_group,len(g))
            if k>0: parts.append(g.sample(n=k,random_state=seed,replace=False))
        pool=pd.concat(parts,ignore_index=False) if parts else neg
        if len(pool)>n_neg: neg_samp=pool.sample(n=n_neg,random_state=seed,replace=False)
        else:
            need=n_neg-len(pool)
            if need>0:
                remaining=neg.drop(index=pool.index,errors="ignore")
                extra=remaining.sample(n=min(need,len(remaining)),random_state=seed,replace=False)
                neg_samp=pd.concat([pool,extra],ignore_index=False)
            else: neg_samp=pool
    out=pd.concat([pos,neg_samp],ignore_index=False).sample(frac=1.0,random_state=seed)
    if temp: out=out.drop(columns=[c for c in temp if c in out.columns],errors="ignore")
    return out.reset_index(drop=True)

def balance_train_df(train_df:pd.DataFrame,mode:str="none",target_col:str="atraso15",target_pos:float=0.35,neg_cap_per_group:int=500,seed:int=42)->pd.DataFrame:
    if mode=="none": return train_df
    groups=[]
    if mode=="undersample_neg": groups=choose_groups_for_sampling(train_df)
    elif mode=="month_quota":
        if "partida_prevista" in train_df.columns:
            dt=pd.to_datetime(train_df["partida_prevista"],errors="coerce"); groups=[dt.dt.to_period("M").astype(str).rename("ano_mes")]
        elif "mes" in train_df.columns: groups=["mes"]
    elif mode=="route_quota":
        if "rota" in train_df.columns and "mes" in train_df.columns: groups=["rota","mes"]
        elif "rota" in train_df.columns: groups=["rota"]
        elif all(c in train_df.columns for c in ["origem_icao","destino_icao"]): groups=["origem_icao","destino_icao"]
    else: raise ValueError(f"Modo de balanceamento desconhecido: {mode}")
    return stratified_negative_undersample(train_df,target_col,target_pos,groups,neg_cap_per_group,seed)

def resolve_catboost_device(want_device:str):
    want_device=(want_device or "cpu").lower()
    return {"task_type":"GPU","devices":"0"} if want_device=="gpu" else {}

def train_catboost(Xtr,ytr,Xva,yva,cols,cat_idx,args,cat_device_params):
    pos_rate=float(np.mean(ytr)); class_weights=None
    if 0<pos_rate<1: w_pos=(1.0-pos_rate)/max(pos_rate,1e-9); class_weights=[1.0,w_pos]
    model=CatBoostClassifier(loss_function="Logloss",eval_metric="AUC",learning_rate=args.lr,depth=args.depth,l2_leaf_reg=args.l2,iterations=args.iters,od_type="Iter",od_wait=args.od_wait,random_seed=args.seed,verbose=args.verbose,class_weights=class_weights,**cat_device_params)
    try:
        model.fit(Pool(Xtr,ytr,cat_features=cat_idx),eval_set=Pool(Xva,yva,cat_features=cat_idx),use_best_model=True)
    except Exception as e:
        if cat_device_params:
            warnings.warn(f"CatBoost GPU falhou ({e}). Fallback CPU…")
            model=CatBoostClassifier(loss_function="Logloss",eval_metric="AUC",learning_rate=args.lr,depth=args.depth,l2_leaf_reg=args.l2,iterations=args.iters,od_type="Iter",od_wait=args.od_wait,random_seed=args.seed,verbose=args.verbose,class_weights=class_weights)
            model.fit(Pool(Xtr,ytr,cat_features=cat_idx),eval_set=Pool(Xva,yva,cat_features=cat_idx),use_best_model=True)
        else: raise
    return model

def feature_importances(model,cols):
    imp=model.get_feature_importance()
    return dict(zip(cols,[float(x) for x in imp]))

def per_100_report(y_true,proba,thr):
    y_pred=(proba>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y_true,y_pred).ravel(); n=len(y_true); scale=100.0/max(n,1)
    return {"base_rate_delay_per_100":round(scale*(tp+fn),2),"alerts_per_100":round(scale*(tp+fp),2),
            "tp_per_100":round(scale*tp,2),"fp_per_100":round(scale*fp,2),"fn_per_100":round(scale*fn,2),"tn_per_100":round(scale*tn,2),
            "precision":float(tp/max(tp+fp,1e-9)),"recall":float(tp/max(tp+fn,1e-9))}

def build_perf_row(name,y_true,proba,thr):
    y_pred=(proba>=thr).astype(int); rep=classification_report(y_true,y_pred,output_dict=True); tn,fp,fn,tp=confusion_matrix(y_true,y_pred).ravel(); n=len(y_true)
    return {"Name":name,"Precision":float(rep["1"]["precision"]),"Recall":float(rep["1"]["recall"]),"F1":float(rep["1"]["f1-score"]),
            "Accuracy":float(rep["accuracy"]),"Alerts/100":round(100.0*(tp+fp)/max(n,1),1),"BaseRate/100":round(100.0*(tp+fn)/max(n,1),1)}

def print_and_save_summary_table(rows:list[dict]):
    df=pd.DataFrame(rows,columns=["Name","Precision","Recall","F1","Accuracy","Alerts/100","BaseRate/100"])
    show=df.copy()
    for c in ["Precision","Recall","F1","Accuracy"]: show[c]=show[c].map(lambda x:f"{x:.4f}")
    for c in ["Alerts/100","BaseRate/100"]: show[c]=show[c].map(lambda x:f"{x:.2f}")
    print("\n— Tabela resumo (classe=1) —"); print(show.to_string(index=False))
    out_csv=REPORTS_DIR/"summary_table.csv"; df.to_csv(out_csv,index=False,encoding="utf-8"); print(f"✔ Tabela resumo salva em {out_csv}")

# ---------- Classificação e resumo final ----------
def _classify_offline(roc_auc: float, pr_auc: float, base_rate: float) -> str:
    pr_ratio = pr_auc / max(base_rate, 1e-9)
    if roc_auc >= 0.85 or pr_ratio >= 3.0: return "excelente"
    if roc_auc >= 0.75 or pr_ratio >= 2.0: return "ótimo"
    if roc_auc >= 0.65 or pr_ratio >= 1.5: return "bom"
    return "ruim"

def _print_and_save_quick_summary(metrics: dict, mini_val: dict, mini_tst: dict):
    test = metrics["test"]; val = metrics["val"]
    base = metrics["class_balance"]["test_pos_rate"]
    thr = metrics["threshold"]; pol = metrics["threshold_policy"]
    label = _classify_offline(test["AUC"], test["PR_AUC"], base)
    txt = []
    txt.append("— Resumo rápido —")
    txt.append(f"Conjunto: TEST | AUC={test['AUC']:.4f} | PR-AUC={test['PR_AUC']:.4f} (baseline={base:.3f}) | F1={test['F1']:.4f}")
    txt.append(f"Threshold={thr:.4f} ({pol['policy']}) | Precision@thr={pol.get('precision',float('nan')):.3f} | Recall@thr={pol.get('recall',float('nan')):.3f}")
    txt.append(f"Por 100 voos (TEST): atrasos reais={mini_tst['base_rate_delay_per_100']:.1f} | alertas={mini_tst['alerts_per_100']:.1f} | TP={mini_tst['tp_per_100']:.1f} | FP={mini_tst['fp_per_100']:.1f}")
    txt.append(f"Classificação offline: {label.upper()}")
    out = "\n".join(txt)
    print("\n"+out)
    (REPORTS_DIR/"quick_summary.txt").write_text(out+"\n", encoding="utf-8")

# ---- RF CPU helpers ----
def _cat_maps_from_train(X,cat_cols):
    maps={}
    for c in cat_cols:
        s=X[c].astype(str).fillna("__NA__"); cats=pd.Index(pd.unique(s)); maps[c]={v:i+1 for i,v in enumerate(cats)}
    return maps

def _apply_cat_maps(X,cat_cols,maps):
    X=X.copy()
    for c in cat_cols:
        m=maps[c]; s=X[c].astype(str).fillna("__NA__"); X[c]=s.map(m).fillna(0).astype("int32")
    return X

def _train_medians(X,num_cols): return {c:pd.to_numeric(X[c],errors="coerce").median() for c in num_cols}
def _apply_medians(X,num_cols,med):
    X=X.copy()
    for c in num_cols: X[c]=pd.to_numeric(X[c],errors="coerce").fillna(med[c]).astype("float32")
    return X

def _parse_max_features(v):
    if v is None: return None
    if isinstance(v,(int,float)): return int(v) if v>=1 else max(0.05,min(1.0,float(v)))
    s=str(v).strip().lower()
    if s in ("sqrt","log2"): return s
    if s in ("none","null"): return None
    try:
        f=float(s)
        return int(f) if f>=1 else max(0.05,min(1.0,f))
    except ValueError:
        return "sqrt"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=["catboost","rf"],default="catboost")
    p.add_argument("--iters",type=int,default=5000); p.add_argument("--lr",type=float,default=0.03); p.add_argument("--depth",type=int,default=8); p.add_argument("--l2",type=float,default=3.0)
    p.add_argument("--od-wait",dest="od_wait",type=int,default=500); p.add_argument("--verbose",type=int,default=200)
    p.add_argument("--balance",choices=["none","undersample_neg","month_quota","route_quota"],default="none")
    p.add_argument("--target-pos",dest="target_pos",type=float,default=0.35); p.add_argument("--neg-cap-per-group",dest="neg_cap_per_group",type=int,default=500)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--th-policy",choices=["f1","f_beta","prec_at","recall_at"],default="f1"); p.add_argument("--beta",type=float,default=1.0)
    p.add_argument("--min-precision",dest="min_precision",type=float,default=None); p.add_argument("--min-recall",dest="min_recall",type=float,default=None)
    p.add_argument("--device",choices=["cpu","gpu"],default="cpu")
    p.add_argument("--drop-features",nargs="*",default=[]); p.add_argument("--prune-below",type=float,default=None)
    p.add_argument("--rf-trees",dest="rf_trees",type=int,default=600)
    p.add_argument("--rf-max-depth",dest="rf_max_depth",type=int,default=18)
    p.add_argument("--rf-max-features",dest="rf_max_features",default="sqrt")
    p.add_argument("--rf-min-leaf",dest="rf_min_leaf",type=int,default=1)
    args=p.parse_args()

    tr=load_split("train"); va=load_split("val"); te=load_split("test")
    tr,va,te=impute_dist_km_by_route(tr,va,te)

    if args.balance!="none":
        before=len(tr)
        tr=balance_train_df(tr,mode=args.balance,target_col=TARGET,target_pos=args.target_pos,neg_cap_per_group=args.neg_cap_per_group,seed=args.seed)
        after=len(tr); pos_rate=float(tr[TARGET].mean())
        print(f"✔ Balanceamento '{args.balance}': {before:,} → {after:,} | pos_rate≈{pos_rate:.2%}")

    if args.drop_features:
        print("🔧 Removendo features:",", ".join(args.drop_features))
        tr=tr.drop(columns=[c for c in args.drop_features if c in tr.columns],errors="ignore")
        va=va.drop(columns=[c for c in args.drop_features if c in va.columns],errors="ignore")
        te=te.drop(columns=[c for c in args.drop_features if c in te.columns],errors="ignore")

    cat_cols,num_cols=infer_feature_columns(tr)
    Xtr,ytr,cols,cat_idx=build_Xy(tr,cat_cols,num_cols)
    Xva,yva,_,_=build_Xy(va,cat_cols,num_cols)
    Xte,yte,_,_=build_Xy(te,cat_cols,num_cols)

    model_type=args.model; feat_imp_map=None

    if args.model=="catboost":
        cat_dev=resolve_catboost_device(args.device)
        print("⚙️  Modelo: CatBoost | Dispositivo:",("GPU" if cat_dev else "CPU"))
        model=train_catboost(Xtr,ytr,Xva,yva,cols,cat_idx,args,cat_dev)
        p_tr=model.predict_proba(Xtr)[:,1]; p_va=model.predict_proba(Xva)[:,1]; p_te=model.predict_proba(Xte)[:,1]
        feat_imp_map=feature_importances(model,cols)
        out_path=MODELS_DIR/"catboost_delay.cbm"; model.save_model(str(out_path))
        print(f"\n✔ Modelo (catboost) salvo em {out_path}")

    elif args.model=="rf":
        print("⚙️  Modelo: RandomForest (sklearn CPU)")
        from sklearn.ensemble import RandomForestClassifier as RF
        cat_in=[c for c in cols if c in CATEGORICAL_BASE]; num_in=[c for c in cols if c not in CATEGORICAL_BASE]
        cat_maps=_cat_maps_from_train(Xtr,cat_in)
        Xtr_enc=_apply_cat_maps(Xtr,cat_in,cat_maps); Xva_enc=_apply_cat_maps(Xva,cat_in,cat_maps); Xte_enc=_apply_cat_maps(Xte,cat_in,cat_maps)
        med=_train_medians(Xtr_enc,num_in)
        Xtr_enc=_apply_medians(Xtr_enc,num_in,med); Xva_enc=_apply_medians(Xva_enc,num_in,med); Xte_enc=_apply_medians(Xte_enc,num_in,med)
        Xtr_np=Xtr_enc[cols].to_numpy(dtype=np.float32); Xva_np=Xva_enc[cols].to_numpy(dtype=np.float32); Xte_np=Xte_enc[cols].to_numpy(dtype=np.float32)

        rf_max_features=_parse_max_features(args.rf_max_features)
        rf=RF(n_estimators=args.rf_trees,
              max_depth=(args.rf_max_depth or None),
              max_features=rf_max_features,
              min_samples_leaf=args.rf_min_leaf,
              bootstrap=True, random_state=args.seed, n_jobs=-1)

        rf.fit(Xtr_np,ytr)
        p_tr=rf.predict_proba(Xtr_np)[:,1]; p_va=rf.predict_proba(Xva_np)[:,1]; p_te=rf.predict_proba(Xte_np)[:,1]
        fi=rf.feature_importances_.ravel().tolist(); feat_imp_map=dict(zip(cols,[float(x) for x in fi]))
        import joblib; save_path=MODELS_DIR/"rf_cpu_delay.joblib"
        joblib.dump({"rf":rf,"cols":cols,"cat_maps":cat_maps,"medians":med},save_path)
        print(f"\n✔ Modelo (rf cpu) salvo em {save_path}")

    else:
        raise SystemExit("Modelo inválido.")

    thr,thr_info=pick_threshold(yva,p_va,policy=args.th_policy,beta=args.beta,min_precision=args.min_precision,min_recall=args.min_recall)
    metrics={"sizes":{"train":int(len(tr)),"val":int(len(va)),"test":int(len(te))},
             "class_balance":{"train_pos_rate":float(np.mean(ytr)),"val_pos_rate":float(np.mean(yva)),"test_pos_rate":float(np.mean(yte))},
             "threshold":thr,"threshold_policy":thr_info,
             "train":eval_at_threshold(ytr,p_tr,thr),"val":eval_at_threshold(yva,p_va,thr),"test":eval_at_threshold(yte,p_te,thr),
             "features":cols,"categorical":cat_cols,"numerical":num_cols,
             "categorical_idx":[int(i) for i in range(len(cols)) if cols[i] in cat_cols],
             "feature_importance":(feat_imp_map if feat_imp_map is not None else {c:0.0 for c in cols}),
             "model_type":model_type,
             "params":{"iters":args.iters,"lr":args.lr,"depth":args.depth,"l2":args.l2,"od_wait":args.od_wait,
                       "balance":args.balance,"target_pos":args.target_pos,"neg_cap_per_group":args.neg_cap_per_group,
                       "device":args.device,"drop_features":args.drop_features,"prune_below":args.prune_below,
                       "seed":args.seed,"rf_trees":args.rf_trees,"rf_max_depth":args.rf_max_depth,
                       "rf_min_leaf":args.rf_min_leaf,"rf_max_features":str(args.rf_max_features)}}
    (REPORTS_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
    pd.DataFrame({"feature":cols,"importance":[metrics["feature_importance"][c] for c in cols]}).sort_values("importance",ascending=False).to_csv(REPORTS_DIR/(f"feature_importance_{model_type}.csv"),index=False,encoding="utf-8")
    print(f"✔ Threshold ({metrics['threshold_policy']['policy']}): {thr:.4f}")
    print("✔ Métricas em reports/metrics.json")
    print(f"Val  → AUC: {metrics['val']['AUC']:.4f} | PR-AUC: {metrics['val']['PR_AUC']:.4f} | F1: {metrics['val']['F1']:.4f}")
    print(f"Test → AUC: {metrics['test']['AUC']:.4f} | PR-AUC: {metrics['test']['PR_AUC']:.4f} | F1: {metrics['test']['F1']:.4f}")
    top8=list(pd.Series(metrics["feature_importance"]).sort_values(ascending=False).head(8).index); print("Top features:",top8)

    mini_val=per_100_report(yva,p_va,thr); mini_tst=per_100_report(yte,p_te,thr)
    (REPORTS_DIR/"mini_report_val.json").write_text(json.dumps(mini_val,indent=2,ensure_ascii=False),encoding="utf-8")
    (REPORTS_DIR/"mini_report_test.json").write_text(json.dumps(mini_tst,indent=2,ensure_ascii=False),encoding="utf-8")
    def _fmt(tag,r): return f"{tag}: a cada 100 voos → atrasos reais: {r['base_rate_delay_per_100']:.1f} | alertas: {r['alerts_per_100']:.1f} (TP {r['tp_per_100']:.1f}, FP {r['fp_per_100']:.1f}, FN {r['fn_per_100']:.1f}) | precisão {r['precision']:.2f}, recall {r['recall']:.2f}"
    print("\n— Mini-relatórios (por 100 voos) —"); print(_fmt("VAL",mini_val)); print(_fmt("TEST",mini_tst))

    rows=[build_perf_row("TRAIN",ytr,p_tr,thr),build_perf_row("VAL",yva,p_va,thr),build_perf_row("TEST",yte,p_te,thr)]
    print_and_save_summary_table(rows)

    _print_and_save_quick_summary(metrics, mini_val, mini_tst)

    if args.prune_below is not None and args.model!="catboost":
        print("✂️  Prune ignorado para RF. Use --model catboost para podar por importância.")
    if args.prune_below is not None and args.model=="catboost":
        imp_dict=feature_importances(model,cols); to_drop=sorted([c for c,v in imp_dict.items() if float(v)<args.prune_below])
        if to_drop:
            print(f"\n✂️  Prune < {args.prune_below:.4f}: removendo {len(to_drop)} features.")
            tr2=tr.drop(columns=[c for c in to_drop if c in tr.columns],errors="ignore")
            va2=va.drop(columns=[c for c in to_drop if c in va.columns],errors="ignore")
            te2=te.drop(columns=[c for c in to_drop if c in te.columns],errors="ignore")
            cat_cols2,num_cols2=infer_feature_columns(tr2)
            Xtr2,ytr2,cols2,cat_idx2=build_Xy(tr2,cat_cols2,num_cols2)
            Xva2,yva2,_,_=build_Xy(va2,cat_cols2,num_cols2)
            Xte2,yte2,_,_=build_Xy(te2,cat_cols2,num_cols2)
            cat_dev=resolve_catboost_device(args.device)
            model2=train_catboost(Xtr2,ytr2,Xva2,yva2,cols2,cat_idx2,args,cat_dev)
            p_tr2=model2.predict_proba(Xtr2)[:,1]; p_va2=model2.predict_proba(Xva2)[:,1]; p_te2=model2.predict_proba(Xte2)[:,1]
            thr2,thr_info2=pick_threshold(yva2,p_va2,policy=args.th_policy,beta=args.beta,min_precision=args.min_precision,min_recall=args.min_recall)
            metrics_pruned={"prune_below":args.prune_below,"dropped_features":to_drop,"sizes":{"train":int(len(tr2)),"val":int(len(va2)),"test":int(len(te2))},
                            "threshold":thr2,"threshold_policy":thr_info2,
                            "train":eval_at_threshold(ytr2,p_tr2,thr2),"val":eval_at_threshold(yva2,p_va2,thr2),"test":eval_at_threshold(yte2,p_te2,thr2),
                            "features":cols2,"categorical":cat_cols2,"numerical":num_cols2,"feature_importance":feature_importances(model2,cols2),
                            "model_type":"catboost","params":{"pruned":True}|metrics["params"]}
            (REPORTS_DIR/"metrics_pruned.json").write_text(json.dumps(metrics_pruned,indent=2,ensure_ascii=False),encoding="utf-8")
            pd.DataFrame({"feature":cols2,"importance":[metrics_pruned["feature_importance"][c] for c in cols2]}).sort_values("importance",ascending=False).to_csv(REPORTS_DIR/"feature_importance_catboost_pruned.csv",index=False,encoding="utf-8")
            out_path2=MODELS_DIR/"catboost_delay_pruned.cbm"; model2.save_model(str(out_path2)); print(f"✔ Modelo podado salvo em {out_path2}")
        else:
            print(f"✂️  Prune: nenhuma feature com importância < {args.prune_below:.4f}")

if __name__=="__main__": main()
