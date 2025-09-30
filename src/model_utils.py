import joblib
import pandas as pd

MODEL_PATH = "data/models/rf_cpu_delay.joblib"
_rf_bundle = joblib.load(MODEL_PATH)
rf = _rf_bundle["rf"]
cols = _rf_bundle["cols"]
cat_maps = _rf_bundle["cat_maps"]
medians = _rf_bundle["medians"]

def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    for c, m in cat_maps.items():
        if c in df.columns:
            df[c] = df[c].map(m).fillna(0).astype("int32")
    for c in medians:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(medians[c]).astype("float32")
    return df[cols]

def predict_delay(data: dict):
    X = preprocess_input(data)
    prob = rf.predict_proba(X)[0, 1]
    return prob
