"""
Exporte toutes les données nécessaires aux dashboards HTML/CSS/JS.
Génère : data/dashboard_data.json

Lancer avant d'ouvrir les dashboards :
    python3 export_data.py
"""

import json, os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

print("Chargement des données...")
df = pd.read_csv("predictive_maintenance_v3.csv", parse_dates=["timestamp"])
df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

# Feature engineering
df["cumcount"]     = df.groupby("machine_id").cumcount()
df["hours_sq"]     = df["hours_since_maintenance"] ** 2
df["vib_x_temp"]   = df["vibration_rms"]     * df["temperature_motor"]
df["vib_x_hours"]  = df["vibration_rms"]     * df["hours_since_maintenance"]
df["temp_x_hours"] = df["temperature_motor"] * df["hours_since_maintenance"]

FEATURES = [
    "vibration_rms", "temperature_motor", "current_phase_avg",
    "pressure_level", "rpm", "hours_since_maintenance",
    "ambient_temp", "cumcount", "hours_sq",
    "vib_x_temp", "vib_x_hours", "temp_x_hours",
]

# ── Linear regression as JS approximation ────────────────────────────────────
print("Entraînement du modèle linéaire (approximation JS)...")
mask = df["rul_hours"].notna() & df[FEATURES].notna().all(axis=1)
X    = df.loc[mask, FEATURES].values
y    = df.loc[mask, "rul_hours"].values

imp  = SimpleImputer(strategy="median")
X_i  = imp.fit_transform(X)

lr   = LinearRegression().fit(X_i, y)
coefs = {f: float(c) for f, c in zip(FEATURES, lr.coef_)}
coefs["intercept"] = float(lr.intercept_)
lr_r2 = float(lr.score(X_i, y))
print(f"  LR R² (approximation) : {lr_r2:.3f}")

# ── Sample sensor data (20 machines, 300 pts each) ───────────────────────────
print("Export données capteurs...")
sample = {}
for mid in sorted(df["machine_id"].unique()):
    mdf = df[df["machine_id"] == mid].sort_values("timestamp").head(300)
    sample[str(mid)] = {
        "timestamps": mdf["timestamp"].dt.strftime("%Y-%m-%dT%H:%M").tolist(),
        "rul":         mdf["rul_hours"].round(2).tolist(),
        "vibration":   mdf["vibration_rms"].round(3).tolist(),
        "temperature": mdf["temperature_motor"].round(2).tolist(),
        "rpm":         mdf["rpm"].round(0).tolist(),
        "pressure":    mdf["pressure_level"].round(2).tolist(),
        "current":     mdf["current_phase_avg"].round(3).tolist(),
        "machine_type":str(mdf["machine_type"].iloc[0]),
    }

# ── Summary statistics ────────────────────────────────────────────────────────
NUM = ["vibration_rms","temperature_motor","current_phase_avg",
       "pressure_level","rpm","hours_since_maintenance","ambient_temp","rul_hours"]
stats = df[NUM].describe().round(3).to_dict()

# ── Missing values ────────────────────────────────────────────────────────────
mv     = df.isnull().sum()
mv_pct = (mv / len(df) * 100).round(2)
missing = {k: {"count": int(v), "pct": float(mv_pct[k])}
           for k, v in mv.items() if v > 0}

# ── Correlation matrix ────────────────────────────────────────────────────────
corr = df[NUM].corr().round(3)
corr_data = {
    "labels": NUM,
    "matrix": corr.values.round(3).tolist(),
    "rul_correlations": corr["rul_hours"].drop("rul_hours").round(3).to_dict(),
}

# ── RUL distribution ──────────────────────────────────────────────────────────
rul_vals = df["rul_hours"].dropna()
counts, bins = np.histogram(rul_vals, bins=40)
rul_hist = {"counts": counts.tolist(), "bins": bins.round(2).tolist(),
            "mean": float(rul_vals.mean()), "median": float(rul_vals.median()),
            "std":  float(rul_vals.std())}

# ── Sensor distributions by machine type ─────────────────────────────────────
sensor_by_type = {}
for s in ["vibration_rms", "temperature_motor", "rpm", "pressure_level", "current_phase_avg"]:
    sensor_by_type[s] = {}
    for mt in sorted(df["machine_type"].dropna().unique()):
        v = df[df["machine_type"] == mt][s].dropna()
        sensor_by_type[s][mt] = {
            "min": float(v.min()), "q1": float(v.quantile(.25)),
            "median": float(v.median()), "q3": float(v.quantile(.75)),
            "max": float(v.max()), "mean": float(v.mean()),
        }

# ── RUL by machine type ───────────────────────────────────────────────────────
rul_by_type = {}
for mt in sorted(df["machine_type"].dropna().unique()):
    v = df[df["machine_type"] == mt]["rul_hours"].dropna()
    rul_by_type[mt] = {"mean": float(v.mean()), "median": float(v.median()),
                       "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}

# ── RUL by operating mode ─────────────────────────────────────────────────────
rul_by_mode = {}
for mode in sorted(df["operating_mode"].dropna().unique()):
    v = df[df["operating_mode"] == mode]["rul_hours"].dropna()
    rul_by_mode[mode] = {"mean": float(v.mean()), "median": float(v.median()),
                          "q1": float(v.quantile(.25)), "q3": float(v.quantile(.75)),
                          "min": float(v.min()), "max": float(v.max())}

# ── Model results (from notebook) ────────────────────────────────────────────
model_results = {
    "models":  ["Ridge Regression", "Random Forest", "XGBoost", "MLP"],
    "mae":     [16.288, 3.077, 4.402, 4.911],
    "rmse":    [20.468, 5.575, 6.542, 7.604],
    "r2":      [0.394,  0.955, 0.938, 0.916],
    "cv_mae":  [16.603, 3.097, 4.430, 4.943],
    "cv_std":  [0.114,  0.073, 0.114, 0.200],
}
feature_importance = {
    "features":   ["cumcount","hours_sq","hours_since_maintenance","rpm",
                   "temperature_motor","machine_type_enc","current_phase_avg",
                   "vib_x_temp","pressure_level","vibration_rms"],
    "importance": [0.400,0.235,0.080,0.038,0.027,0.020,0.017,0.013,0.012,0.010],
    "is_new":     [True,True,False,False,False,False,False,True,False,False],
}

# ── Dataset info ──────────────────────────────────────────────────────────────
dataset_info = {
    "n_rows":    int(len(df)),
    "n_machines":int(df["machine_id"].nunique()),
    "n_days":    14,
    "n_vars":    15,
    "failure_24h": int(df["failure_within_24h"].sum()),
    "machine_types":    sorted(df["machine_type"].dropna().unique().tolist()),
    "operating_modes":  sorted(df["operating_mode"].dropna().unique().tolist()),
    "machine_type_counts": df["machine_type"].value_counts().to_dict(),
}

# ── Variable metadata ─────────────────────────────────────────────────────────
variables = [
    {"name":"timestamp",              "type":"datetime","status":"drop",  "desc":"Horodatage — non généralisable"},
    {"name":"machine_id",             "type":"int64",   "status":"drop",  "desc":"ID machine — non généralisable"},
    {"name":"machine_type",           "type":"object",  "status":"keep",  "desc":"Type de machine — encodé"},
    {"name":"vibration_rms",          "type":"float64", "status":"keep",  "desc":"Vibration RMS (g) — capteur clé"},
    {"name":"temperature_motor",      "type":"float64", "status":"keep",  "desc":"Température moteur (°C)"},
    {"name":"current_phase_avg",      "type":"float64", "status":"keep",  "desc":"Courant de phase moyen (A)"},
    {"name":"pressure_level",         "type":"float64", "status":"keep",  "desc":"Pression (bar)"},
    {"name":"rpm",                    "type":"float64", "status":"keep",  "desc":"Vitesse de rotation (RPM)"},
    {"name":"operating_mode",         "type":"object",  "status":"keep",  "desc":"Mode opératoire — encodé"},
    {"name":"hours_since_maintenance","type":"float64", "status":"keep",  "desc":"Heures depuis maintenance — très important"},
    {"name":"ambient_temp",           "type":"float64", "status":"keep",  "desc":"Température ambiante (°C)"},
    {"name":"rul_hours",              "type":"float64", "status":"target","desc":"🎯 VARIABLE CIBLE"},
    {"name":"failure_within_24h",     "type":"int64",   "status":"leak",  "desc":"⚠️ Data leakage — dérivé du RUL"},
    {"name":"failure_type",           "type":"object",  "status":"leak",  "desc":"⚠️ Data leakage — connu après panne"},
    {"name":"estimated_repair_cost",  "type":"int64",   "status":"leak",  "desc":"⚠️ Data leakage — corrélé à la panne"},
]

# ── Assemble & save ───────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
payload = {
    "lr_model":         {"coefs": coefs, "features": FEATURES, "r2": lr_r2},
    "sample_sensors":   sample,
    "stats":            stats,
    "missing":          missing,
    "correlation":      corr_data,
    "rul_hist":         rul_hist,
    "sensor_by_type":   sensor_by_type,
    "rul_by_type":      rul_by_type,
    "rul_by_mode":      rul_by_mode,
    "model_results":    model_results,
    "feature_importance": feature_importance,
    "dataset_info":     dataset_info,
    "variables":        variables,
}

out = "data/dashboard_data.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

size = os.path.getsize(out) / 1024
print(f"✓ Données exportées → {out}  ({size:.0f} Ko)")
print(f"  {dataset_info['n_rows']:,} enregistrements · {dataset_info['n_machines']} machines")

# ── Injection directe dans les HTML (évite les problèmes fetch/CORS) ──────────
import re

data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
data_script = f'<script id="dash-data">window.DASH_DATA={data_json};</script>'
pattern = re.compile(r'<script id="dash-data">.*?</script>', re.DOTALL)

for html in ["dashboard_client.html", "dashboard_technique.html"]:
    if not os.path.exists(html):
        print(f"  [skip] {html} introuvable")
        continue
    with open(html, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = pattern.sub(lambda _: data_script, content)
    if new_content == content:
        print(f"  [warn] placeholder <script id='dash-data'> absent de {html}")
    else:
        with open(html, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✓ Données injectées dans {html}")
