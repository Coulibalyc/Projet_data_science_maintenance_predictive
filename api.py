"""
API REST — Maintenance Prédictive Industrielle v2.0
Prédiction de la Durée de Vie Restante (RUL)

Lancer : uvicorn api:app --reload --port 8000
Docs   : http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import joblib, numpy as np, pandas as pd, os, time
from datetime import datetime

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model_pipeline.pkl")
try:
    pipeline        = joblib.load(MODEL_PATH)
    MODEL_LOADED    = True
    MODEL_LOADED_AT = datetime.utcnow().isoformat()
    MODEL_NAME      = type(pipeline.named_steps["model"]).__name__
except Exception as e:
    pipeline = None; MODEL_LOADED = False; MODEL_LOADED_AT = None; MODEL_NAME = "unknown"
    print(f"[WARN] Modèle non chargé : {e}")

app = FastAPI(
    title="API Maintenance Prédictive — RUL",
    description="Service d'inférence RUL — Random Forest (R²=0.955, MAE=3.077h)",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MACHINE_TYPES   = {"CNC", "Pump", "Compressor", "Robotic Arm"}
OPERATING_MODES = {"idle", "normal", "peak"}

class SensorInput(BaseModel):
    machine_type            : str            = Field(..., examples=["CNC"])
    vibration_rms           : float          = Field(..., ge=0, le=20,    examples=[1.2])
    temperature_motor       : float          = Field(..., ge=0, le=200,   examples=[65.0])
    current_phase_avg       : Optional[float]= Field(None, ge=0, le=50,  examples=[5.5])
    pressure_level          : Optional[float]= Field(None, ge=0, le=200, examples=[22.0])
    rpm                     : Optional[float]= Field(None, ge=0, le=5000,examples=[900.0])
    operating_mode          : str            = Field(..., examples=["normal"])
    hours_since_maintenance : float          = Field(..., ge=0, le=5000,  examples=[250.0])
    ambient_temp            : float          = Field(..., ge=-20, le=60,  examples=[22.0])
    cumcount                : Optional[int]  = Field(None, ge=0,
        description="Nb mesures depuis début du cycle. Défaut=500 si non fourni.",
        examples=[500])

    @field_validator("machine_type")
    @classmethod
    def check_machine(cls, v):
        if v not in MACHINE_TYPES:
            raise ValueError(f"machine_type doit être parmi {MACHINE_TYPES}")
        return v

    @field_validator("operating_mode")
    @classmethod
    def check_mode(cls, v):
        if v not in OPERATING_MODES:
            raise ValueError(f"operating_mode doit être parmi {OPERATING_MODES}")
        return v

class PredictionResponse(BaseModel):
    rul_hours         : float
    alert_level       : str
    alert_message     : str
    model_name        : str
    inference_time_ms : float
    features_used     : dict

class BatchResponse(BaseModel):
    predictions    : List[PredictionResponse]
    total_machines : int
    critical_count : int
    warning_count  : int
    ok_count       : int

class HealthResponse(BaseModel):
    status      : str
    model_loaded: bool
    model_name  : str
    loaded_at   : Optional[str]
    timestamp   : str
    metrics     : dict

class ModelInfoResponse(BaseModel):
    model_name          : str
    model_type          : str
    input_features      : list
    target              : str
    performance         : dict
    feature_engineering : dict


def build_features(data: SensorInput) -> pd.DataFrame:
    vib, temp, hours = data.vibration_rms, data.temperature_motor, data.hours_since_maintenance
    cc = data.cumcount if data.cumcount is not None else 500
    return pd.DataFrame([{
        "vibration_rms"          : vib,
        "temperature_motor"      : temp,
        "current_phase_avg"      : data.current_phase_avg  or 6.0,
        "pressure_level"         : data.pressure_level     or 35.0,
        "rpm"                    : data.rpm                or 900.0,
        "hours_since_maintenance": hours,
        "ambient_temp"           : data.ambient_temp,
        "cumcount"               : cc,
        "hours_sq"               : hours ** 2,
        "vib_x_temp"             : vib * temp,
        "vib_x_hours"            : vib * hours,
        "temp_x_hours"           : temp * hours,
    }])

def get_alert(rul: float):
    if rul < 10:
        return "critical", f"ALERTE CRITIQUE — {rul:.1f}h restantes. Intervention immédiate."
    elif rul < 30:
        return "warning",  f"ALERTE MODÉRÉE — {rul:.1f}h restantes. Planifier sous 48h."
    return "ok", f"État normal — {rul:.1f}h estimées avant maintenance."


@app.get("/", tags=["Info"])
def root():
    return {"message": "API Maintenance Prédictive RUL v2.0", "docs": "/docs",
            "endpoints": ["GET /health", "GET /model-info", "POST /predict", "POST /predict/batch"]}

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    return HealthResponse(
        status="ok" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED, model_name=MODEL_NAME,
        loaded_at=MODEL_LOADED_AT, timestamp=datetime.utcnow().isoformat(),
        metrics={"R2_test": 0.955, "MAE_test": 3.077, "RMSE_test": 5.575, "CV_MAE": "3.097 ± 0.073"}
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")
    return ModelInfoResponse(
        model_name="Random Forest Regressor",
        model_type="Régression — sklearn RandomForestRegressor (300 arbres)",
        input_features=["vibration_rms","temperature_motor","current_phase_avg","pressure_level",
                        "rpm","hours_since_maintenance","ambient_temp","cumcount",
                        "hours_sq","vib_x_temp","vib_x_hours","temp_x_hours"],
        target="rul_hours (Remaining Useful Life en heures)",
        performance={"R2_test": 0.955, "MAE_test": "3.077h", "RMSE_test": "5.575h",
                     "CV_MAE": "3.097h ± 0.073h (5-fold)", "baseline_R2": 0.671,
                     "gain_feature_engineering": "+42% de R²"},
        feature_engineering={
            "cumcount"    : "Nb mesures depuis début du cycle — proxy du vieillissement (40% importance)",
            "hours_sq"    : "hours² — relation non-linéaire avec le temps (23% importance)",
            "vib_x_temp"  : "vibration × température — stress thermomécanique",
            "vib_x_hours" : "vibration × heures — vibration cumulée",
            "temp_x_hours": "température × heures — chaleur cumulée",
        }
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(data: SensorInput):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")
    input_df = build_features(data)
    t0 = time.perf_counter()
    try:
        raw = pipeline.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inférence : {e}")
    ms  = (time.perf_counter() - t0) * 1000
    rul = float(max(0.0, round(raw, 2)))
    alert_level, alert_message = get_alert(rul)
    return PredictionResponse(rul_hours=rul, alert_level=alert_level,
        alert_message=alert_message, model_name=MODEL_NAME,
        inference_time_ms=round(ms, 3), features_used=input_df.iloc[0].to_dict())

@app.post("/predict/batch", response_model=BatchResponse, tags=["Prédiction"])
def predict_batch(machines: List[SensorInput]):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")
    if len(machines) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 machines par batch.")
    preds = []
    for data in machines:
        df  = build_features(data)
        t0  = time.perf_counter()
        raw = pipeline.predict(df)[0]
        ms  = (time.perf_counter() - t0) * 1000
        rul = float(max(0.0, round(raw, 2)))
        al, am = get_alert(rul)
        preds.append(PredictionResponse(rul_hours=rul, alert_level=al, alert_message=am,
            model_name=MODEL_NAME, inference_time_ms=round(ms,3),
            features_used=df.iloc[0].to_dict()))
    return BatchResponse(predictions=preds, total_machines=len(preds),
        critical_count=sum(1 for p in preds if p.alert_level=="critical"),
        warning_count =sum(1 for p in preds if p.alert_level=="warning"),
        ok_count      =sum(1 for p in preds if p.alert_level=="ok"))
