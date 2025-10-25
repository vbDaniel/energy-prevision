from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
import pandas as pd
import joblib
import holidays

# -----------------------
# Configuração do App
# -----------------------
app = FastAPI(title="Energy Prevision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar diretórios estáticos para servir artefatos diretamente
if os.path.isdir("relatorios"):
    app.mount("/static/relatorios", StaticFiles(directory="relatorios"), name="relatorios")
if os.path.isdir("identificador"):
    app.mount("/static/identificador", StaticFiles(directory="identificador"), name="identificador")
if os.path.isdir("models"):
    app.mount("/static/models", StaticFiles(directory="models"), name="models")

# -----------------------
# Artefatos (carregamento lazy)
# -----------------------
ARTIFACTS = {
    "energy_model_paths": [
        "models/gru_energy_model_best.keras",
        "gru_energy_model_best.keras",
        "models/gru_energy_model_best.h5",
        "gru_energy_model_best.h5",
        "models/gru_energy_model.h5",
        "gru_energy_model.h5",
    ],
    "energy_scaler_path": "scaler_energy.pkl",
    "causas_gru_model_path": "models/classificador_causas_gru_best.keras",
    "causas_gru_scaler_path": "models/classificador_causas_gru_scaler.pkl",
    "picos_rf_model_path": "models/classificador_picos.joblib",
    "picos_rf_scaler_path": "models/classificador_picos_scaler.joblib",
}

_loaded = {
    "energy_model": None,
    "energy_scaler": None,
    "causas_gru_model": None,
    "causas_gru_scaler": None,
    "picos_rf_model": None,
    "picos_rf_scaler": None,
}

CAUSES = ["cozinha", "lavanderia", "aquecedor_ar", "outros"]
ID_TO_CAUSE = {i: c for i, c in enumerate(CAUSES)}

# -----------------------
# Schemas de requisição
# -----------------------
class EnergyRecord(BaseModel):
    datetime: Optional[str] = None
    Global_active_power: float
    temp: Optional[float] = None
    holiday: Optional[int] = None
    hour: Optional[int] = None
    dayofweek: Optional[int] = None
    month: Optional[int] = None

class EnergyPredictPayload(BaseModel):
    records: List[EnergyRecord]
    lookback: Optional[int] = 24 * 7

class SequenceRecord(BaseModel):
    data: Dict[str, Any]

class SequencePredictPayload(BaseModel):
    records: List[SequenceRecord]
    lookback: Optional[int] = 24 * 7

class PeaksClassifyPayload(BaseModel):
    records: List[SequenceRecord]

# -----------------------
# Utilitários
# -----------------------

def load_energy_model():
    # Importa tensorflow apenas quando necessário
    from tensorflow.keras.models import load_model as tf_load_model
    if _loaded["energy_model"] is None:
        for p in ARTIFACTS["energy_model_paths"]:
            if os.path.exists(p):
                try:
                    _loaded["energy_model"] = tf_load_model(p)
                    break
                except Exception:
                    continue
        if _loaded["energy_model"] is None:
            raise FileNotFoundError("Modelo de energia não encontrado em nenhum dos caminhos conhecidos.")
    return _loaded["energy_model"]


def load_energy_scaler():
    if _loaded["energy_scaler"] is None:
        path = ARTIFACTS["energy_scaler_path"]
        if not os.path.exists(path):
            raise FileNotFoundError("Scaler de energia não encontrado: scaler_energy.pkl")
        _loaded["energy_scaler"] = joblib.load(path)
    return _loaded["energy_scaler"]


def load_causas_gru():
    from tensorflow.keras.models import load_model as tf_load_model
    if _loaded["causas_gru_model"] is None:
        p = ARTIFACTS["causas_gru_model_path"]
        if not os.path.exists(p):
            raise FileNotFoundError("Modelo GRU de causas não encontrado: classificador_causas_gru_best.keras")
        _loaded["causas_gru_model"] = tf_load_model(p)
    if _loaded["causas_gru_scaler"] is None:
        sp = ARTIFACTS["causas_gru_scaler_path"]
        if not os.path.exists(sp):
            raise FileNotFoundError("Scaler GRU de causas não encontrado: classificador_causas_gru_scaler.pkl")
        _loaded["causas_gru_scaler"] = joblib.load(sp)
    return _loaded["causas_gru_model"], _loaded["causas_gru_scaler"]


def load_picos_rf():
    if _loaded["picos_rf_model"] is None:
        p = ARTIFACTS["picos_rf_model_path"]
        if not os.path.exists(p):
            raise FileNotFoundError("Modelo RandomForest de picos não encontrado: classificador_picos.joblib")
        _loaded["picos_rf_model"] = joblib.load(p)
    if _loaded["picos_rf_scaler"] is None:
        sp = ARTIFACTS["picos_rf_scaler_path"]
        if not os.path.exists(sp):
            raise FileNotFoundError("Scaler de picos não encontrado: classificador_picos_scaler.joblib")
        _loaded["picos_rf_scaler"] = joblib.load(sp)
    return _loaded["picos_rf_model"], _loaded["picos_rf_scaler"]


def ensure_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        df["hour"] = df.get("hour", pd.Series(index=df.index))
        df["dayofweek"] = df.get("dayofweek", pd.Series(index=df.index))
        df["month"] = df.get("month", pd.Series(index=df.index))
        df["holiday"] = df.get("holiday", pd.Series(index=df.index))

        df.loc[df["hour"].isna(), "hour"] = dt.dt.hour
        df.loc[df["dayofweek"].isna(), "dayofweek"] = dt.dt.weekday
        df.loc[df["month"].isna(), "month"] = dt.dt.month
        years = sorted(set(dt.dt.year.dropna().astype(int).tolist()))
        if years:
            fr_holidays = holidays.France(years=years)
            fr_holiday_dates = set(fr_holidays)
            df.loc[df["holiday"].isna(), "holiday"] = dt.dt.date.isin(fr_holiday_dates).astype(int)
        else:
            df.loc[df["holiday"].isna(), "holiday"] = 0
    return df


def timeseries_dataset_from_df(df_scaled: pd.DataFrame, target_col: str, lookback: int, batch_size: int = 64):
    import tensorflow as tf
    arr = df_scaled.values
    targets = df_scaled[target_col].values[lookback - 1:]
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=arr,
        targets=targets,
        sequence_length=lookback,
        sequence_stride=1,
        shuffle=False,
        batch_size=batch_size,
    )
    return ds, targets


def invert_scale_predictions(y_scaled: np.ndarray, ref_df_scaled: pd.DataFrame, scaler) -> np.ndarray:
    zeros = np.zeros((len(y_scaled), ref_df_scaled.shape[1]), dtype=np.float32)
    zeros[:, 0] = np.array(y_scaled).ravel()
    inv = scaler.inverse_transform(zeros)[:, 0]
    return inv

# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/artifacts")
async def artifacts():
    relatorios_dir = os.path.join("relatorios")
    identificador_dir = os.path.join("identificador")
    models_dir = os.path.join("models")
    def list_dir_safe(d):
        try:
            return sorted(os.listdir(d)) if os.path.exists(d) else []
        except Exception:
            return []
    return {
        "relatorios": list_dir_safe(relatorios_dir),
        "identificador": list_dir_safe(identificador_dir),
        "models": list_dir_safe(models_dir),
    }


@app.get("/reports/previsoes-energia")
async def get_previsoes_energia():
    path = os.path.join("relatorios", "previsoes_energia.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Arquivo de previsões não encontrado")
    df = pd.read_csv(path)
    return {
        "count": len(df),
        "rows": df.to_dict(orient="records"),
    }


@app.get("/identificador/picos-causas")
async def get_picos_causas_previstas():
    path = os.path.join("identificador", "picos_causas_previstas.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Arquivo de picos causas não encontrado")
    df = pd.read_csv(path)
    return {
        "count": len(df),
        "rows": df.to_dict(orient="records"),
    }


@app.get("/identificador/resultados-picos")
async def get_resultados_picos():
    path = os.path.join("identificador", "resultados_picos.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Arquivo resultados_picos.json não encontrado")
    with open(path, "r") as f:
        data = json.load(f)
    return data


@app.post("/predict/energy")
async def predict_energy(payload: EnergyPredictPayload):
    scaler = load_energy_scaler()
    model = load_energy_model()

    features = ["Global_active_power", "temp", "holiday", "hour", "dayofweek", "month"]

    df = pd.DataFrame([r.model_dump() for r in payload.records])
    df = ensure_time_features(df)

    for col in features:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Coluna ausente para previsão: {col}")

    try:
        df_scaled_arr = scaler.transform(df[features])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao aplicar scaler: {e}")
    df_scaled = pd.DataFrame(df_scaled_arr, columns=features)

    lookback = payload.lookback or (24 * 7)
    if len(df_scaled) < lookback:
        raise HTTPException(status_code=400, detail=f"Dados insuficientes: lookback={lookback}, registros={len(df_scaled)}")

    ds, targets = timeseries_dataset_from_df(df_scaled, target_col="Global_active_power", lookback=lookback)

    # Importa tensorflow apenas aqui
    import tensorflow as tf
    y_pred_scaled = model.predict(ds)

    y_true_inv = invert_scale_predictions(targets, df_scaled, scaler)
    y_pred_inv = invert_scale_predictions(y_pred_scaled, df_scaled, scaler)

    if "datetime" in df.columns:
        time_index = pd.to_datetime(df["datetime"]).iloc[lookback - 1: lookback - 1 + len(y_true_inv)]
        datetimes = time_index.astype(str).tolist()
    else:
        datetimes = [None] * len(y_true_inv)

    return {
        "lookback": lookback,
        "count": len(y_true_inv),
        "predictions": [
            {
                "datetime": datetimes[i],
                "real_kW": float(y_true_inv[i]),
                "previsto_kW": float(y_pred_inv[i]),
                "residuo_kW": float(y_true_inv[i] - y_pred_inv[i]),
            }
            for i in range(len(y_true_inv))
        ],
    }


@app.post("/classify/causas-gru")
async def classify_causas_gru(payload: SequencePredictPayload):
    model, scaler = load_causas_gru()

    if not hasattr(scaler, "feature_names_in_"):
        raise HTTPException(status_code=500, detail="Scaler de causas não possui feature_names_in_")
    features = list(scaler.feature_names_in_)

    df = pd.DataFrame([rec.data for rec in payload.records])
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Colunas ausentes para classificação GRU de causas: {missing}")

    X_scaled = scaler.transform(df[features])
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    lookback = payload.lookback or (24 * 7)
    if len(X_scaled_df) < lookback:
        raise HTTPException(status_code=400, detail=f"Dados insuficientes: lookback={lookback}, registros={len(X_scaled_df)})")

    import tensorflow as tf
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=X_scaled_df.values,
        targets=None,
        sequence_length=lookback,
        sequence_stride=1,
        shuffle=False,
        batch_size=64,
    )

    y_pred = model.predict(ds)
    y_idx = np.argmax(y_pred, axis=1)
    y_labels = [ID_TO_CAUSE.get(int(i), "outros") for i in y_idx]

    return {
        "lookback": lookback,
        "count": len(y_labels),
        "predictions": [
            {
                "index": i,
                "cause_id": int(y_idx[i]),
                "cause_label": y_labels[i],
                "probs": [float(p) for p in y_pred[i].tolist()],
            }
            for i in range(len(y_labels))
        ],
    }


@app.post("/classify/picos-rf")
async def classify_picos_rf(payload: PeaksClassifyPayload):
    model, scaler = load_picos_rf()

    if not hasattr(scaler, "feature_names_in_"):
        raise HTTPException(status_code=500, detail="Scaler de picos não possui feature_names_in_")
    features = list(scaler.feature_names_in_)

    df = pd.DataFrame([rec.data for rec in payload.records])
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Colunas ausentes para classificação de picos: {missing}")

    X_scaled = scaler.transform(df[features])
    y_pred = model.predict(X_scaled)
    y_labels = [ID_TO_CAUSE.get(int(i), "outros") for i in y_pred]

    return {
        "count": len(y_labels),
        "predictions": [
            {
                "index": i,
                "cause_id": int(y_pred[i]),
                "cause_label": y_labels[i],
            }
            for i in range(len(y_labels))
        ],
    }


@app.on_event("startup")
async def startup_event():
    try:
        load_energy_scaler()
    except Exception:
        pass
    # Não pré-carregamos modelos pesados por padrão; serão carregados sob demanda.