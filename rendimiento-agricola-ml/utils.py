"""Lógica de predicción separada de la interfaz, para poder probarla sin Streamlit."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "modelo_rendimiento.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

COLUMNAS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_type"]


def cargar_modelo(ruta: Path = MODEL_PATH):
    return joblib.load(ruta)


def cargar_metadatos(ruta: Path = METADATA_PATH) -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def rendimiento_medio(metadata: dict) -> float:
    """Rendimiento medio de entrenamiento.

    Las versiones nuevas del cuaderno lo guardan en `rendimiento_medio_train`;
    si el metadata es antiguo se usa la mediana (percentil 50) como referencia.
    """
    if "rendimiento_medio_train" in metadata:
        return float(metadata["rendimiento_medio_train"])
    perc = metadata["rendimiento_percentiles"]
    return float(perc["valor"][perc["percentil"].index(50)])


def predecir_con_intervalo(modelo, metadata: dict, entrada: dict) -> dict:
    """Predice el rendimiento y devuelve el intervalo empírico del 95 %.

    `entrada` usa NaN / None para los datos que no se conocen: el pipeline los imputa.
    """
    fila = {c: entrada.get(c, np.nan) for c in COLUMNAS}
    df = pd.DataFrame([fila], columns=COLUMNAS)
    # Los valores ausentes en columnas numéricas deben ser float, no object
    for c in COLUMNAS[:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    pred = float(modelo.predict(df)[0])
    iq = metadata["intervalo_residuos"]
    return {
        "prediccion": pred,
        "inferior": max(0.0, pred + float(iq["q025"])),
        "superior": pred + float(iq["q975"]),
        "referencia": rendimiento_medio(metadata),
    }
