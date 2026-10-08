import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import (METADATA_PATH, MODEL_PATH, cargar_metadatos, cargar_modelo,  # noqa: E402
                   predecir_con_intervalo, rendimiento_medio)


def test_rendimiento_medio_usa_el_guardado():
    assert rendimiento_medio({"rendimiento_medio_train": 3.9}) == 3.9


def test_rendimiento_medio_cae_a_la_mediana():
    meta = {"rendimiento_percentiles": {"percentil": [0, 50, 100], "valor": [1.0, 3.8, 6.0]}}
    assert rendimiento_medio(meta) == 3.8


requiere_modelo = pytest.mark.skipif(
    not (MODEL_PATH.exists() and METADATA_PATH.exists()),
    reason="Falta models/: ejecuta el notebook y copia los artefactos.",
)


@requiere_modelo
def test_prediccion_completa_e_intervalo_coherente():
    modelo, meta = cargar_modelo(), cargar_metadatos()
    r = predecir_con_intervalo(modelo, meta, {
        "N": 80, "P": 60, "K": 100, "temperature": 24, "humidity": 60,
        "ph": 6.5, "rainfall": 170, "soil_type": meta["tipos_de_suelo"][0]})
    assert np.isfinite(r["prediccion"])
    assert 0 <= r["inferior"] <= r["prediccion"] <= r["superior"]


@requiere_modelo
def test_prediccion_con_todo_faltante():
    modelo, meta = cargar_modelo(), cargar_metadatos()
    r = predecir_con_intervalo(modelo, meta, {c: np.nan for c in
                                              ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_type"]})
    assert np.isfinite(r["prediccion"])
