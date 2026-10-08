import warnings

import numpy as np
import pandas as pd
import streamlit as st

from utils import (METADATA_PATH, MODEL_PATH, cargar_metadatos, cargar_modelo,
                   predecir_con_intervalo)

st.set_page_config(page_title="Predicción de Rendimiento Agrícola", page_icon="🌾", layout="wide")


@st.cache_resource
def get_modelo():
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo = cargar_modelo()
    return modelo, [str(a.message) for a in avisos]


@st.cache_data
def get_metadatos():
    return cargar_metadatos()


# ---------------------------------------------------------------- carga segura
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.title("🌾 Predicción del Rendimiento Agrícola")
    st.error(
        "No se encontraron los archivos del modelo. Ejecuta el notebook "
        "`notebooks/Proyecto_ML_Rendimiento_Agricola.ipynb` y copia la carpeta `models/` "
        "(`modelo_rendimiento.joblib` y `metadata.json`) al repositorio."
    )
    st.stop()

modelo, avisos_carga = get_modelo()
metadata = get_metadatos()
nombre_modelo = metadata.get("modelo", "modelo")
rangos = metadata["rangos_entrenamiento"]

st.title("🌾 Predicción del Rendimiento Agrícola")
st.markdown(
    "Estima el rendimiento esperado en **toneladas por hectárea (t/ha)** a partir de las "
    f"condiciones del suelo y del clima, usando el modelo **{nombre_modelo}** entrenado en el proyecto."
)

if avisos_carga:
    st.warning(
        "El modelo se guardó con otra versión de scikit-learn que la instalada aquí. "
        "Instala la versión indicada en `requirements.txt` para evitar resultados inesperados."
    )

col_inputs, col_results = st.columns([1.2, 1])

# --------------------------------------------------------------------- entradas
ETIQUETAS = {
    "N": "Nitrógeno (N)", "P": "Fósforo (P)", "K": "Potasio (K)",
    "temperature": "Temperatura (°C)", "humidity": "Humedad (%)",
    "ph": "pH del suelo", "rainfall": "Lluvia (mm)",
}


def control_numerico(var: str, paso: float) -> float:
    """Slider con casilla para marcar el dato como desconocido (el pipeline lo imputa)."""
    info = rangos[var]
    c_slider, c_check = st.columns([3, 1])
    with c_check:
        usar = st.checkbox("Conocido", value=True, key=f"usar_{var}")
    with c_slider:
        valor = st.slider(
            f"{ETIQUETAS[var]} · mediana {info['mediana']:.1f}",
            min_value=float(info["min"]), max_value=float(info["max"]),
            value=float(info["mediana"]), step=paso,
            disabled=not usar, key=f"val_{var}",
        )
    return valor if usar else np.nan


entrada = {}
with col_inputs:
    st.subheader("🔬 Parámetros de la parcela")
    st.caption("Desmarca «Conocido» para simular un dato faltante: el modelo lo imputa con la mediana de entrenamiento.")
    tab_nutr, tab_clima, tab_suelo = st.tabs(["🧪 Nutrientes", "🌤️ Clima y pH", "🌱 Tipo de suelo"])

    with tab_nutr:
        for var in ["N", "P", "K"]:
            entrada[var] = control_numerico(var, 1.0)
    with tab_clima:
        for var, paso in [("temperature", 0.1), ("humidity", 0.5), ("ph", 0.1), ("rainfall", 1.0)]:
            entrada[var] = control_numerico(var, paso)
    with tab_suelo:
        c_sel, c_check = st.columns([3, 1])
        with c_check:
            usar_suelo = st.checkbox("Conocido", value=True, key="usar_soil_type")
        with c_sel:
            suelo = st.selectbox("Tipo de suelo", metadata["tipos_de_suelo"], disabled=not usar_suelo)
        entrada["soil_type"] = suelo if usar_suelo else np.nan
        st.caption("En este conjunto de datos el tipo de suelo no aportó información predictiva.")

# ------------------------------------------------------------------- resultados
with col_results:
    st.subheader("📊 Predicción")
    try:
        r = predecir_con_intervalo(modelo, metadata, entrada)
        st.metric("Rendimiento estimado", f"{r['prediccion']:.2f} t/ha",
                  delta=f"{r['prediccion'] - r['referencia']:+.2f} vs. media de entrenamiento")
        st.success(f"🎯 **Intervalo de predicción del 95 %:** {r['inferior']:.2f} a {r['superior']:.2f} t/ha")
        st.caption("Intervalo empírico calculado con los residuos de validación cruzada. "
                   "Es ancho porque el modelo orienta, pero no fija con precisión el rendimiento de una parcela.")
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo calcular la predicción: {e}")

    st.divider()
    with st.expander("💡 ¿Qué variables pesan en el modelo?"):
        imp = pd.DataFrame(
            [{"Variable": k, "Aumento del RMSE al desordenarla (t/ha)": v["media"]}
             for k, v in metadata["importancia_permutacion"].items()]
        ).sort_values("Aumento del RMSE al desordenarla (t/ha)", ascending=False)
        st.dataframe(imp, hide_index=True)
        st.caption("Casi toda la capacidad predictiva está en N, P y K.")

    with st.expander("📈 Desempeño en el conjunto de prueba"):
        mp, base = metadata["metricas_prueba"], metadata["linea_base_prueba"]
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{mp['R2']:.3f}")
        c2.metric("RMSE", f"{mp['RMSE']:.3f} t/ha")
        c3.metric("MAE", f"{mp['MAE']:.3f} t/ha")
        st.write(f"Mejora del error frente a predecir siempre la media: **{100 * (1 - mp['RMSE'] / base['RMSE']):.1f} %**.")
        st.caption("Validado con 5 pliegues dentro de entrenamiento y evaluado una sola vez en el 20 % de prueba.")

    with st.expander("🏁 Comparación de los 7 modelos"):
        comp = pd.DataFrame(metadata["comparacion_modelos"]).rename(columns={
            "modelo": "Modelo", "cv_rmse": "RMSE (CV)", "cv_r2": "R² (CV)",
            "test_r2": "R² (prueba)", "test_rmse": "RMSE (prueba)", "test_mae": "MAE (prueba)"})
        st.dataframe(comp.round(3), hide_index=True)
