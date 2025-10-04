# app.py
import re
import unicodedata
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Aprobación - Predicción", page_icon="✅", layout="wide")

# ========= Utilidades de normalización de encabezados =========
def _norm_header(s: str) -> str:
    """Normaliza Unicode, reemplaza NBSP, colapsa espacios y quita bordes."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _soft_key(s: str) -> str:
    """Clave 'suave' para matchear encabezados independientemente de guiones/underscores/casing."""
    s = _norm_header(s)
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

# ========= Carga del modelo (cache) =========
@st.cache_resource
def load_artifact(path: str = "modelo_aprobacion.joblib"):
    art = joblib.load(path)
    pipe = art["pipeline_trained"]
    req_cols = art["input_feature_names"]
    # normalizamos solo para mostrar, pero mantenemos los nombres oficiales:
    req_cols = [ _norm_header(c) for c in req_cols ]
    return art, pipe, req_cols

art, pipe, req_cols = load_artifact()

st.title("Predicción de Aprobación del Curso")
st.caption("Pipeline con imputación + OneHot + escalado + modelo entrenado (del artifact)")

# ========= Sidebar: Configuración =========
st.sidebar.header("Configuración")
thr = st.sidebar.slider("Umbral (threshold) para clasificar", 0.0, 1.0, 0.5, 0.01)
st.sidebar.write(f"Umbral actual: **{thr:.2f}**")

# ========= Sección de carga =========
st.subheader("1) Carga tus datos nuevos (CSV o Excel)")
up = st.file_uploader("Selecciona un archivo .csv o .xlsx", type=["csv", "xlsx"])

def read_uploaded(f):
    if f is None:
        return None
    name = f.name.lower()
    if name.endswith(".csv"):
        # intenta UTF-8; si falla, cae a latin-1
        try:
            return pd.read_csv(f)
        except Exception:
            f.seek(0)
            return pd.read_csv(f, encoding="latin-1", sep=None, engine="python")
    if name.endswith(".xlsx"):
        return pd.read_excel(f)
    return None

df_new = read_uploaded(up)

# ========= Mostrar columnas esperadas =========
with st.expander("Ver columnas esperadas por el modelo"):
    st.write(req_cols)

if df_new is not None:
    # ---- Normaliza encabezados del archivo cargado ----
    df_new.columns = [_norm_header(c) for c in df_new.columns]

    st.subheader("2) Vista previa de tus datos")
    st.dataframe(df_new.head(20), use_container_width=True)

    # ---- Matching por clave suave: renombra columnas del archivo al nombre oficial esperado ----
    req_map = {_soft_key(c): c for c in req_cols}           # clave suave -> nombre oficial
    have_map = {_soft_key(c): c for c in df_new.columns}    # clave suave -> nombre presente

    # faltantes/extras según clave suave
    faltan_soft = [k for k in req_map.keys() if k not in have_map]
    extra_soft  = [k for k in have_map.keys() if k not in req_map]

    # renombrar a nombres oficiales
    rename_map = {have_map[k]: req_map[k] for k in req_map.keys() if k in have_map}
    df_new = df_new.rename(columns=rename_map)

    # construir listas legibles para mensajes
    faltan = [req_map[k] for k in faltan_soft]
    extra  = [have_map[k] for k in extra_soft]

    # ---- Mensajes de validación ----
    col1, col2 = st.columns(2)
    with col1:
        if extra:
            st.warning(f"Columnas extra (se ignorarán en el cálculo): {extra}")
        else:
            st.success("No hay columnas extra.")
    with col2:
        if faltan:
            st.error(f"FALTAN columnas requeridas: {faltan}")
        else:
            st.success("No faltan columnas requeridas.")

    # Si faltan columnas, no permitimos predecir
    can_predict = len(faltan) == 0

    st.subheader("3) Ejecutar predicción")
    if st.button("Predecir", type="primary", disabled=not can_predict):
        # Mantener solo las columnas que el modelo espera (en orden exacto)
        X = df_new.reindex(columns=req_cols)

        # Predicción
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)[:, 1]
            pred = (proba >= float(thr)).astype(int)
        else:
            pred = pipe.predict(X).astype(int)
            proba = np.full(shape=len(pred), fill_value=np.nan)

        # ========= SALIDA: SOLO columnas requeridas + predicciones =========
        base = X.copy()
        base["proba_aprueba"] = proba
        base["pred_aprueba"] = pred
        df_out = base

        st.success("¡Predicción completada!")

        # ---- Métricas de referencia si viene la verdad (opcional) ----
        if "Aprobo_Calculo" in df_new.columns:
            try:
                y_true = df_new["Aprobo_Calculo"].astype(int)
                acc = (y_true == pred).mean()
                tp = ((y_true == 1) & (pred == 1)).sum()
                fp = ((y_true == 0) & (pred == 1)).sum()
                fn = ((y_true == 1) & (pred == 0)).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                st.info(f"**Métricas de referencia** (si tu archivo trae 'Aprobo_Calculo'): "
                        f"ACC={acc:.3f} | PREC={prec:.3f} | REC={rec:.3f} | F1={f1:.3f}")
            except Exception:
                st.caption("No se pudieron calcular métricas de referencia (tipos/valores no válidos).")

        # ---- Mostrar muestra y descargas ----
        st.subheader("4) Resultados (muestra)")
        st.dataframe(df_out.head(50), use_container_width=True)

        def to_csv_bytes(df):
            return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        def to_excel_bytes(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="predicciones")
            return output.getvalue()

        st.download_button(
            label="⬇️ Descargar CSV (solo columnas del modelo + predicciones)",
            data=to_csv_bytes(df_out),
            file_name="predicciones.csv",
            mime="text/csv"
        )
        st.download_button(
            label="⬇️ Descargar Excel (solo columnas del modelo + predicciones)",
            data=to_excel_bytes(df_out),
            file_name="predicciones.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Sube un archivo CSV o XLSX para comenzar.")
