import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from pathlib import Path

st.set_page_config(page_title="Aprobación - Predicción", page_icon="✅", layout="wide")

# ========= Carga del modelo (cache) =========
@st.cache_resource
def load_artifact(path: str = "modelo_aprobacion.joblib"):
    art = joblib.load(path)
    pipe = art["pipeline_trained"]
    req_cols = art["input_feature_names"]
    return art, pipe, req_cols

art, pipe, req_cols = load_artifact()

st.title("Predicción de Aprobación del Curso")
st.caption("Pipeline con imputación + OneHot + escalado + modelo (LogisticRegression o MLP, según tu artifact)")

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
        # intenta con UTF-8; si falla, intenta latin-1
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
    st.subheader("2) Vista previa de tus datos")
    st.dataframe(df_new.head(20), use_container_width=True)

    # --------- Validación de columnas ----------
    extra = [c for c in df_new.columns if c not in req_cols]
    faltan = [c for c in req_cols if c not in df_new.columns]

    col1, col2 = st.columns(2)
    with col1:
        if extra:
            st.warning(f"Columnas extra (serán ignoradas): {extra}")
        else:
            st.success("No hay columnas extra.")
    with col2:
        if faltan:
            st.error(f"FALTAN columnas requeridas: {faltan}")
        else:
            st.success("No faltan columnas requeridas.")

    # Si faltan columnas, bloqueamos el botón de predicción
    can_predict = len(faltan) == 0

    st.subheader("3) Ejecutar predicción")
    if st.button("Predecir", type="primary", disabled=not can_predict):
        # Mantener solo las columnas esperadas, en el orden correcto
        X = df_new.reindex(columns=req_cols)

        # Predicción
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)[:, 1]
            pred = (proba >= float(thr)).astype(int)
        else:
            # fallback si el modelo no tiene predict_proba
            pred = pipe.predict(X).astype(int)
            # probabilidad dummy (no disponible)
            proba = np.full(shape=len(pred), fill_value=np.nan)

        # Construir salida
        df_out = df_new.copy()
        df_out["proba_aprueba"] = proba
        df_out["pred_aprueba"] = pred

        # Métricas rápidas si el archivo trae la verdad (opcional)
        # Si el usuario incluyó 'Aprobo_Calculo', podemos calcular F1/ACC de referencia
        if "Aprobo_Calculo" in df_new.columns:
            try:
                y_true = df_new["Aprobo_Calculo"].astype(int)
                acc = (y_true == pred).mean()
                # F1 binario (manejo de división por cero)
                tp = ((y_true == 1) & (pred == 1)).sum()
                fp = ((y_true == 0) & (pred == 1)).sum()
                fn = ((y_true == 1) & (pred == 0)).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                st.info(f"**Métricas de referencia** (usando 'Aprobo_Calculo' si existe en tu archivo): "
                        f"ACC={acc:.3f} | PREC={prec:.3f} | REC={rec:.3f} | F1={f1:.3f}")
            except Exception:
                st.caption("No se pudieron calcular métricas de referencia (tipos/valores no válidos).")

        st.success("¡Predicción completada!")

        # Mostrar una muestra
        st.subheader("4) Resultados (muestra)")
        st.dataframe(df_out.head(50), use_container_width=True)

        # Botones de descarga
        def to_csv_bytes(df):
            return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        def to_excel_bytes(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="predicciones")
            return output.getvalue()

        st.download_button(
            label="⬇️ Descargar CSV",
            data=to_csv_bytes(df_out),
            file_name="predicciones.csv",
            mime="text/csv"
        )
        st.download_button(
            label="⬇️ Descargar Excel",
            data=to_excel_bytes(df_out),
            file_name="predicciones.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Sube un archivo CSV o XLSX para comenzar.")
