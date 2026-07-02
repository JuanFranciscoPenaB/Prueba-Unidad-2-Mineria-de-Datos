import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import io

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(page_title="Dashboard Salud - Big Data", layout="wide")

st.title("📊 Dashboard de Datos de Salud de Pacientes")

# =========================
# CARGA DE DATOS
# =========================
# Cambia el nombre del archivo si es necesario
df = pd.read_csv("healthcare_dataset.csv", sep=",")

# =========================
# MENÚ LATERAL (POWER BI STYLE)
# =========================
menu = st.sidebar.selectbox(
    "📌 Menú de Navegación",
    [
        "📂 Exploración de Datos",
        "🧹 Limpieza de Datos",
        "🔄 Transformación de Datos",
        "📊 Visualización",
        "📦 Dataset Final",
        "⬇️ Exportar Datos",
    ]
)

# =========================================================
# 📂 EXPLORACIÓN DE DATOS
# =========================================================
if menu == "📂 Exploración de Datos":

    st.header("Exploración de Datos")

    st.subheader("Vista general")
    st.dataframe(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Primeras filas")
        st.dataframe(df.head())
    with col2:
        st.subheader("Últimas filas")
        st.dataframe(df.tail())

    st.subheader("Info del DataFrame")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Estadísticas descriptivas")
    st.dataframe(df.describe())

    col3, col4, col5 = st.columns(3)
    with col3:
        st.subheader("Valores nulos")
        st.write(df.isnull().sum())
    with col4:
        st.subheader("Duplicados")
        st.write(f"{df.duplicated().sum()} filas duplicadas")
    with col5:
        st.subheader("Tipos de datos")
        st.dataframe(df.dtypes.astype(str))

# =========================================================
# 🧹 LIMPIEZA DE DATOS
# =========================================================
elif menu == "🧹 Limpieza de Datos":

    st.header("Limpieza de Datos")

    st.subheader("Antes de limpiar")
    st.write(f"Filas: {df.shape[0]} | Nulos totales: {df.isnull().sum().sum()}")
    st.dataframe(df)

    df_clean = df.copy()

    # Treatment_Plan tiene nulos -> se rellenan con "Sin Plan Asignado"
    df_clean["Treatment_Plan"] = df_clean["Treatment_Plan"].fillna("Sin Plan Asignado")

    # Eliminar duplicados si existieran
    df_clean = df_clean.drop_duplicates()

    st.subheader("Después de limpiar (Treatment_Plan nulo → 'Sin Plan Asignado')")
    st.dataframe(df_clean)

    st.subheader("Verificación de nulos post-limpieza")
    st.write(df_clean.isnull().sum())

# =========================================================
# 🔄 TRANSFORMACIÓN DE DATOS
# =========================================================
elif menu == "🔄 Transformación de Datos":

    st.header("Transformación de Datos")

    df_t = df.copy()
    df_t["Treatment_Plan"] = df_t["Treatment_Plan"].fillna("Sin Plan Asignado")

    # ------------------------
    # Normalización de BMI
    # ------------------------
    df_t["BMI_norm"] = (
        (df_t["BMI"] - df_t["BMI"].min()) /
        (df_t["BMI"].max() - df_t["BMI"].min())
    )

    # ------------------------
    # Z-score de Cholesterol_Level
    # ------------------------
    df_t["Colesterol_z"] = (
        (df_t["Cholesterol_Level"] - df_t["Cholesterol_Level"].mean()) /
        df_t["Cholesterol_Level"].std()
    )

    # ------------------------
    # Log de Blood_Pressure
    # ------------------------
    df_t["Presion_log"] = np.log1p(df_t["Blood_Pressure"])

    # ------------------------
    # Binning de BMI (categorías clínicas estándar)
    # ------------------------
    df_t["Categoria_BMI"] = pd.cut(
        df_t["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]
    )

    # ------------------------
    # Binning de edad
    # ------------------------
    df_t["Categoria_Edad"] = pd.cut(
        df_t["Age"],
        bins=[0, 6, 12, 20, 25, 60, 200],
        labels=["Infancia", "Niñez", "Adolescencia", "Juventud", "Adultez", "Ancianidad"],
        right=False
    )

    st.subheader("Datos transformados")
    st.dataframe(df_t)

    st.session_state["df_t"] = df_t

# =========================================================
# 📊 VISUALIZACIÓN (DASHBOARD DINÁMICO)
# =========================================================
elif menu == "📊 Visualización":

    st.header("Visualización Interactiva de Datos")

    df_v = df.copy()
    df_v["Treatment_Plan"] = df_v["Treatment_Plan"].fillna("Sin Plan Asignado")
    df_v["Categoria_BMI"] = pd.cut(
        df_v["BMI"], bins=[0, 18.5, 25, 30, 100],
        labels=["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]
    )

    # -------- FILTROS DINÁMICOS --------
    st.sidebar.subheader("🎛️ Filtros")
    generos = st.sidebar.multiselect(
        "Género", options=df_v["Gender"].unique(), default=list(df_v["Gender"].unique())
    )
    diagnosticos = st.sidebar.multiselect(
        "Diagnóstico", options=df_v["Diagnosis"].unique(), default=list(df_v["Diagnosis"].unique())
    )
    edad_min, edad_max = st.sidebar.slider(
        "Rango de edad", int(df_v["Age"].min()), int(df_v["Age"].max()),
        (int(df_v["Age"].min()), int(df_v["Age"].max()))
    )

    df_f = df_v[
        (df_v["Gender"].isin(generos)) &
        (df_v["Diagnosis"].isin(diagnosticos)) &
        (df_v["Age"].between(edad_min, edad_max))
    ]

    # -------- KPIs --------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pacientes filtrados", len(df_f))
    k2.metric("Edad promedio", f"{df_f['Age'].mean():.1f}" if len(df_f) else "-")
    k3.metric("Colesterol promedio", f"{df_f['Cholesterol_Level'].mean():.0f}" if len(df_f) else "-")
    k4.metric("BMI promedio", f"{df_f['BMI'].mean():.1f}" if len(df_f) else "-")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de edad")
        fig = px.histogram(df_f, x="Age", nbins=20, color="Gender",
                            title="Distribución de edad por género")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Pacientes por diagnóstico")
        fig = px.bar(df_f["Diagnosis"].value_counts().reset_index(),
                     x="Diagnosis", y="count", color="Diagnosis",
                     title="Cantidad de pacientes por diagnóstico")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Colesterol vs Presión Arterial")
        fig = px.scatter(df_f, x="Blood_Pressure", y="Cholesterol_Level",
                          color="Diagnosis", size="BMI", hover_data=["Age", "Gender"],
                          title="Relación Presión Arterial - Colesterol")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Outliers de BMI")
        fig = px.box(df_f, x="Diagnosis", y="BMI", color="Diagnosis",
                     title="Distribución de BMI por diagnóstico")
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Planes de tratamiento")
        fig = px.pie(df_f, names="Treatment_Plan", title="Distribución de planes de tratamiento")
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Categoría de BMI por género")
        fig = px.bar(df_f.groupby(["Categoria_BMI", "Gender"], observed=True).size().reset_index(name="count"),
                     x="Categoria_BMI", y="count", color="Gender", barmode="group",
                     title="Categoría de BMI segmentada por género")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mapa de calor de correlaciones")
    num_cols = ["Age", "Blood_Pressure", "Heart_Rate", "Cholesterol_Level", "BMI"]
    corr = df_f[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                     title="Correlación entre variables numéricas")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📦 DATASET FINAL
# =========================================================
elif menu == "📦 Dataset Final":

    st.header("Dataset Final Procesado")

    df_final = df.copy()
    df_final["Treatment_Plan"] = df_final["Treatment_Plan"].fillna("Sin Plan Asignado")

    df_final["BMI_norm"] = (
        (df_final["BMI"] - df_final["BMI"].min()) /
        (df_final["BMI"].max() - df_final["BMI"].min())
    )
    df_final["Colesterol_z"] = (
        (df_final["Cholesterol_Level"] - df_final["Cholesterol_Level"].mean()) /
        df_final["Cholesterol_Level"].std()
    )
    df_final["Categoria_BMI"] = pd.cut(
        df_final["BMI"], bins=[0, 18.5, 25, 30, 100],
        labels=["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]
    )

    st.dataframe(df_final)

    st.subheader("Estadísticas finales")
    st.dataframe(df_final.describe())

    st.session_state["df_final"] = df_final

# =========================================================
# ⬇️ EXPORTAR DATOS
# =========================================================
elif menu == "⬇️ Exportar Datos":

    st.header("Descarga de Datos Procesados")

    df_exp = df.copy()
    df_exp["Treatment_Plan"] = df_exp["Treatment_Plan"].fillna("Sin Plan Asignado")
    df_exp["BMI_norm"] = (
        (df_exp["BMI"] - df_exp["BMI"].min()) /
        (df_exp["BMI"].max() - df_exp["BMI"].min())
    )
    df_exp["Categoria_BMI"] = pd.cut(
        df_exp["BMI"], bins=[0, 18.5, 25, 30, 100],
        labels=["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]
    )

    st.dataframe(df_exp)

    csv = df_exp.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Descargar CSV",
        data=csv,
        file_name="healthcare_dataset_procesado.csv",
        mime="text/csv"
    )