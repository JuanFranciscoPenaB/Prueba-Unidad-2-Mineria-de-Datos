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
df = pd.read_csv("healthcare_dataset.csv", sep=",")

# =========================
# RENOMBRADO DE COLUMNAS (inglés -> español)
# =========================
df = df.rename(columns={
    "Patient_ID": "ID_Paciente",
    "Age": "Edad",
    "Gender": "Genero",
    "Blood_Pressure": "Presion_Arterial",
    "Heart_Rate": "Ritmo_Cardiaco",
    "Cholesterol_Level": "Colesterol",
    "BMI": "IMC",
    "Diagnosis": "Diagnostico",
    "Treatment_Plan": "Plan_Tratamiento",
    "Follow_Up_Date": "Fecha_Seguimiento",
})

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

    # Plan_Tratamiento tiene nulos -> se rellenan con "Sin Plan Asignado"
    df_clean["Plan_Tratamiento"] = df_clean["Plan_Tratamiento"].fillna("Sin Plan Asignado")

    # Eliminar duplicados si existieran
    df_clean = df_clean.drop_duplicates()

    st.subheader("Después de limpiar (Plan_Tratamiento nulo → 'Sin Plan Asignado')")
    st.dataframe(df_clean)

    st.subheader("Verificación de nulos post-limpieza")
    st.write(df_clean.isnull().sum())

# =========================================================
# 🔄 TRANSFORMACIÓN DE DATOS
# =========================================================
elif menu == "🔄 Transformación de Datos":

    st.header("Transformación de Datos")

    df_t = df.copy()
    df_t["Plan_Tratamiento"] = df_t["Plan_Tratamiento"].fillna("Sin Plan Asignado")

    # ------------------------
    # Normalización de IMC
    # ------------------------
    df_t["IMC_norm"] = (
        (df_t["IMC"] - df_t["IMC"].min()) /
        (df_t["IMC"].max() - df_t["IMC"].min())
    )

    # ------------------------
    # Z-score de Colesterol
    # ------------------------
    df_t["Colesterol_z"] = (
        (df_t["Colesterol"] - df_t["Colesterol"].mean()) /
        df_t["Colesterol"].std()
    )

    # ------------------------
    # Log de Presion_Arterial
    # ------------------------
    df_t["Presion_log"] = np.log1p(df_t["Presion_Arterial"])

    # ------------------------
    # Binning de IMC (categorías clínicas estándar)
    # ------------------------
    df_t["Categoria_IMC"] = pd.cut(
        df_t["IMC"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]
    )

    # ------------------------
    # Binning de edad
    # ------------------------
    df_t["Categoria_Edad"] = pd.cut(
        df_t["Edad"],
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
    df_v["Plan_Tratamiento"] = df_v["Plan_Tratamiento"].fillna("Sin Plan Asignado")
    df_v["Categoria_IMC"] = pd.cut(
        df_v["IMC"], bins=[0, 18.5, 25, 30, 100],
        labels=["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]
    )

    # -------- FILTROS DINÁMICOS --------
    st.sidebar.subheader("🎛️ Filtros")
    generos = st.sidebar.multiselect(
        "Género", options=df_v["Genero"].unique(), default=list(df_v["Genero"].unique())
    )
    diagnosticos = st.sidebar.multiselect(
        "Diagnóstico", options=df_v["Diagnostico"].unique(), default=list(df_v["Diagnostico"].unique())
    )
    edad_min, edad_max = st.sidebar.slider(
        "Rango de edad", int(df_v["Edad"].min()), int(df_v["Edad"].max()),
        (int(df_v["Edad"].min()), int(df_v["Edad"].max()))
    )

    df_f = df_v[
        (df_v["Genero"].isin(generos)) &
        (df_v["Diagnostico"].isin(diagnosticos)) &
        (df_v["Edad"].between(edad_min, edad_max))
    ]

    # -------- KPIs --------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pacientes filtrados", len(df_f))
    k2.metric("Edad promedio", f"{df_f['Edad'].mean():.1f}" if len(df_f) else "-")
    k3.metric("Colesterol promedio", f"{df_f['Colesterol'].mean():.0f}" if len(df_f) else "-")
    k4.metric("IMC promedio", f"{df_f['IMC'].mean():.1f}" if len(df_f) else "-")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de edad")
        fig = px.histogram(df_f, x="Edad", nbins=20, color="Genero",
                            title="Distribución de edad por género")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Pacientes por diagnóstico")
        fig = px.bar(df_f["Diagnostico"].value_counts().reset_index(),
                     x="Diagnostico", y="count", color="Diagnostico",
                     title="Cantidad de pacientes por diagnóstico")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Colesterol vs Presión Arterial")
        fig = px.scatter(df_f, x="Presion_Arterial", y="Colesterol",
                          color="Diagnostico", size="IMC", hover_data=["Edad", "Genero"],
                          title="Relación Presión Arterial - Colesterol")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Outliers de IMC")
        fig = px.box(df_f, x="Diagnostico", y="IMC", color="Diagnostico",
                     title="Distribución de IMC por diagnóstico")
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Planes de tratamiento")
        fig = px.pie(df_f, names="Plan_Tratamiento", title="Distribución de planes de tratamiento")
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Categoría de IMC por género")
        fig = px.bar(df_f.groupby(["Categoria_IMC", "Genero"], observed=True).size().reset_index(name="count"),
                     x="Categoria_IMC", y="count", color="Genero", barmode="group",
                     title="Categoría de IMC segmentada por género")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mapa de calor de correlaciones")
    num_cols = ["Edad", "Presion_Arterial", "Ritmo_Cardiaco", "Colesterol", "IMC"]
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
    df_final["Plan_Tratamiento"] = df_final["Plan_Tratamiento"].fillna("Sin Plan Asignado")

    df_final["IMC_norm"] = (
        (df_final["IMC"] - df_final["IMC"].min()) /
        (df_final["IMC"].max() - df_final["IMC"].min())
    )
    df_final["Colesterol_z"] = (
        (df_final["Colesterol"] - df_final["Colesterol"].mean()) /
        df_final["Colesterol"].std()
    )
    df_final["Categoria_IMC"] = pd.cut(
        df_final["IMC"], bins=[0, 18.5, 25, 30, 100],
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
    df_exp["Plan_Tratamiento"] = df_exp["Plan_Tratamiento"].fillna("Sin Plan Asignado")
    df_exp["IMC_norm"] = (
        (df_exp["IMC"] - df_exp["IMC"].min()) /
        (df_exp["IMC"].max() - df_exp["IMC"].min())
    )
    df_exp["Categoria_IMC"] = pd.cut(
        df_exp["IMC"], bins=[0, 18.5, 25, 30, 100],
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