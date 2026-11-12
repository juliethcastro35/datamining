import joblib, pandas as pd, streamlit as st

paquete = joblib.load("modelo_riesgo_crediticio.joblib")
st.title("Clasificación de riesgo crediticio")
st.caption("Prototipo académico con datos sintéticos")

c1, c2 = st.columns(2)
with c1:
    edad = st.number_input("Edad", 21, 68, 35)
    nivel = st.selectbox("Nivel educativo", ["Bachiller", "Técnico", "Profesional", "Posgrado"])
    empleo = st.selectbox("Tipo de empleo", ["Empleado indefinido", "Empleado término fijo", "Independiente", "Pensionado"])
    estrato = st.slider("Estrato", 1, 6, 3)
    vivienda = st.selectbox("Vivienda", ["Propia", "Arrendada", "Familiar"])
    personas = st.slider("Personas a cargo", 0, 5, 1)
    ingresos = st.number_input("Ingresos mensuales (COP)", 1_300_000, 14_000_000, 3_500_000, step=50_000)
    antig = st.number_input("Antigüedad laboral (años)", 0.0, 35.0, 5.0)
with c2:
    monto = st.number_input("Monto solicitado (COP)", 1_500_000, 90_000_000, 12_000_000, step=100_000)
    plazo = st.selectbox("Plazo (meses)", [12, 24, 36, 48, 60, 72], index=2)
    creditos = st.slider("Créditos activos", 0, 6, 1)
    endeud = st.slider("Ratio de endeudamiento", 0.02, 0.85, 0.25)
    moras = st.slider("Moras previas", 0, 8, 0)
    dias = st.slider("Máx. días de mora", 0, 120, 0)
    puntaje = st.slider("Puntaje crediticio", 300, 950, 700)

if st.button("Evaluar riesgo"):
    r = 0.0165
    cuota = monto * r / (1 - (1 + r) ** (-plazo))
    fila = pd.DataFrame([{"edad": edad, "nivel_educativo": nivel, "tipo_empleo": empleo, "estrato": estrato,
        "vivienda": vivienda, "personas_a_cargo": personas, "ingresos_mensuales": ingresos,
        "antiguedad_laboral_anios": antig, "monto_solicitado": monto, "plazo_meses": plazo,
        "ratio_cuota_ingreso": round(cuota / ingresos, 3), "creditos_activos": creditos,
        "ratio_endeudamiento": endeud, "moras_previas": moras, "max_dias_mora": dias,
        "puntaje_crediticio": puntaje}])[paquete["columnas"]]
    proba = paquete["modelo"].predict_proba(fila)[0]
    clase = int((proba @ paquete["costos"]).argmin())
    st.subheader("Riesgo: " + paquete["orden"][clase])
    st.bar_chart(pd.Series(proba, index=paquete["orden"]))