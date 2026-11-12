import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

paquete = joblib.load("modelo_riesgo_crediticio.joblib")
app = FastAPI(title="API de riesgo crediticio")

class Solicitante(BaseModel):
    edad: int
    nivel_educativo: str
    tipo_empleo: str
    estrato: int
    vivienda: str
    personas_a_cargo: int
    ingresos_mensuales: float
    antiguedad_laboral_anios: float
    monto_solicitado: float
    plazo_meses: int
    ratio_cuota_ingreso: float
    creditos_activos: int
    ratio_endeudamiento: float
    moras_previas: int
    max_dias_mora: int
    puntaje_crediticio: float

@app.post("/predecir")
def predecir(s: Solicitante):
    fila = pd.DataFrame([s.model_dump()])[paquete["columnas"]]
    proba = paquete["modelo"].predict_proba(fila)[0]
    clase = int((proba @ paquete["costos"]).argmin())
    return {"riesgo": paquete["orden"][clase],
            "probabilidades": dict(zip(paquete["orden"], [round(float(p), 3) for p in proba]))}