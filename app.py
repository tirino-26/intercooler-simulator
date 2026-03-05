"""
Intercooler Prediction Tracker - Cloud Deployment Version
==========================================================
Ready to deploy on Render, Railway, or any cloud platform
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
from scipy.optimize import brentq
import joblib
import os
from pathlib import Path

# =============================================================================
# App Configuration
# =============================================================================
app = FastAPI(
    title="Intercooler Prediction Tracker",
    description="Digital Twin for LDPE Heat Exchanger - K1202E1A",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Constants
# =============================================================================
A_HEAT_TRANSFER = 198  # m2
CP_WATER = 4.18  # kJ/kg.K
MODEL_DIR = Path(__file__).parent

# =============================================================================
# Load Models
# =============================================================================
try:
    model_inlet_T_gas_A = joblib.load(MODEL_DIR / "inlet_T_gas_A_lr.pkl")
    model_U_A = joblib.load(MODEL_DIR / "U_A_lr.pkl")
    model_Q_A = joblib.load(MODEL_DIR / "Q_A_lr.pkl")
    print("✓ All models loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: Could not load models: {e}")
    model_inlet_T_gas_A = None
    model_U_A = None
    model_Q_A = None

# =============================================================================
# Pydantic Models
# =============================================================================
class DayInput(BaseModel):
    operating_day: int = Field(..., ge=1)
    inlet_T_CW: float = Field(..., gt=0, lt=50)
    flow_CW_A: float = Field(..., gt=0, lt=500)

class PredictionRequest(BaseModel):
    day_inputs: List[DayInput]

class DayPrediction(BaseModel):
    operating_day: int
    inlet_T_CW: float
    flow_CW_A: float
    inlet_T_gas_A_pre: float
    Q_A_pre: float
    U_A_pre: float
    outlet_T_CW_A_pre: float
    outlet_T_gas_pre: float

# =============================================================================
# Helper Functions
# =============================================================================
def predict_inlet_T_gas_A(operating_day: int) -> float:
    if model_inlet_T_gas_A is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = pd.DataFrame({'operating_day': [operating_day]})
    return float(model_inlet_T_gas_A.predict(X)[0])

def predict_U_A(inlet_T_gas_A_pre: float, inlet_T_CW: float, flow_CW_A: float, operating_day: int) -> float:
    if model_U_A is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = pd.DataFrame({
        'inlet_T_gas_A_pre': [inlet_T_gas_A_pre],
        'inlet T CW': [inlet_T_CW],
        'Flow CW A': [flow_CW_A],
        'operating_day': [operating_day]
    })
    return float(model_U_A.predict(X)[0])

def predict_Q_A(flow_CW_A: float, inlet_T_CW: float, inlet_T_gas_A_pre: float) -> float:
    if model_Q_A is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = pd.DataFrame({
        'Flow CW A': [flow_CW_A],
        'inlet T CW': [inlet_T_CW],
        'inlet_T_gas_A_pre': [inlet_T_gas_A_pre]
    })
    return float(model_Q_A.predict(X)[0])

def calculate_outlet_T_CW_A(Q_A_pre: float, flow_CW_A: float, inlet_T_CW: float) -> float:
    m_A = flow_CW_A * 1000 / 3600
    return Q_A_pre / (m_A * CP_WATER) + inlet_T_CW

def solve_outlet_T_gas(U_A_pre: float, Q_A_pre: float, inlet_T_gas_A_pre: float, 
                       outlet_T_CW_A_pre: float, inlet_T_CW: float) -> float:
    dT1 = inlet_T_gas_A_pre - outlet_T_CW_A_pre
    
    def equation(T_gas_out):
        dT2 = T_gas_out - inlet_T_CW
        if dT1 <= 0 or dT2 <= 0 or dT1 == dT2:
            return float('inf')
        LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
        Q_calc = U_A_pre * A_HEAT_TRANSFER * LMTD / 1000
        return Q_calc - Q_A_pre
    
    try:
        return brentq(equation, inlet_T_CW + 0.1, inlet_T_gas_A_pre - 0.1)
    except:
        return inlet_T_gas_A_pre - 10.0

def make_prediction(operating_day: int, inlet_T_CW: float, flow_CW_A: float) -> DayPrediction:
    inlet_T_gas_A_pre = predict_inlet_T_gas_A(operating_day)
    U_A_pre = predict_U_A(inlet_T_gas_A_pre, inlet_T_CW, flow_CW_A, operating_day)
    Q_A_pre = predict_Q_A(flow_CW_A, inlet_T_CW, inlet_T_gas_A_pre)
    outlet_T_CW_A_pre = calculate_outlet_T_CW_A(Q_A_pre, flow_CW_A, inlet_T_CW)
    outlet_T_gas_pre = solve_outlet_T_gas(U_A_pre, Q_A_pre, inlet_T_gas_A_pre, outlet_T_CW_A_pre, inlet_T_CW)
    
    return DayPrediction(
        operating_day=operating_day,
        inlet_T_CW=round(inlet_T_CW, 2),
        flow_CW_A=round(flow_CW_A, 2),
        inlet_T_gas_A_pre=round(inlet_T_gas_A_pre, 2),
        Q_A_pre=round(Q_A_pre, 2),
        U_A_pre=round(U_A_pre, 2),
        outlet_T_CW_A_pre=round(outlet_T_CW_A_pre, 2),
        outlet_T_gas_pre=round(outlet_T_gas_pre, 2)
    )

# =============================================================================
# API Endpoints
# =============================================================================
@app.post("/api/predict/batch")
async def predict_batch(request: PredictionRequest):
    predictions = [make_prediction(d.operating_day, d.inlet_T_CW, d.flow_CW_A) 
                   for d in request.day_inputs]
    return {"status": "success", "predictions": predictions}

@app.get("/api/model/info")
async def get_model_info():
    info = {"inlet_T_gas_A_model": None, "U_A_model": None, "Q_A_model": None}
    
    if model_inlet_T_gas_A is not None:
        info["inlet_T_gas_A_model"] = {
            "type": "Linear Regression",
            "intercept": float(model_inlet_T_gas_A.intercept_),
            "coefficients": {"operating_day": float(model_inlet_T_gas_A.coef_[0])}
        }
    if model_U_A is not None:
        info["U_A_model"] = {
            "type": "Linear Regression",
            "intercept": float(model_U_A.intercept_),
            "coefficients": dict(zip(
                ["inlet_T_gas_A_pre", "inlet_T_CW", "Flow_CW_A", "operating_day"],
                [float(c) for c in model_U_A.coef_]
            ))
        }
    if model_Q_A is not None:
        info["Q_A_model"] = {
            "type": "Linear Regression",
            "intercept": float(model_Q_A.intercept_),
            "coefficients": dict(zip(
                ["Flow_CW_A", "inlet_T_CW", "inlet_T_gas_A_pre"],
                [float(c) for c in model_Q_A.coef_]
            ))
        }
    return info

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "models_loaded": model_inlet_T_gas_A is not None}

# =============================================================================
# Serve HTML Frontend
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = MODEL_DIR / "prediction_tracker.html"
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    else:
        return "<h1>HTML file not found</h1>"

# =============================================================================
# For local development
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
