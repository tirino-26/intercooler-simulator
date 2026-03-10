"""
Intercooler Prediction Tracker - R5 (Physics Solver Edition)
=============================================================
Models used:
  • best_curve_model_inlet_T_gas_A.pkl  — Curve fitting (inlet T gas A vs operating_day)
  • best_model_Rf.pkl                   — XGBoost (Rf prediction)
  • best_model_Cp_gas_A.pkl             — Linear Regression (Cp_gas_A prediction)

Physics solver:
  Three equations, two unknowns (outlet_T_gas_A, outlet_T_CW_A):
    Q_LMTD = U × A × LMTD
    Q_gas   = Cp_gas × ΔT_gas
    Q_CW    = m_CW × Cp_water × ΔT_CW

Deploy URL: https://intercooler-simulator.onrender.com/
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import joblib
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# App Setup
# =============================================================================
app = FastAPI(
    title="Intercooler Prediction Tracker",
    description="Digital Twin for LDPE Heat Exchanger K1202E1A — Physics Solver",
    version="5.0.0",
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
A_HX    = 198.0   # m²  heat transfer area
CP_WATER = 4.18   # kJ/kg·K
MODEL_DIR = Path(__file__).parent

# =============================================================================
# Load Models
# =============================================================================
curve_func   = None
curve_popt   = None
rf_model     = None
cp_model     = None
feature_names: list[str] = []
grade_cols:   list[str]  = []
available_grades: list[str] = []

try:
    curve_func, curve_popt = joblib.load(MODEL_DIR / "best_curve_model_inlet_T_gas_A.pkl")
    rf_model = joblib.load(MODEL_DIR / "best_model_Rf.pkl")
    cp_model = joblib.load(MODEL_DIR / "best_model_Cp_gas_A.pkl")

    # Detect feature columns from XGBoost model
    try:
        feature_names = list(rf_model.get_booster().feature_names)
    except Exception:
        feature_names = list(rf_model.feature_names_in_)

    grade_cols       = [f for f in feature_names if f.startswith("grade_")]
    available_grades = sorted([g.replace("grade_", "") for g in grade_cols])

    print("✓ Models loaded successfully")
    print(f"  Rf features     : {feature_names}")
    print(f"  Available grades: {available_grades}")

except Exception as e:
    print(f"⚠  Model load error: {e}")


# =============================================================================
# Pydantic Schemas
# =============================================================================
class DayInput(BaseModel):
    operating_day: int   = Field(..., ge=1, description="Operating day (1-based)")
    inlet_T_CW:   float  = Field(..., gt=0, lt=60,  description="Inlet cooling water temperature (°C)")
    flow_CW_A:    float  = Field(..., gt=0, lt=1000, description="Cooling water flow rate (m³/hr)")
    MFR:          float  = Field(100.0, gt=0,        description="Mass flow rate gas (ton/hr)")
    ld_grade:     str    = Field("",                  description="LD-grade code")
    is_transition: int   = Field(0, ge=0, le=1,       description="1 if grade-change day")


class PredictionRequest(BaseModel):
    day_inputs: List[DayInput]
    U0: float = Field(500.0, gt=0, description="Initial clean U value (W/m²·K) at operating day 1")


class DayResult(BaseModel):
    operating_day:      int
    inlet_T_CW:         float
    flow_CW_A:          float
    inlet_T_gas_A_pre:  float
    Rf_pre:             float
    U_A_pre:            float
    Cp_gas_A_pre:       float
    Q_A_pre:            Optional[float]
    outlet_T_CW_A_pre:  Optional[float]
    outlet_T_gas_pre:   Optional[float]


# =============================================================================
# Core Functions
# =============================================================================
def _predict_inlet_T_gas_A(operating_day: int) -> float:
    """Curve fitting: inlet T gas A from operating day."""
    return float(curve_func(operating_day, *curve_popt))


def _predict_Rf(inlet_T_gas_A: float, inlet_T_CW: float, flow_CW_A: float,
                MFR: float, ld_grade: str, is_transition: int) -> float:
    """XGBoost: predict fouling resistance Rf."""
    row: dict[str, float] = {
        "inlet T gas A": inlet_T_gas_A,
        "inlet T CW":    inlet_T_CW,
        "Flow CW A":     flow_CW_A,
        "MFR":           MFR,
        "is_transition": float(is_transition),
    }
    # One-hot encode grade
    for gc in grade_cols:
        row[gc] = 1.0 if gc == f"grade_{ld_grade}" else 0.0

    X = pd.DataFrame([row])[feature_names]
    return float(rf_model.predict(X)[0])


def _predict_Cp_gas_A(inlet_T_gas_A: float, inlet_T_CW: float, flow_CW_A: float) -> float:
    """Linear Regression: predict Cp_gas_A (kW/K)."""
    X = pd.DataFrame({
        "inlet T gas A": [inlet_T_gas_A],
        "inlet T CW":    [inlet_T_CW],
        "Flow CW A":     [flow_CW_A],
    })
    return float(cp_model.predict(X)[0])


def _physics_solver(inlet_T_gas_A: float, inlet_T_CW: float, flow_CW_A: float,
                    U_pre: float, Cp_gas_A: float):
    """
    Solve two equations for (outlet_T_gas_A, outlet_T_CW_A).

    Units:
      U_pre   — W/m²·K  (divide by 1000 → kW/m²·K)
      Q_*     — kW
    """
    m_CW = flow_CW_A * 1000.0 / 3600.0   # kg/s
    U_kW = U_pre / 1000.0                  # kW/m²·K

    def equations(unknowns):
        T_gas_out, T_CW_out = unknowns
        dT1 = inlet_T_gas_A - T_CW_out   # hot-in – cold-out  (counter-current)
        dT2 = T_gas_out     - inlet_T_CW  # hot-out – cold-in

        if dT1 <= 0 or dT2 <= 0 or abs(dT1 - dT2) < 1e-8:
            LMTD = (dT1 + dT2) / 2.0
        else:
            LMTD = (dT1 - dT2) / np.log(dT1 / dT2)

        Q_LMTD = U_kW * A_HX * LMTD
        Q_gas  = Cp_gas_A * (inlet_T_gas_A - T_gas_out)
        Q_CW   = m_CW * CP_WATER * (T_CW_out - inlet_T_CW)
        return [Q_LMTD - Q_CW, Q_gas - Q_CW]

    # Initial guess (rough estimate)
    T_gas_guess = inlet_T_gas_A - 55.0
    T_CW_guess  = inlet_T_CW  + 8.0

    try:
        sol, _, ier, _ = fsolve(equations, [T_gas_guess, T_CW_guess], full_output=True)
        T_gas_out, T_CW_out = sol

        # Relaxed sanity check (allows temperature cross, which is real in this HX)
        if (ier == 1
                and T_gas_out < inlet_T_gas_A   # gas cools
                and T_CW_out  > inlet_T_CW       # CW heats up
                and T_gas_out > 0
                and T_CW_out  > 0):
            return float(T_gas_out), float(T_CW_out)
    except Exception:
        pass
    return None, None


def predict_one_day(day: DayInput, U0: float) -> DayResult:
    """Full prediction pipeline for a single day."""
    if rf_model is None or cp_model is None or curve_func is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Step 1 — Inlet T gas A from curve model
    inlet_T_gas_A = _predict_inlet_T_gas_A(day.operating_day)

    # Step 2 — Rf from XGBoost
    Rf_pre = _predict_Rf(
        inlet_T_gas_A, day.inlet_T_CW, day.flow_CW_A,
        day.MFR, day.ld_grade, day.is_transition
    )

    # Step 3 — U_pre from Rf  (Rf = 1/U - 1/U0)
    U_pre = 1.0 / (Rf_pre + 1.0 / U0)

    # Step 4 — Cp_gas_A from linear regression
    Cp_gas_A = _predict_Cp_gas_A(inlet_T_gas_A, day.inlet_T_CW, day.flow_CW_A)

    # Step 5 — Physics solver → outlet temperatures
    outlet_T_gas, outlet_T_CW = _physics_solver(
        inlet_T_gas_A, day.inlet_T_CW, day.flow_CW_A, U_pre, Cp_gas_A
    )

    # Step 6 — Q_A from CW energy balance
    Q_A = None
    if outlet_T_CW is not None:
        m_CW = day.flow_CW_A * 1000.0 / 3600.0
        Q_A  = m_CW * CP_WATER * (outlet_T_CW - day.inlet_T_CW)

    return DayResult(
        operating_day      = day.operating_day,
        inlet_T_CW         = round(day.inlet_T_CW, 2),
        flow_CW_A          = round(day.flow_CW_A, 2),
        inlet_T_gas_A_pre  = round(inlet_T_gas_A, 2),
        Rf_pre             = round(Rf_pre, 6),
        U_A_pre            = round(U_pre, 2),
        Cp_gas_A_pre       = round(Cp_gas_A, 4),
        Q_A_pre            = round(Q_A, 2) if Q_A is not None else None,
        outlet_T_CW_A_pre  = round(outlet_T_CW, 2) if outlet_T_CW is not None else None,
        outlet_T_gas_pre   = round(outlet_T_gas, 2) if outlet_T_gas is not None else None,
    )


# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": rf_model is not None,
        "available_grades": available_grades,
    }


@app.get("/api/grades")
async def get_grades():
    """Return list of available LD-grades that the model knows."""
    return {"grades": available_grades}


@app.post("/api/predict/batch")
async def predict_batch(request: PredictionRequest):
    results = []
    errors  = []
    for day in request.day_inputs:
        try:
            results.append(predict_one_day(day, request.U0))
        except Exception as e:
            errors.append({"operating_day": day.operating_day, "error": str(e)})

    return {
        "status": "success",
        "predictions": results,
        "errors": errors,
        "model_info": {
            "A_m2": A_HX,
            "Cp_water_kJ_kgK": CP_WATER,
            "U0_W_m2K": request.U0,
        },
    }


@app.get("/api/model/info")
async def model_info():
    return {
        "version": "R5",
        "models": {
            "inlet_T_gas_A": "Curve Fitting (best_curve_model_inlet_T_gas_A.pkl)",
            "Rf":            "XGBoost     (best_model_Rf.pkl)",
            "Cp_gas_A":      "Linear Reg  (best_model_Cp_gas_A.pkl)",
        },
        "solver": "scipy.fsolve — 2 equations (Q_LMTD, Q_gas, Q_CW)",
        "heat_exchanger": {"A_m2": A_HX, "Cp_water": CP_WATER},
        "feature_names": feature_names,
        "available_grades": available_grades,
        "models_loaded": rf_model is not None,
    }


# =============================================================================
# Serve Frontend
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = MODEL_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>index.html not found</h1>"


# =============================================================================
# Entry Point (local dev)
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print(f"\n  http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
