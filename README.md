# Intercooler Prediction Tracker — R5 (Render Deploy)

**URL:** https://intercooler-simulator.onrender.com/

## Models Used
| Model File | Algorithm | Predicts |
|---|---|---|
| `best_curve_model_inlet_T_gas_A.pkl` | Curve Fitting | Inlet T gas A from operating_day |
| `best_model_Rf.pkl` | XGBoost | Fouling resistance Rf |
| `best_model_Cp_gas_A.pkl` | Linear Regression | Cp_gas_A (kW/K) |

## Physics Solver
- Equations: `Q_LMTD = Q_gas = Q_CW`
- Unknowns: `outlet_T_gas_A`, `outlet_T_CW_A`
- Area: `A = 198 m²`, `Cp_water = 4.18 kJ/kg·K`

## Files
```
render_deploy_R5/
├── app.py                             ← FastAPI backend (main)
├── index.html                         ← Web frontend
├── requirements.txt
├── Procfile
├── runtime.txt
├── render.yaml
├── .gitignore
├── copy_models.bat                    ← Helper to copy .pkl files
├── best_model_Rf.pkl                  ← (copy before push)
├── best_curve_model_inlet_T_gas_A.pkl ← (copy before push)
└── best_model_Cp_gas_A.pkl            ← (copy before push)
```

## Local Test
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8001
```
