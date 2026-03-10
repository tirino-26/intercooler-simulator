@echo off
REM ============================================================
REM  copy_models.bat
REM  Copy the 3 trained .pkl files into render_deploy_R5 folder
REM ============================================================

SET SRC=d:\LDPE heat ex prediction con\Process vs Outlet temp\digital_twin for intercooler prediction
SET DST=%~dp0

echo Copying models to %DST%

copy "%SRC%\best_model_Rf.pkl"                     "%DST%"
copy "%SRC%\best_curve_model_inlet_T_gas_A.pkl"    "%DST%"
copy "%SRC%\best_model_Cp_gas_A.pkl"               "%DST%"

echo.
echo Done! Files in deploy folder:
dir "%DST%*.pkl"
pause
