@echo off
REM Node13 demo: replay ESA telemetry through the triage engine, then open the dashboard.
REM Run from the folder that contains node13_app\, node13_v2\ and data\
python -m node13_app.cli replay --channels 41-46 --from 2010-01-01 --to 2010-07-01 --budget 0.2 --reset
streamlit run node13_app\dashboard.py
