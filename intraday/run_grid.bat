@echo off
REM Full intraday experiment: 3 bar sizes x 2 overnight rules, net of costs.
REM Needs output\intraday\close_15m.parquet (run download_bars.py first).
REM Expect roughly: 60-min ~10 min each, 30-min ~25 min, 15-min ~1 hour.
cd /d "%~dp0.."
for %%B in (60 30 15) do (
  for %%O in (hold flat) do (
    echo.
    echo ===== %%B-minute bars, overnight=%%O =====
    python intraday\build_family.py --bar %%B --overnight %%O || goto :err
    python intraday\backtest_intraday.py --bar %%B --overnight %%O || goto :err
    python intraday\baselines.py --bar %%B --overnight %%O || goto :err
  )
)
python intraday\summarize.py
goto :eof
:err
echo.
echo A step failed; see the error above.
exit /b 1
