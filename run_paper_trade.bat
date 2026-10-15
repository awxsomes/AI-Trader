@echo off
REM Daily rebalance of the Alpaca PAPER account. Used by Windows Task Scheduler.
REM Keys come from the ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables
REM (set once with setx, see README section 4.2).
cd /d "%~dp0"
if not exist live\logs mkdir live\logs
echo ===== %date% %time% ===== >> live\logs\scheduler.log
python live\paper_trade.py --execute >> live\logs\scheduler.log 2>&1
