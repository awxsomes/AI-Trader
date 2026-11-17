@echo off
REM First-hour chart-pattern trader on the PATTERN Alpaca paper account.
REM Schedule with Windows Task Scheduler: weekdays at 9:20 AM (Eastern).
REM It waits for the 9:30 open, trades until 10:30, then exits; on market
REM holidays it exits at once. Keys come from PATTERN_ALPACA_API_KEY and
REM PATTERN_ALPACA_SECRET_KEY (see patterns\README.md).
cd /d "%~dp0\.."
if not exist patterns\logs mkdir patterns\logs
echo ===== %date% %time% ===== >> patterns\logs\scheduler.log
python patterns\pattern_trade.py --execute >> patterns\logs\scheduler.log 2>&1
