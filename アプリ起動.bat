@echo off
rem VSM Analyzer 開発起動: バックエンド + Tauri をまとめて起動し、
rem ウィンドウを閉じるとプロセスを後片付けする (main.py が担当)。
rem 直接 `npm run tauri dev` を叩くと端末割り込みで「Aborted!」等が起きるため main.py を経由する。
cd /d "%~dp0"
python main.py
