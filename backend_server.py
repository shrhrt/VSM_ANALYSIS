# -*- coding: utf-8 -*-
"""
PyInstaller / 本番ビルド用エントリポイント。
tauri build でサイドカーとして同梱される。
"""
import sys
import uvicorn
from backend.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
