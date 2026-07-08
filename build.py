# -*- coding: utf-8 -*-
"""
配布用ビルドスクリプト: python build.py で .exe/.msi インストーラを生成する

  1. PyInstaller で Python バックエンドを .exe 化
  2. vsm-tauri/src-tauri/binaries/ にサイドカーとしてコピー
  3. tauri build で Tauri アプリをビルド
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TAURI_DIR = ROOT / "vsm-tauri"
BINARIES_DIR = TAURI_DIR / "src-tauri" / "binaries"
TARGET_TRIPLE = "x86_64-pc-windows-msvc"
SIDECAR_NAME = f"backend-{TARGET_TRIPLE}.exe"


def build_python_backend():
    print("▶ Step 1: Python バックエンドを PyInstaller でビルド中...")
    print("-" * 60)

    # Conda 環境では Library\bin の DLL が PyInstaller の検索対象外になるため PATH に追加
    conda_bin = r"C:\Users\shrhr\anaconda3\Library\bin"
    conda_dlls = r"C:\Users\shrhr\anaconda3\DLLs"
    env = os.environ.copy()
    env["PATH"] = conda_bin + os.pathsep + conda_dlls + os.pathsep + env.get("PATH", "")
    print(f"  Conda DLL パスを追加: {conda_bin}")

    result = subprocess.run(
        [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--name", "backend",
            "--noconfirm",
            "--collect-all", "uvicorn",
            "--collect-all", "fastapi",
            "--collect-all", "pydantic",
            "--hidden-import", "anyio.backends.asyncio",
            "--exclude-module", "matplotlib",
            "--exclude-module", "tkinter",
            "--exclude-module", "_tkinter",
            "--add-data", f"analysis{';'}analysis",
            "--add-data", f"backend{';'}backend",
            "backend_server.py",
        ],
        cwd=ROOT,
        env=env,
    )

    if result.returncode != 0:
        print("✗ PyInstaller ビルド失敗", file=sys.stderr)
        sys.exit(result.returncode)

    src_exe = ROOT / "dist" / "backend.exe"
    if not src_exe.exists():
        print(f"✗ ビルド成果物が見つかりません: {src_exe}", file=sys.stderr)
        sys.exit(1)

    BINARIES_DIR.mkdir(parents=True, exist_ok=True)
    dst_exe = BINARIES_DIR / SIDECAR_NAME
    shutil.copy2(src_exe, dst_exe)
    print(f"✓ サイドカー配置完了: {dst_exe}")


def build_tauri():
    print("\n▶ Step 2: tauri build 中 (5〜15分かかります)...")
    print("-" * 60)

    result = subprocess.run(
        "npm run tauri -- build",
        cwd=TAURI_DIR,
        shell=True,
    )

    if result.returncode != 0:
        print(f"✗ tauri build 失敗 (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    bundle_dir = TAURI_DIR / "src-tauri" / "target" / "release" / "bundle"
    print(f"\n✓ ビルド完了。インストーラ出力先:\n  {bundle_dir}")


if __name__ == "__main__":
    build_python_backend()
    build_tauri()
