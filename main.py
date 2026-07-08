# -*- coding: utf-8 -*-
"""
開発起動スクリプト: python main.py でアプリを起動する

  1. Python バックエンド (FastAPI) をバックグラウンドで起動
  2. Tauri 開発サーバを起動 (ウィンドウが閉じるまでブロック)
  3. ウィンドウを閉じると Python バックエンドも自動停止
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
TAURI_DIR = ROOT / "vsm-tauri"
BACKEND_PORT = 8000


def _backend_python() -> str:
    """バックエンド起動に使う Python を返す。

    VS Code の実行ボタン等でシステム Python が選ばれても確実に動くよう、
    プロジェクト内の .venv があればそれを最優先で使う (uvicorn 等が入っているため)。
    無ければ現在の実行 Python にフォールバックする。
    """
    for candidate in (
        ROOT / ".venv" / "Scripts" / "python.exe",  # Windows
        ROOT / ".venv" / "bin" / "python",          # macOS / Linux
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_port(port: int) -> None:
    """ポートを LISTEN している既存プロセスがあれば終了させる。

    前回の起動で残ったバックエンド (.venv python のシムが孫として残しやすい) が
    ポートを塞いでいると、新しい uvicorn が bind できず即終了し、
    「バックエンド不在 → グラフが出ない」状態になる。起動前に必ず掃除する。
    """
    if not _port_in_use(port):
        return
    print(f"   ポート{port}が使用中です。既存プロセスを解放します...")
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(["netstat", "-ano"], text=True, errors="ignore")
        except Exception:
            return
        pids = set()
        for line in out.splitlines():
            if f":{port} " in line and "LISTENING" in line.upper():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pids.add(parts[-1])
        for pid in pids:
            subprocess.run(["taskkill", "/F", "/T", "/PID", pid],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   → 既存プロセス PID {pid} を終了しました")
        time.sleep(1.0)


def _kill_tree(proc: subprocess.Popen) -> None:
    """プロセスツリーごと確実に終了させる。

    .venv の python.exe は本体 python へ中継する「シム」なので、
    proc.terminate() ではシムだけが止まり、実際に動いている孫プロセス
    (uvicorn 本体など) が残ってしまう。Windows では taskkill /T で
    子孫まとめて終了させる。
    """
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def main():
    # 前回の残骸がポートを塞いでいたら掃除してから起動する
    _free_port(BACKEND_PORT)

    py = _backend_python()
    print(f"▶ Python バックエンドを起動中 (port {BACKEND_PORT})...\n   使用 Python: {py}")
    backend = subprocess.Popen(
        [
            py, "-m", "uvicorn",
            "backend.main:app",
            "--port", str(BACKEND_PORT),
            "--log-level", "warning",
        ],
        cwd=ROOT,
    )

    print("▶ Tauri 開発サーバを起動中...")

    # Tauri (npm→cmd バッチ) を「独立したプロセスグループ」＋「端末入力から切り離し」で起動する。
    # これをしないと、VS Code のターミナルが実行時に送り込む有効化コマンドや Ctrl+C が
    # 動作中の npm/tauri に割り込み、「Aborted!」や「バッチ ジョブを終了しますか (Y/N)?」で
    # 固まってしまう。stdin を DEVNULL にすることで、その割り込みが Tauri に届かなくなる。
    popen_kwargs = {"cwd": TAURI_DIR, "shell": True, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    tauri = subprocess.Popen("npm run tauri dev", **popen_kwargs)
    try:
        # ウィンドウを閉じると npm run tauri dev が終了し、ここが返る
        tauri.wait()
    except KeyboardInterrupt:
        print("\n▶ 中断を検知しました。停止処理を行います...")
    finally:
        print("▶ 停止中 (Tauri / バックエンド)...")
        _kill_tree(tauri)
        _kill_tree(backend)


if __name__ == "__main__":
    main()
