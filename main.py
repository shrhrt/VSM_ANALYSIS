# -*- coding: utf-8 -*-
"""
アプリケーションのエントリポイント
"""

import os
import sys
from tkinterdnd2 import TkinterDnD
from app.vsm_app import VSMApp


def get_resource_path(relative_path):
    """PyInstaller / Nuitka / 通常実行の全環境でリソースパスを取得"""
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller: 一時解凍フォルダ
        base_path = sys._MEIPASS
    elif "__compiled__" in globals():
        # Nuitka standalone: exe と同じフォルダ
        base_path = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # 通常の Python スクリプト実行
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    root = TkinterDnD.Tk()

    # ウィンドウのアイコンを設定
    icon_path = get_resource_path(os.path.join("assets", "app_icon.ico"))
    if os.path.exists(icon_path):
        root.iconbitmap(icon_path)

    app = VSMApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
