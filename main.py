# -*- coding: utf-8 -*-
"""
アプリケーションのエントリポイント
"""

import os
import sys
from tkinterdnd2 import TkinterDnD
from app.vsm_app import VSMApp


def get_resource_path(relative_path):
    """PyInstallerでビルドされた実行ファイル内のリソースパスを取得"""
    try:
        # PyInstaller実行時は一時フォルダ(_MEIPASS)のパスを返す
        base_path = sys._MEIPASS
    except Exception:
        # 通常のPythonスクリプトとして実行した場合は現在のディレクトリを返す
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    root = TkinterDnD.Tk()

    # ウィンドウのアイコンを設定
    icon_path = get_resource_path(os.path.join("assets", "app_icon.ico"))
    if os.path.exists(icon_path):
        root.iconbitmap(icon_path)

    app = VSMApp(root)
    root.mainloop()
