# -*- coding: utf-8 -*-
"""
アプリケーションのエントリポイント
"""

from tkinterdnd2 import TkinterDnD
from app.vsm_app import VSMApp

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = VSMApp(root)
    root.mainloop()
