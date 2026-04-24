# -*- coding: utf-8 -*-
"""
アプリケーションのエントリポイント
"""
import tkinter as tk
from tkinterdnd2 import TkinterDnD
from app.vsm_app import VSMApp

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = VSMApp(root)
    root.mainloop()
