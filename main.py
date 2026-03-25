# -*- coding: utf-8 -*-
import tkinter as tk
from tkinterdnd2 import TkinterDnD
from app.vsm_app import VSMApp

if __name__ == "__main__":
    """
    アプリケーションのエントリポイント。
    """
    root = TkinterDnD.Tk()
    app = VSMApp(root)
    root.mainloop()
