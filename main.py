# -*- coding: utf-8 -*-
import tkinter as tk
from app.vsm_app import VSMApp

if __name__ == "__main__":
    """
    アプリケーションのエントリポイント。
    """
    root = tk.Tk()
    app = VSMApp(root)
    root.mainloop()
