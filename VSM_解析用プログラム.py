# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, colorchooser
import sys
from contextlib import redirect_stdout
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import TclError

import vsm_analysis_logic as vsm_logic


# -----------------------------------------------------------------------------
# GUIアプリケーションのクラス
# -----------------------------------------------------------------------------
class VSMApp:
    def __init__(self, root):
        self.vsm_data = []
        self._update_job = None
        self.all_metadata = {}
        self.file_color_vars = []
        self.base_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        self.root = root
        self.root.title("VSM Data Analyzer")
        self.root.geometry("1200x850")

        # --- スタイル変数 ---
        self.thick_var = tk.StringVar(value="100.0")
        self.area_var = tk.StringVar(value="1.0")
        self.offset_correction_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)
        self.unit_mode_var = tk.StringVar(value="SI (T, kA/m)")
        self.marker_size_var = tk.StringVar(value="5")
        self.line_width_var = tk.StringVar(value="1.5")
        self.axis_label_fontsize_var = tk.StringVar(value="16")
        self.tick_label_fontsize_var = tk.StringVar(value="12")
        self.legend_fontsize_var = tk.StringVar(value="12")
        self.xlim_min_var = tk.StringVar(value="")
        self.xlim_max_var = tk.StringVar(value="")
        self.ylim_min_var = tk.StringVar(value="")
        self.ylim_max_var = tk.StringVar(value="")
        self.show_grid_var = tk.BooleanVar(value=True)
        self.show_zero_lines_var = tk.BooleanVar(value=True)
        self.save_width_var = tk.StringVar(value="6.0")
        self.save_height_var = tk.StringVar(value="6.0")
        self.save_dpi_var = tk.StringVar(value="300")

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self._configure_styles()
        self.root.configure(bg=self.style.lookup(".", "background"))

        # --- メインレイアウト (PanedWindowベース) ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        left_pane = ttk.Frame(main_paned_window, padding=0)
        main_paned_window.add(left_pane, weight=1)

        left_paned_window = ttk.PanedWindow(left_pane, orient=tk.VERTICAL)
        left_paned_window.pack(fill=tk.BOTH, expand=True)

        graph_frame = ttk.LabelFrame(main_paned_window, text=" グラフ ", padding=10)
        main_paned_window.add(graph_frame, weight=2)
        graph_frame.grid_rowconfigure(1, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)

        notebook_frame = ttk.Frame(left_paned_window, padding=0)
        left_paned_window.add(notebook_frame, weight=1)

        notebook = ttk.Notebook(notebook_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_analysis = ttk.Frame(notebook, padding="10")
        tab_style = ttk.Frame(notebook, padding="10")
        tab_export = ttk.Frame(notebook, padding="10")
        notebook.add(tab_analysis, text="解析")
        notebook.add(tab_style, text="グラフ設定")
        notebook.add(tab_export, text="保存")

        log_outer_frame = ttk.Frame(
            left_paned_window, padding=(0, 10, 0, 0)
        )  # 上にスペース
        left_paned_window.add(log_outer_frame, weight=1)

        log_frame = ttk.LabelFrame(log_outer_frame, text=" ログ ", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 9), bg="white", fg="black"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- 各タブのコントロールを作成 ---
        self._create_analysis_controls(tab_analysis)
        self._create_style_controls(tab_style)
        self._create_export_controls(tab_export)

        # --- グラフ埋め込み ---
        self.fig = plt.figure(figsize=(9, 9), facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame, pack_toolbar=False)
        toolbar.config(background=self.style.lookup(".", "background"))
        toolbar._message_label.config(
            background=self.style.lookup(".", "background"), foreground="black"
        )
        for button in toolbar.winfo_children():
            if isinstance(button, (tk.Button, tk.Checkbutton)):
                button.config(
                    background=self.style.lookup(".", "background"),
                    foreground="black",
                    highlightbackground=self.style.lookup(".", "background"),
                )
        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="ew", padx=5)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        self._add_traces()
        self.update_graph()

    def _configure_styles(self):
        bg, fg, entry_bg = "SystemButtonFace", "black", "white"
        border, accent = "#CCCCCC", "#007ACC"
        self.style.configure(".", background=bg, foreground=fg, font=("Arial", 10))
        self.style.configure("TLabel", background=bg, foreground=fg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TNotebook", background=bg, borderwidth=0)
        self.style.configure(
            "TNotebook.Tab",
            background=bg,
            foreground="#666666",
            padding=[10, 5],
            font=("Arial", 10, "bold"),
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", accent)],
            foreground=[("selected", "white")],
        )
        self.style.configure(
            "TEntry", fieldbackground=entry_bg, foreground=fg, insertcolor=fg
        )
        self.style.configure(
            "TLabelframe", background=bg, bordercolor=border, foreground=fg
        )
        self.style.configure(
            "TLabelframe.Label",
            background=bg,
            foreground=accent,
            font=("Arial", 11, "bold"),
        )
        self.style.configure(
            "TButton",
            background=accent,
            foreground="white",
            font=("Arial", 11, "bold"),
            borderwidth=0,
        )
        self.style.map("TButton", background=[("active", "#005F9E")])
        self.style.configure("TCheckbutton", background=bg, foreground=fg)

    def _create_analysis_controls(self, parent):
        file_frame = ttk.LabelFrame(parent, text=" ファイル ", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(
            file_frame, text="ファイルを選択", command=self.load_files, padding="10 5"
        ).pack(fill=tk.X)
        self.info_button = ttk.Button(
            file_frame,
            text="測定情報を表示",
            command=self.show_metadata_window,
            state=tk.DISABLED,
        )
        self.info_button.pack(fill=tk.X, pady=(5, 0))

        settings_frame = ttk.LabelFrame(parent, text=" 解析設定 ", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        settings_frame.grid_columnconfigure(1, weight=1)

        unit_label = ttk.Label(settings_frame, text="表示単位系:")
        unit_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        unit_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.unit_mode_var,
            values=["SI (T, kA/m)", "CGS (Oe, emu/cm³)", "Normalized (T, M/Ms)"],
            state="readonly",
        )
        unit_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 5))

        ttk.Label(settings_frame, text="膜厚 (nm):").grid(row=1, column=0, sticky="w")
        self.thick_entry = ttk.Entry(
            settings_frame, textvariable=self.thick_var, width=10
        )
        self.thick_entry.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(settings_frame, text="基板面積 (cm²):").grid(
            row=2, column=0, sticky="w"
        )
        self.area_entry = ttk.Entry(
            settings_frame, textvariable=self.area_var, width=10
        )
        self.area_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=(5, 0))
        self.offset_check = ttk.Checkbutton(
            settings_frame,
            text="磁化オフセット補正",
            variable=self.offset_correction_var,
        )
        self.offset_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.legend_check = ttk.Checkbutton(
            settings_frame, text="凡例を表示", variable=self.show_legend_var
        )
        self.legend_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))

        demag_outer_frame = ttk.LabelFrame(parent, text=" 反磁性補正 ", padding="10")
        demag_outer_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create a canvas and a scrollbar
        demag_canvas = tk.Canvas(
            demag_outer_frame,
            borderwidth=0,
            background=self.style.lookup(".", "background"),
            highlightthickness=0,
            height=200,
        )
        demag_scrollbar = ttk.Scrollbar(
            demag_outer_frame, orient="vertical", command=demag_canvas.yview
        )
        self.demag_scrollable_frame = ttk.Frame(demag_canvas)

        self.demag_scrollable_frame.bind(
            "<Configure>",
            lambda e: demag_canvas.configure(scrollregion=demag_canvas.bbox("all")),
        )

        demag_canvas.create_window((0, 0), window=self.demag_scrollable_frame, anchor="nw")
        demag_canvas.configure(yscrollcommand=demag_scrollbar.set)

        demag_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        demag_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.apply_to_all_button = ttk.Button(
            demag_outer_frame,
            text="一番上の設定を全ファイルに適用",
            command=self._apply_first_file_settings_to_all,
            state=tk.DISABLED,
        )
        self.apply_to_all_button.pack(fill=tk.X, pady=(5, 0), padx=5)

        ms_frame = ttk.LabelFrame(parent, text=" 飽和磁化 (Ms) 計算 ", padding="10")
        ms_frame.pack(fill=tk.X, pady=(0, 10))
        self.ms_settings_button = ttk.Button(
            ms_frame,
            text="計算範囲を手動指定...",
            command=self._show_ms_settings_window,
            state=tk.DISABLED,
        )
        self.ms_settings_button.pack(fill=tk.X, pady=5)

    def _show_ms_settings_window(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("飽和磁化(Ms) 計算範囲設定")
        settings_window.geometry("650x500")
        settings_window.transient(self.root)
        settings_window.grab_set()

        main_frame = ttk.Frame(settings_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame, borderwidth=0, background=self.style.lookup(".", "background"), highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=(0, 0, 10, 0))

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        file_settings_vars = []

        def _toggle_manual_entries(widgets, manual_var):
            state = tk.NORMAL if manual_var.get() else tk.DISABLED
            for widget in widgets:
                widget.config(state=state)

        for i, data in enumerate(self.vsm_data):
            file_path = data["path"]
            current_settings = data.get("ms_calc_settings", {})

            manual_var = tk.BooleanVar(value=current_settings.get("manual", False))
            link_var = tk.BooleanVar(value=current_settings.get("link_ranges", True))
            pos_min_var = tk.StringVar(value=current_settings.get("pos_range", ("1.5", "2.0"))[0])
            pos_max_var = tk.StringVar(value=current_settings.get("pos_range", ("1.5", "2.0"))[1])
            neg_min_var = tk.StringVar(value=current_settings.get("neg_range", ("-2.0", "-1.5"))[0])
            neg_max_var = tk.StringVar(value=current_settings.get("neg_range", ("-2.0", "-1.5"))[1])

            file_settings_vars.append({
                "manual": manual_var,
                "link": link_var,
                "pos_min": pos_min_var,
                "pos_max": pos_max_var,
                "neg_min": neg_min_var,
                "neg_max": neg_max_var,
            })

            frame = ttk.LabelFrame(scrollable_frame, text=file_path.name, padding=10)
            frame.pack(fill=tk.X, expand=True, pady=5)
            frame.grid_columnconfigure(1, weight=1)
            frame.grid_columnconfigure(3, weight=1)

            manual_check = ttk.Checkbutton(frame, text="手動範囲で計算", variable=manual_var)
            manual_check.grid(row=0, column=0, columnspan=2, sticky="w")
            
            link_check = ttk.Checkbutton(frame, text="正負の範囲を連動", variable=link_var)
            link_check.grid(row=0, column=2, columnspan=2, sticky="w")

            pos_min_entry = ttk.Entry(frame, textvariable=pos_min_var, width=7)
            pos_min_entry.grid(row=1, column=1, sticky="ew", padx=(0,2))
            ttk.Label(frame, text="～").grid(row=1, column=2)
            pos_max_entry = ttk.Entry(frame, textvariable=pos_max_var, width=7)
            pos_max_entry.grid(row=1, column=3, sticky="ew", padx=(2,0))
            ttk.Label(frame, text=" (正 H)").grid(row=1, column=4, sticky="w")
            
            neg_min_entry = ttk.Entry(frame, textvariable=neg_min_var, width=7)
            neg_min_entry.grid(row=2, column=1, sticky="ew", pady=(5, 0), padx=(0,2))
            ttk.Label(frame, text="～").grid(row=2, column=2, pady=(5, 0))
            neg_max_entry = ttk.Entry(frame, textvariable=neg_max_var, width=7)
            neg_max_entry.grid(row=2, column=3, sticky="ew", pady=(5, 0), padx=(2,0))
            ttk.Label(frame, text=" (負 H)").grid(row=2, column=4, sticky="w", pady=(5, 0))
            
            manual_entries = [pos_min_entry, pos_max_entry, neg_min_entry, neg_max_entry]
            
            def on_pos_change(*args, p_min_v=pos_min_var, p_max_v=pos_max_var, n_min_v=neg_min_var, n_max_v=neg_max_var, l_v=link_var):
                if l_v.get():
                    try:
                        p_min = float(p_min_v.get())
                        n_max_v.set(str(-p_min))
                    except (ValueError, TclError): pass
                    try:
                        p_max = float(p_max_v.get())
                        n_min_v.set(str(-p_max))
                    except (ValueError, TclError): pass
            
            pos_min_var.trace_add("write", on_pos_change)
            pos_max_var.trace_add("write", on_pos_change)

            manual_var.trace_add("write", lambda *a, w=manual_entries, v=manual_var: _toggle_manual_entries(w, v))
            _toggle_manual_entries(manual_entries, manual_var)

        button_frame = ttk.Frame(settings_window, padding=(10, 0, 10, 10))
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        def save_settings():
            try:
                for i, data in enumerate(self.vsm_data):
                    vars_dict = file_settings_vars[i]
                    settings = {
                        "manual": vars_dict["manual"].get(),
                        "link_ranges": vars_dict["link"].get(),
                        "pos_range": (
                            vars_dict["pos_min"].get(),
                            vars_dict["pos_max"].get(),
                        ),
                        "neg_range": (
                            vars_dict["neg_min"].get(),
                            vars_dict["neg_max"].get(),
                        ),
                    }
                    data["ms_calc_settings"] = settings
                settings_window.destroy()
                self.update_graph()
                messagebox.showinfo("成功", "飽和磁化の計算設定を保存しました。", parent=settings_window)
            except ValueError:
                messagebox.showerror("入力エラー", "磁場範囲には有効な数値を入力してください。", parent=settings_window)

        ttk.Button(button_frame, text="キャンセル", command=settings_window.destroy).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(button_frame, text="OK & 保存", command=save_settings).grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _create_style_controls(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        axis_grid_frame = ttk.LabelFrame(parent, text=" 軸とグリッド ", padding="10")
        axis_grid_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            axis_grid_frame, text="グリッド線を表示", variable=self.show_grid_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            axis_grid_frame, text="原点線を表示", variable=self.show_zero_lines_var
        ).pack(anchor="w", pady=(5, 0))
        plot_frame = ttk.LabelFrame(parent, text=" プロット ", padding="10")
        plot_frame.pack(fill=tk.X, pady=(0, 10))
        plot_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(plot_frame, text="マーカーサイズ:").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(plot_frame, textvariable=self.marker_size_var, width=10).grid(
            row=0, column=1, sticky="ew", padx=5, pady=(0, 5)
        )
        ttk.Label(plot_frame, text="線幅:").grid(
            row=1, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(plot_frame, textvariable=self.line_width_var, width=10).grid(
            row=1, column=1, sticky="ew", padx=5, pady=(0, 5)
        )
        font_frame = ttk.LabelFrame(parent, text=" フォントサイズ ", padding="10")
        font_frame.pack(fill=tk.X, pady=(0, 10))
        font_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(font_frame, text="軸ラベル:").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(font_frame, textvariable=self.axis_label_fontsize_var, width=10).grid(
            row=0, column=1, sticky="ew", padx=5, pady=(0, 5)
        )
        ttk.Label(font_frame, text="目盛り:").grid(
            row=1, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(font_frame, textvariable=self.tick_label_fontsize_var, width=10).grid(
            row=1, column=1, sticky="ew", padx=5, pady=(0, 5)
        )
        ttk.Label(font_frame, text="凡例:").grid(
            row=2, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(font_frame, textvariable=self.legend_fontsize_var, width=10).grid(
            row=2, column=1, sticky="ew", padx=5, pady=(0, 5)
        )
        self.individual_color_frame = ttk.LabelFrame(
            parent, text=" ファイルリストと描画順 ", padding="10"
        )
        self.individual_color_frame.pack(fill=tk.X, pady=(10, 0))
        self.individual_color_frame.grid_columnconfigure(1, weight=1)
        axes_frame = ttk.LabelFrame(parent, text=" 描画範囲 ", padding="10")
        axes_frame.pack(fill=tk.X, pady=(0, 10))
        axes_frame.grid_columnconfigure(1, weight=1)
        axes_frame.grid_columnconfigure(3, weight=1)
        ttk.Label(axes_frame, text="X軸 (T):").grid(row=0, column=0, sticky="w")
        ttk.Entry(axes_frame, textvariable=self.xlim_min_var, width=7).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(axes_frame, text="～").grid(row=0, column=2)
        ttk.Entry(axes_frame, textvariable=self.xlim_max_var, width=7).grid(
            row=0, column=3, sticky="ew"
        )
        ttk.Label(axes_frame, text="Y軸 (kA/m):").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        ttk.Entry(axes_frame, textvariable=self.ylim_min_var, width=7).grid(
            row=1, column=1, sticky="ew", pady=(5, 0)
        )
        ttk.Label(axes_frame, text="～").grid(row=1, column=2, pady=(5, 0))
        ttk.Entry(axes_frame, textvariable=self.ylim_max_var, width=7).grid(
            row=1, column=3, sticky="ew", pady=(5, 0)
        )

    def _create_export_controls(self, parent):
        save_settings_frame = ttk.LabelFrame(
            parent, text=" 画像サイズ設定 ", padding="10"
        )
        save_settings_frame.pack(fill=tk.X, pady=(0, 10))
        save_settings_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(save_settings_frame, text="幅:").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(save_settings_frame, textvariable=self.save_width_var, width=10).grid(
            row=0, column=1, sticky="ew", padx=5, pady=3
        )
        ttk.Label(save_settings_frame, text="inch").grid(row=0, column=2, sticky="w")
        ttk.Label(save_settings_frame, text="高さ:").grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Entry(
            save_settings_frame, textvariable=self.save_height_var, width=10
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=3)
        ttk.Label(save_settings_frame, text="inch").grid(row=1, column=2, sticky="w")
        dpi_frame = ttk.LabelFrame(parent, text=" 解像度設定 ", padding="10")
        dpi_frame.pack(fill=tk.X, pady=(0, 10))
        dpi_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(dpi_frame, text="DPI:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(dpi_frame, textvariable=self.save_dpi_var, width=10).grid(
            row=0, column=1, sticky="ew", padx=5, pady=3
        )
        save_button_frame = ttk.Frame(parent, padding="10 10 10 0")
        save_button_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(
            save_button_frame,
            text="画像を保存 (Save Image)",
            command=self.save_figure,
            padding="10",
        ).pack(fill=tk.X, expand=True, side=tk.BOTTOM)

    def _add_traces(self):
        trace_vars = [
            self.thick_var,
            self.area_var,
            self.offset_correction_var,
            self.show_legend_var,
            self.marker_size_var,
            self.line_width_var,
            self.axis_label_fontsize_var,
            self.tick_label_fontsize_var,
            self.legend_fontsize_var,
            self.xlim_min_var,
            self.xlim_max_var,
            self.ylim_min_var,
            self.ylim_max_var,
            self.show_grid_var,
            self.show_zero_lines_var,
            self.unit_mode_var,
        ]
        for var in trace_vars:
            var.trace_add("write", self._schedule_update)

    def _update_file_list_ui(self):
        # Clear existing widgets
        for widget in self.individual_color_frame.winfo_children():
            widget.destroy()

        # Re-create widgets for each file
        for i, data in enumerate(self.vsm_data):
            row_frame = ttk.Frame(self.individual_color_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            filename = data["path"].name
            color_var = self.file_color_vars[i]

            # Up/Down buttons
            up_button = ttk.Button(
                row_frame, text="↑", width=3, command=lambda idx=i: self._move_file_up(idx)
            )
            up_button.pack(side=tk.LEFT, padx=(0, 2))
            if i == 0:
                up_button.config(state=tk.DISABLED)

            down_button = ttk.Button(
                row_frame, text="↓", width=3, command=lambda idx=i: self._move_file_down(idx)
            )
            down_button.pack(side=tk.LEFT, padx=(0, 5))
            if i == len(self.vsm_data) - 1:
                down_button.config(state=tk.DISABLED)
            
            # File label
            display_name = (filename[:25] + "..") if len(filename) > 27 else filename
            ttk.Label(row_frame, text=display_name).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )
            
            # Color preview and button
            preview = tk.Label(row_frame, text="", bg=color_var.get(), width=4)
            preview.pack(side=tk.RIGHT, padx=5)
            color_var.trace_add(
                "write",
                lambda *args, p=preview, cv=color_var: (
                    p.config(bg=cv.get()),
                    self._schedule_update(),
                ),
            )
            ttk.Button(
                row_frame,
                text="色選択",
                width=6,
                command=lambda idx=i: self.choose_individual_color(idx),
            ).pack(side=tk.RIGHT)

    def _move_file_up(self, index):
        if index > 0:
            # Swap data
            self.vsm_data.insert(index - 1, self.vsm_data.pop(index))
            self.file_color_vars.insert(index - 1, self.file_color_vars.pop(index))

            # Rebuild all dynamic UIs and update graph
            self._update_file_list_ui()
            self._update_demag_settings_ui()
            self.update_graph()

    def _move_file_down(self, index):
        if index < len(self.vsm_data) - 1:
            # Swap data
            self.vsm_data.insert(index + 1, self.vsm_data.pop(index))
            self.file_color_vars.insert(index + 1, self.file_color_vars.pop(index))

            # Rebuild all dynamic UIs and update graph
            self._update_file_list_ui()
            self._update_demag_settings_ui()
            self.update_graph()

    def choose_individual_color(self, index):
        if index >= len(self.file_color_vars):
            return
        color_var = self.file_color_vars[index]
        path_name = self.vsm_data[index]["path"].name
        title_name = (path_name[:40] + "..") if len(path_name) > 42 else path_name
        color_code = colorchooser.askcolor(
            title=f"'{title_name}' の色を選択", initialcolor=color_var.get()
        )
        if color_code and color_code[1]:
            color_var.set(color_code[1])

    def _schedule_update(self, *args):
        if self._update_job:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(250, self.update_graph)

    def log_message(self, message):
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def show_metadata_window(self):
        if not self.all_metadata:
            messagebox.showinfo("情報", "表示できる測定情報がありません。", parent=self.root)
            return
        info_window = tk.Toplevel(self.root)
        info_window.title("測定情報")
        info_window.geometry("500x650")
        info_window.configure(bg=self.style.lookup(".", "background"))
        top_frame = ttk.Frame(info_window, padding="10 10 10 0")
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="ファイルを選択:").pack(side=tk.LEFT, padx=(0, 10))
        file_names = list(self.all_metadata.keys())
        selected_file = tk.StringVar(value=file_names[0])
        file_menu = ttk.Combobox(
            top_frame, textvariable=selected_file, values=file_names, state="readonly"
        )
        file_menu.pack(fill=tk.X, expand=True)
        text_widget = scrolledtext.ScrolledText(
            info_window, wrap=tk.WORD, font=("Arial", 10), bg="white", fg="black"
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        def update_display(event=None):
            filename = selected_file.get()
            metadata = self.all_metadata.get(filename, {})
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            display_keys = {
                "date": "測定日",
                "sample name": "サンプル名",
                "comment": "コメント",
                "lock-in amp. sensitivity": "感度(mV)",
                "lock-in amp. time constant": "時定数(msec)",
                "measuring points": "測定点数",
                "max magnetic field": "最大磁場(Oe)",
                "max magnetization": "最大磁化(emu)",
                "lock-in amp. phase": "位相(deg)",
                "pole piece gap": "磁極間距離(mm)",
                "calibration value": "校正値",
            }
            info_text = f"ファイル: {filename}\n\n--- 測定パラメータ ---\n\n"
            for key, desc in display_keys.items():
                info_text += f"■ {desc}\n  {metadata.get(key, '---')}\n\n"
            text_widget.insert(tk.END, info_text)
            text_widget.config(state=tk.DISABLED)

        file_menu.bind("<<ComboboxSelected>>", update_display)
        update_display()

    def load_files(self):
        files = filedialog.askopenfilenames(
            title="解析したいVSMファイルを選択",
            filetypes=[("VSM files", "*.VSM"), ("All files", "*.*")],
            parent=self.root
        )
        if not files:
            return
        self.vsm_data, self.file_color_vars = [], []
        for i, file_path in enumerate(files):
            path = Path(file_path)
            header_row = vsm_logic.find_header_row(path)
            try:
                try:
                    df = pd.read_csv(path, header=header_row, encoding="shift-jis")
                except UnicodeDecodeError:
                    df = pd.read_csv(path, header=header_row, encoding="utf-8")
                df.dropna(inplace=True)
                if not {"H(Oe)", "M(emu)"}.issubset(df.columns):
                    messagebox.showwarning(
                        "形式エラー", f"ファイル '{path.name}' に必要な列がありません。", parent=self.root
                    )
                    continue
                self.vsm_data.append({"path": path, "df": df})
                color_var = tk.StringVar(
                    value=self.base_colors[i % len(self.base_colors)]
                )
                self.file_color_vars.append(color_var)
            except Exception as e:
                messagebox.showerror("読込エラー", f"'{path.name}'の読込失敗:\n{e}", parent=self.root)

        self.info_button.config(state=tk.NORMAL if self.vsm_data else tk.DISABLED)
        self._update_file_list_ui()
        self._update_demag_settings_ui()
        self.update_graph()

    def _update_demag_settings_ui(self):
        # Clear existing widgets
        for widget in self.demag_scrollable_frame.winfo_children():
            widget.destroy()

        if not self.vsm_data:
            ttk.Label(self.demag_scrollable_frame, text="ファイルが読み込まれていません。").pack()
            return

        def _toggle_manual_entries(widgets, manual_var):
            state = tk.NORMAL if manual_var.get() else tk.DISABLED
            for widget in widgets:
                widget.config(state=state)

        for i, data in enumerate(self.vsm_data):
            # Ensure settings exist
            if "demag_settings" not in data:
                data["demag_settings"] = {
                    "enabled": True,
                    "manual": False,
                    "pos_range": ("1.5", "2.0"),
                    "neg_range": ("-2.0", "-1.5"),
                    "link_ranges": True,
                }
            
            # Store vars in the data dict to keep them alive
            if "demag_vars" not in data:
                 data["demag_vars"] = {}

            current_settings = data["demag_settings"]
            
            enabled_var = tk.BooleanVar(value=current_settings.get("enabled", True))
            manual_var = tk.BooleanVar(value=current_settings.get("manual", False))
            pos_min_var = tk.StringVar(value=current_settings.get("pos_range", ["1.5", "2.0"])[0])
            pos_max_var = tk.StringVar(value=current_settings.get("pos_range", ["1.5", "2.0"])[1])
            neg_min_var = tk.StringVar(value=current_settings.get("neg_range", ["-2.0", "-1.5"])[0])
            neg_max_var = tk.StringVar(value=current_settings.get("neg_range", ["-2.0", "-1.5"])[1])
            link_var = tk.BooleanVar(value=current_settings.get("link_ranges", True))
            
            data["demag_vars"] = {
                "enabled": enabled_var,
                "manual": manual_var,
                "pos_min": pos_min_var,
                "pos_max": pos_max_var,
                "neg_min": neg_min_var,
                "neg_max": neg_max_var,
                "link": link_var,
            }
            
            # --- Create Widgets ---
            frame = ttk.LabelFrame(self.demag_scrollable_frame, text=data["path"].name, padding=10)
            frame.pack(fill=tk.X, expand=True, pady=5, padx=5)
            frame.grid_columnconfigure(1, weight=1)
            frame.grid_columnconfigure(3, weight=1)

            enabled_check = ttk.Checkbutton(frame, text="反磁性補正", variable=enabled_var)
            enabled_check.grid(row=0, column=0, sticky="w")

            manual_check = ttk.Checkbutton(frame, text="手動範囲", variable=manual_var)
            manual_check.grid(row=0, column=1, sticky="w")
            
            link_check = ttk.Checkbutton(frame, text="正負の範囲を連動", variable=link_var)
            link_check.grid(row=0, column=2, columnspan=2, sticky="w")

            pos_min_entry = ttk.Entry(frame, textvariable=pos_min_var, width=7)
            pos_min_entry.grid(row=1, column=1, sticky="ew", padx=(0,2))
            ttk.Label(frame, text="～").grid(row=1, column=2)
            pos_max_entry = ttk.Entry(frame, textvariable=pos_max_var, width=7)
            pos_max_entry.grid(row=1, column=3, sticky="ew", padx=(2,0))
            ttk.Label(frame, text=" (正 H)").grid(row=1, column=4, sticky="w")
            
            neg_min_entry = ttk.Entry(frame, textvariable=neg_min_var, width=7)
            neg_min_entry.grid(row=2, column=1, sticky="ew", pady=(5, 0), padx=(0,2))
            ttk.Label(frame, text="～").grid(row=2, column=2, pady=(5, 0))
            neg_max_entry = ttk.Entry(frame, textvariable=neg_max_var, width=7)
            neg_max_entry.grid(row=2, column=3, sticky="ew", pady=(5, 0), padx=(2,0))
            ttk.Label(frame, text=" (負 H)").grid(row=2, column=4, sticky="w", pady=(5, 0))
            
            manual_entries = [pos_min_entry, pos_max_entry, neg_min_entry, neg_max_entry]
            
            # --- Traces for real-time update ---
            # Pass the internal var name to the callback to identify the trigger
            enabled_var.trace_add("write", lambda *a, idx=i, v=enabled_var: self._on_demag_setting_change(idx, v))
            manual_var.trace_add("write", lambda *a, idx=i, v=manual_var: self._on_demag_setting_change(idx, v))
            link_var.trace_add("write", lambda *a, idx=i, v=link_var: self._on_demag_setting_change(idx, v))
            pos_min_var.trace_add("write", lambda *a, idx=i, v=pos_min_var: self._on_demag_setting_change(idx, v))
            pos_max_var.trace_add("write", lambda *a, idx=i, v=pos_max_var: self._on_demag_setting_change(idx, v))
            neg_min_var.trace_add("write", lambda *a, idx=i, v=neg_min_var: self._on_demag_setting_change(idx, v))
            neg_max_var.trace_add("write", lambda *a, idx=i, v=neg_max_var: self._on_demag_setting_change(idx, v))


            # Add trace to toggle entries and also trigger update
            manual_var.trace_add(
                "write",
                lambda *args, w=manual_entries, v=manual_var: _toggle_manual_entries(w, v),
            )
            _toggle_manual_entries(manual_entries, manual_var)

        # Update button state
        num_files = len(self.vsm_data)
        self.apply_to_all_button.config(state=tk.NORMAL if num_files > 1 else tk.DISABLED)
        self.ms_settings_button.config(state=tk.NORMAL if num_files > 0 else tk.DISABLED)

    def _apply_first_file_settings_to_all(self):
        if len(self.vsm_data) < 2:
            return

        # 1. Get settings from the first file's data model
        first_file_settings = self.vsm_data[0]["demag_settings"].copy()

        # 2. Apply settings to all other files
        for i in range(1, len(self.vsm_data)):
            data = self.vsm_data[i]
            
            # Update the data model
            data["demag_settings"] = first_file_settings.copy()
            
            # Update the UI variables, which will trigger traces
            vars_dict = data.get("demag_vars")
            if vars_dict:
                # Temporarily disable negative traces to prevent feedback loops
                # while we set them manually. This is a bit tricky with current setup.
                # A safer way is to just set the values. The graph will update once at the end.
                vars_dict["enabled"].set(first_file_settings["enabled"])
                vars_dict["manual"].set(first_file_settings["manual"])
                vars_dict["link"].set(first_file_settings["link_ranges"])
                vars_dict["pos_min"].set(first_file_settings["pos_range"][0])
                vars_dict["pos_max"].set(first_file_settings["pos_range"][1])
                vars_dict["neg_min"].set(first_file_settings["neg_range"][0])
                vars_dict["neg_max"].set(first_file_settings["neg_range"][1])
        
        messagebox.showinfo("成功", "一番上のファイルの設定をすべてのファイルに適用しました。", parent=self.root)
        # The last .set() call will have triggered a graph update via its trace.

    def _on_demag_setting_change(self, file_index, var_changed):
        if file_index >= len(self.vsm_data):
            return

        data = self.vsm_data[file_index]
        vars_dict = data.get("demag_vars", {})
        
        if not vars_dict:
            return

        # --- Linking Logic ---
        if vars_dict["link"].get():
            try:
                if var_changed == vars_dict["pos_min"]:
                    new_val = float(vars_dict["pos_min"].get())
                    vars_dict["neg_max"].set(str(-new_val))
                elif var_changed == vars_dict["pos_max"]:
                    new_val = float(vars_dict["pos_max"].get())
                    vars_dict["neg_min"].set(str(-new_val))
            except (ValueError, TclError):
                pass # Ignore errors during typing

        # --- Save and Update ---
        try:
            # Get all current values
            pos_min_str = vars_dict["pos_min"].get()
            pos_max_str = vars_dict["pos_max"].get()
            neg_min_str = vars_dict["neg_min"].get()
            neg_max_str = vars_dict["neg_max"].get()
            
            # Validate that all values are convertible to float before saving
            float(pos_min_str)
            float(pos_max_str)
            float(neg_min_str)
            float(neg_max_str)
            
            settings = {
                "enabled": vars_dict["enabled"].get(),
                "manual": vars_dict["manual"].get(),
                "link_ranges": vars_dict["link"].get(),
                "pos_range": (pos_min_str, pos_max_str),
                "neg_range": (neg_min_str, neg_max_str),
            }
            data["demag_settings"] = settings
            self._schedule_update()
        except (ValueError, TclError):
            # This can happen if an entry is temporarily invalid during typing.
            # We just won't trigger an update in that case.
            pass


    def save_figure(self):
        try:
            w, h, dpi = (
                float(self.save_width_var.get()),
                float(self.save_height_var.get()),
                int(self.save_dpi_var.get()),
            )
            if w <= 0 or h <= 0 or dpi <= 0:
                raise ValueError("値は正数である必要があります。",)
        except ValueError as e:
            messagebox.showerror(
                "入力エラー", f"幅,高さ,DPIには有効な正数を入力してください。\n({e})", parent=self.root
            )
            return
        file_path = filedialog.asksaveasfilename(
            title="画像を保存",
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("JPEG", "*.jpg"),
            ],
            defaultextension=".png",
            parent=self.root
        )
        if not file_path:
            return
        original_size = self.fig.get_size_inches()
        try:
            self.log_message(
                f"画像を保存中: {Path(file_path).name}\n  サイズ: {w}x{h} inches, DPI: {dpi}\n"
            )
            self.fig.set_size_inches(w, h)
            self.fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
            self.log_message("保存が完了しました。\n")
            messagebox.showinfo("成功", f"画像を保存しました:\n{file_path}", parent=self.root)
        except Exception as e:
            self.log_message(f"エラー: 画像保存失敗 - {e}\n")
            messagebox.showerror("保存エラー", f"画像保存中にエラーが発生:\n{e}", parent=self.root)
        finally:
            self.fig.set_size_inches(original_size)
            self.canvas.draw_idle()

    def update_graph(self):
        self.log_text.delete(1.0, tk.END)
        self.ax.clear()
        self.all_metadata = {}

        unit_mode = self.unit_mode_var.get()

        try:
            params = {
                "Thick": float(self.thick_var.get()),
                "Area": float(self.area_var.get()),
                "marker_size": float(self.marker_size_var.get()),
                "line_width": float(self.line_width_var.get()),
                "axis_label_fontsize": int(self.axis_label_fontsize_var.get()),
                "tick_label_fontsize": int(self.tick_label_fontsize_var.get()),
                "legend_fontsize": int(self.legend_fontsize_var.get()),
                "show_grid": self.show_grid_var.get(),
                "show_zero_lines": self.show_zero_lines_var.get(),
                "xlim_min": float(v) if (v := self.xlim_min_var.get()) else None,
                "xlim_max": float(v) if (v := self.xlim_max_var.get()) else None,
                "ylim_min": float(v) if (v := self.ylim_min_var.get()) else None,
                "ylim_max": float(v) if (v := self.ylim_max_var.get()) else None,
            }
        except ValueError:
            return

        if not self.vsm_data:
            vsm_logic.format_axis(self.ax, self.fig, params, unit_mode)
            self.ax.text(
                0.5,
                0.5,
                "ファイルを選択してください",
                ha="center",
                va="center",
                color="gray",
                fontsize=16,
                transform=self.ax.transAxes,
            )
            self.canvas.draw()
            self._update_demag_settings_ui()
            return

        output_stream = io.StringIO()
        with redirect_stdout(output_stream):
            print(f"解析開始: 膜厚={params['Thick']} nm, 面積={params['Area']} cm²\n")
            self._process_and_plot(params, unit_mode)

        self.log_message(output_stream.getvalue())
        self.log_message("\n描画完了。\n")
        self.canvas.draw()
        self.info_button.config(state=tk.NORMAL if self.all_metadata else tk.DISABLED)

    def _process_and_plot(self, params, unit_mode):
        Vol = params["Area"] * params["Thick"] * 1e-7
        h_min_global, h_max_global = float("inf"), float("-inf")
        print("読み込みファイル:")
        [print(f" {i + 1}: {d['path'].name}") for i, d in enumerate(self.vsm_data)]

        for idx, data in enumerate(self.vsm_data):
            file, df = data["path"], data["df"]
            try:
                self.all_metadata[file.name] = vsm_logic.parse_metadata(file)
                min_H_idx = df["H(Oe)"].idxmin()
                if df["H(Oe)"].iloc[min_H_idx:].empty:
                    raise ValueError("不完全なデータ。復路が見つかりません。")
                max_H_idx2 = min_H_idx + df["H(Oe)"].iloc[min_H_idx:].idxmax()
                df_loop = df.iloc[: max_H_idx2 + 1]
                H_raw, M_raw = df_loop["H(Oe)"] * 1e-4, df_loop["M(emu)"] / Vol
                print(f"\n--- 解析: {file.stem} (データ点: {len(H_raw)}) ---")

                # --- 反磁性補正 ---
                slope, r2_pos, r2_neg = 0, 0, 0
                file_specific_settings = data.get("demag_settings")

                if file_specific_settings and file_specific_settings["enabled"]:
                    print("  反磁性補正: 有効")
                    if file_specific_settings["manual"]:
                        print("    傾き計算: 手動設定モード")
                        try:
                            pos_range = (
                                float(file_specific_settings["pos_range"][0]),
                                float(file_specific_settings["pos_range"][1]),
                            )
                            neg_range = (
                                float(file_specific_settings["neg_range"][0]),
                                float(file_specific_settings["neg_range"][1]),
                            )
                            slope, r2_pos, r2_neg = vsm_logic.find_demag_slope_manual(
                                H_raw, M_raw, pos_range, neg_range
                            )
                        except (ValueError, IndexError):
                            print("  エラー: 手動設定の磁場範囲が無効。")
                            slope, r2_pos, r2_neg = 0, 0, 0
                    else:
                        print("    傾き計算: 自動検出モード")
                        slope, r2_pos, r2_neg = vsm_logic.find_demag_slope_auto(
                            H_raw, M_raw
                        )
                    print(
                        f"    補正傾き S: {slope:.6f}, R^2: [正 {r2_pos:.4f}], [負 {r2_neg:.4f}]"
                    )
                else:
                    print("  反磁性補正: 無効")

                M_corrected = M_raw - H_raw * slope
                if self.offset_correction_var.get():
                    print("  磁化オフセット補正: 有効")
                    H_np, M_np = H_raw.values, M_corrected.values
                    Ms_pos = (
                        np.mean(M_np[H_np > np.max(H_np) * 0.9])
                        if np.any(H_np > np.max(H_np) * 0.9)
                        else 0
                    )
                    Ms_neg = (
                        np.mean(M_np[H_np < np.min(H_np) * 0.9])
                        if np.any(H_np < np.min(H_np) * 0.9)
                        else 0
                    )
                    offset = (Ms_pos + Ms_neg) / 2
                    M_final = M_corrected - offset
                    print(f"    補正値: {offset:.4f} kA/m")
                else:
                    print("  磁化オフセット補正: 無効")
                    M_final = M_corrected

                min_H_idx_loop = H_raw.idxmin()
                H_down, M_down = (
                    H_raw.iloc[: min_H_idx_loop + 1].values,
                    M_final.iloc[: min_H_idx_loop + 1].values,
                )
                H_up, M_up = (
                    H_raw.iloc[min_H_idx_loop:].values,
                    M_final.iloc[min_H_idx_loop:].values,
                )

                # --- 飽和磁化 (Ms) 計算 ---
                ms_settings = data.get("ms_calc_settings")
                ms_pos_range, ms_neg_range = None, None

                try:
                    if ms_settings and ms_settings.get("manual"):
                        # Use manual ranges if they are set
                        print("    Ms計算: 手動範囲を使用")
                        ms_pos_range = (
                            float(ms_settings["pos_range"][0]),
                            float(ms_settings["pos_range"][1]),
                        )
                        ms_neg_range = (
                            float(ms_settings["neg_range"][0]),
                            float(ms_settings["neg_range"][1]),
                        )
                    elif params.get("xlim_max") is not None and params.get("xlim_min") is not None:
                        # Otherwise, use plot limits if they are set
                        print("    Ms計算: 描画範囲を使用")
                        h_max_limit = params["xlim_max"]
                        h_min_limit = params["xlim_min"]
                        # Use the outer 10% of the *visible* range if the range is positive
                        if h_max_limit > 0 and h_min_limit < h_max_limit:
                            pos_start = h_max_limit - (h_max_limit - h_min_limit) * 0.1
                            neg_end = h_min_limit + (h_max_limit - h_min_limit) * 0.1
                            ms_pos_range = (pos_start, h_max_limit)
                            ms_neg_range = (h_min_limit, neg_end)

                except (ValueError, IndexError, TypeError):
                    print("  エラー: Ms計算の範囲が無効です。自動計算にフォールバックします。",)
                    ms_pos_range, ms_neg_range = None, None


                Ms_avg = vsm_logic.calculate_saturation_magnetization(
                    H_raw, M_final, pos_range=ms_pos_range, neg_range=ms_neg_range
                )

                Mr_avg = vsm_logic.calculate_remanence(H_down, M_down, H_up, M_up)
                vsm_logic.calculate_coercivity(H_down, M_down, H_up, M_up)

                if Mr_avg is not None and Ms_avg is not None and Ms_avg > 0:
                    squareness = Mr_avg / Ms_avg
                    print(f"  角形比 (S = Mr/Ms): {squareness:.3f}")

                if "CGS" in unit_mode:
                    H_plot_down, H_plot_up = H_down * 10000, H_up * 10000
                    M_plot_down, M_plot_up = M_down, M_up
                elif "Normalized" in unit_mode:
                    H_plot_down, H_plot_up = H_down, H_up
                    if Ms_avg is not None and Ms_avg > 0:
                        M_plot_down, M_plot_up = M_down / Ms_avg, M_up / Ms_avg
                    else:
                        M_plot_down, M_plot_up = M_down, M_up
                else:  # SI
                    H_plot_down, H_plot_up = H_down, H_up
                    M_plot_down, M_plot_up = M_down, M_up

                color = self.file_color_vars[idx].get()
                plot_kwargs = {
                    "marker": "o",
                    "markersize": params["marker_size"],
                    "linestyle": "-",
                    "linewidth": params["line_width"],
                }
                self.ax.plot(H_plot_down, M_plot_down, color=color, **plot_kwargs)
                self.ax.plot(
                    H_plot_up, M_plot_up, color=color, label=file.stem, **plot_kwargs
                )

                if "CGS" in unit_mode:
                    h_min_global = min(h_min_global, H_plot_down.min())
                    h_max_global = max(h_max_global, H_plot_up.max())
                else:
                    h_min_global = min(h_min_global, H_down.min())
                    h_max_global = max(h_max_global, H_up.max())

            except Exception as e:
                print(f"\nエラー: '{file.name}'の処理中に問題発生: {e}")
                import traceback

                traceback.print_exc(file=sys.stdout)
                continue

        if params["xlim_min"] is None and params["xlim_max"] is None:
            params["xlim_min"], params["xlim_max"] = h_min_global, h_max_global

        vsm_logic.format_axis(self.ax, self.fig, params, unit_mode)
        if self.show_legend_var.get() and any(self.ax.get_legend_handles_labels()[1]):
            self.ax.legend(
                fontsize=params["legend_fontsize"],
                loc="best",
                facecolor="white",
                edgecolor="#DDDDDD",
                labelcolor="black",
            )
        self.fig.tight_layout()


if __name__ == "__main__":
    root = tk.Tk()
    app = VSMApp(root)
    root.mainloop()