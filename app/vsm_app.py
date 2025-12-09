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

import app.event_handlers as event_handlers
import app.graph_manager as graph_manager
import app.state_manager as state_manager
import analysis.file_io as file_io
import analysis.calculations as vsm_logic


# -----------------------------------------------------------------------------
# GUIアプリケーションのクラス
# -----------------------------------------------------------------------------
class VSMApp:
    def __init__(self, root):
        self.vsm_data = []
        self._update_job = None
        self.all_metadata = {}
        self.analysis_results = []
        self.file_color_vars = []
        self.state = state_manager.StateManager()
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

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self._configure_styles()
        self.root.configure(bg=self.style.lookup(".", "background"))

        # --- メインレイアウト (PanedWindowベース) ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Left Pane (Single Notebook for all controls and outputs) ---
        left_pane = ttk.Frame(main_paned_window, padding=0)
        main_paned_window.add(left_pane, weight=2)  # Increased weight for left pane

        self.main_notebook = ttk.Notebook(left_pane)  # Renamed to main_notebook
        self.main_notebook.pack(fill=tk.BOTH, expand=True)
        self.main_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # --- Tab Frames ---
        tab_analysis = ttk.Frame(self.main_notebook, padding="10")
        tab_style = ttk.Frame(self.main_notebook, padding="10")
        self.results_tab = ttk.Frame(
            self.main_notebook, padding="10"
        )  # Use self.results_tab directly
        tab_export = ttk.Frame(self.main_notebook, padding="10")
        log_tab = ttk.Frame(self.main_notebook, padding="10")

        # --- Add tabs in specified order ---
        self.main_notebook.add(tab_analysis, text="解析")
        self.main_notebook.add(tab_style, text="グラフ設定")
        self.main_notebook.add(self.results_tab, text="解析結果")  # Results after Style
        self.main_notebook.add(tab_export, text="保存")
        self.main_notebook.add(log_tab, text="ログ")  # Log last

        # --- Log Text Widget ---
        self.log_text = scrolledtext.ScrolledText(
            log_tab, wrap=tk.WORD, font=("Consolas", 9), bg="white", fg="black"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Right Pane (Graph only) ---
        graph_frame = ttk.LabelFrame(main_paned_window, text=" グラフ ", padding=10)
        main_paned_window.add(graph_frame, weight=3)  # Decreased weight for graph frame
        graph_frame.grid_rowconfigure(1, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)

        # --- マネージャーとハンドラーのインスタンスを作成 ---
        # UI作成より前に初期化する必要がある
        self.graph_manager = graph_manager.GraphManager(self)
        self.event_handlers = event_handlers.EventHandlers(self)

        # --- Create controls and results tab structure ---
        self._create_analysis_controls(tab_analysis)
        self._create_style_controls(tab_style)
        self._create_export_controls(tab_export)
        self._create_results_tab()  # Builds the Treeview inside self.results_tab

        # --- Embed Graph ---
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
        self.graph_manager.update_graph()

    def _on_tab_changed(self, event):
        """Event handler for when the notebook tab is changed."""
        selected_tab_id = event.widget.select()
        # Ensure we have a valid tab selected
        if not selected_tab_id:
            return

        try:
            tab_text = event.widget.tab(selected_tab_id, "text")

            # If the user selects the results tab and it has the notification, reset it
            if selected_tab_id == str(self.results_tab) and tab_text.endswith("*"):
                event.widget.tab(selected_tab_id, text="解析結果")
        except tk.TclError:
            # This can happen if the tab is in the process of being destroyed
            pass

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
            file_frame,
            text="ファイルを選択",
            command=self.event_handlers.load_files,
            padding="10 5",
        ).pack(fill=tk.X)
        self.info_button = ttk.Button(
            file_frame,
            text="測定情報を表示",
            command=self.event_handlers.show_metadata_window,
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
            textvariable=self.state.unit_mode_var,
            values=["SI (T, kA/m)", "CGS (Oe, emu/cm³)", "Normalized (T, M/Ms)"],
            state="readonly",
        )
        unit_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 5))

        ttk.Label(settings_frame, text="基板面積 (cm²):").grid(
            row=2, column=0, sticky="w"
        )
        self.area_entry = ttk.Entry(
            settings_frame, textvariable=self.state.area_var, width=10
        )
        self.area_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=(5, 0))
        self.offset_check = ttk.Checkbutton(
            settings_frame,
            text="磁化オフセット補正",
            variable=self.state.offset_correction_var,
        )
        self.offset_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.legend_check = ttk.Checkbutton(
            settings_frame, text="凡例を表示", variable=self.state.show_legend_var
        )
        self.legend_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))

        thickness_outer_frame = ttk.LabelFrame(
            parent, text=" 膜厚設定 (nm) ", padding="10"
        )
        thickness_outer_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        thickness_canvas = tk.Canvas(
            thickness_outer_frame,
            borderwidth=0,
            background=self.style.lookup(".", "background"),
            highlightthickness=0,
            height=100,
        )
        thickness_scrollbar = ttk.Scrollbar(
            thickness_outer_frame, orient="vertical", command=thickness_canvas.yview
        )
        self.thickness_scrollable_frame = ttk.Frame(thickness_canvas)

        self.thickness_scrollable_frame.bind(
            "<Configure>",
            lambda e: thickness_canvas.configure(
                scrollregion=thickness_canvas.bbox("all")
            ),
        )

        thickness_canvas.create_window(
            (0, 0), window=self.thickness_scrollable_frame, anchor="nw"
        )
        thickness_canvas.configure(yscrollcommand=thickness_scrollbar.set)

        thickness_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        thickness_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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

        demag_canvas.create_window(
            (0, 0), window=self.demag_scrollable_frame, anchor="nw"
        )
        demag_canvas.configure(yscrollcommand=demag_scrollbar.set)

        demag_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        demag_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.apply_to_all_button = ttk.Button(
            demag_outer_frame,
            text="一番上の設定を全ファイルに適用",
            command=self.event_handlers.apply_first_file_settings_to_all,
            state=tk.DISABLED,
        )
        self.apply_to_all_button.pack(fill=tk.X, pady=(5, 0), padx=5)

        ms_frame = ttk.LabelFrame(parent, text=" 飽和磁化 (Ms) 計算 ", padding="10")
        ms_frame.pack(fill=tk.X, pady=(0, 10))
        self.ms_settings_button = ttk.Button(
            ms_frame,
            text="計算範囲を手動指定...",
            command=self.event_handlers.show_ms_settings_window,
            state=tk.DISABLED,
        )
        self.ms_settings_button.pack(fill=tk.X, pady=5)

    def _create_style_controls(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        axis_grid_frame = ttk.LabelFrame(parent, text=" 軸とグリッド ", padding="10")
        axis_grid_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            axis_grid_frame, text="グリッド線を表示", variable=self.state.show_grid_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            axis_grid_frame,
            text="原点線を表示",
            variable=self.state.show_zero_lines_var,
        ).pack(anchor="w", pady=(5, 0))
        plot_frame = ttk.LabelFrame(parent, text=" プロット ", padding="10")
        plot_frame.pack(fill=tk.X, pady=(0, 10))
        plot_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(plot_frame, text="マーカーサイズ:").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(plot_frame, textvariable=self.state.marker_size_var, width=10).grid(
            row=0, column=1, sticky="ew", padx=5, pady=(0, 5)
        )
        ttk.Label(plot_frame, text="線幅:").grid(
            row=1, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(plot_frame, textvariable=self.state.line_width_var, width=10).grid(
            row=1, column=1, sticky="ew", padx=5, pady=(0, 5)
        )
        font_frame = ttk.LabelFrame(parent, text=" フォントサイズ ", padding="10")
        font_frame.pack(fill=tk.X, pady=(0, 10))
        font_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(font_frame, text="軸ラベル:").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(
            font_frame, textvariable=self.state.axis_label_fontsize_var, width=10
        ).grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 5))
        ttk.Label(font_frame, text="目盛り:").grid(
            row=1, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(
            font_frame, textvariable=self.state.tick_label_fontsize_var, width=10
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=(0, 5))
        ttk.Label(font_frame, text="凡例:").grid(
            row=2, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(
            font_frame, textvariable=self.state.legend_fontsize_var, width=10
        ).grid(row=2, column=1, sticky="ew", padx=5, pady=(0, 5))
        self.individual_color_frame = ttk.LabelFrame(
            parent, text=" ファイルリストと描画順 ", padding="10"
        )
        self.individual_color_frame.pack(fill=tk.X, pady=(10, 0))
        self.individual_color_frame.grid_columnconfigure(1, weight=1)

        # --- 詳細設定ボタンを追加 ---
        adv_settings_frame = ttk.LabelFrame(parent, text=" 詳細設定 ", padding="10")
        adv_settings_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            adv_settings_frame,
            text="軸・凡例・線のスタイルを設定...",
            command=self._show_advanced_style_window,
        ).pack(fill=tk.X, pady=5)

        axes_frame = ttk.LabelFrame(parent, text=" 描画範囲 ", padding="10")
        axes_frame.pack(fill=tk.X, pady=(0, 10))
        axes_frame.grid_columnconfigure(1, weight=1)
        axes_frame.grid_columnconfigure(3, weight=1)
        ttk.Label(axes_frame, text="X軸 (T):").grid(row=0, column=0, sticky="w")
        ttk.Entry(axes_frame, textvariable=self.state.xlim_min_var, width=7).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(axes_frame, text="～").grid(row=0, column=2)
        ttk.Entry(axes_frame, textvariable=self.state.xlim_max_var, width=7).grid(
            row=0, column=3, sticky="ew"
        )
        ttk.Label(axes_frame, text="Y軸 (kA/m):").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        ttk.Entry(axes_frame, textvariable=self.state.ylim_min_var, width=7).grid(
            row=1, column=1, sticky="ew", pady=(5, 0)
        )
        ttk.Label(axes_frame, text="～").grid(row=1, column=2, pady=(5, 0))
        ttk.Entry(axes_frame, textvariable=self.state.ylim_max_var, width=7).grid(
            row=1, column=3, sticky="ew", pady=(5, 0)
        )

    def _show_advanced_style_window(self):
        # このメソッドは次のステップで実装します
        messagebox.showinfo("未実装", "この機能は現在開発中です。", parent=self.root)

    def _create_export_controls(self, parent):
        save_settings_frame = ttk.LabelFrame(
            parent, text=" 画像サイズ設定 ", padding="10"
        )
        save_settings_frame.pack(fill=tk.X, pady=(0, 10))
        save_settings_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(save_settings_frame, text="幅:").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(
            save_settings_frame, textvariable=self.state.save_width_var, width=10
        ).grid(row=0, column=1, sticky="ew", padx=5, pady=3)
        ttk.Label(save_settings_frame, text="inch").grid(row=0, column=2, sticky="w")
        ttk.Label(save_settings_frame, text="高さ:").grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Entry(
            save_settings_frame, textvariable=self.state.save_height_var, width=10
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=3)
        ttk.Label(save_settings_frame, text="inch").grid(row=1, column=2, sticky="w")
        dpi_frame = ttk.LabelFrame(parent, text=" 解像度設定 ", padding="10")
        dpi_frame.pack(fill=tk.X, pady=(0, 10))
        dpi_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(dpi_frame, text="DPI:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(dpi_frame, textvariable=self.state.save_dpi_var, width=10).grid(
            row=0, column=1, sticky="ew", padx=5, pady=3
        )
        save_button_frame = ttk.Frame(parent, padding="10 10 10 0")
        save_button_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(
            save_button_frame,
            text="画像を保存 (Save Image)",
            command=self.event_handlers.save_figure,
            padding="10",
        ).pack(fill=tk.X, expand=True, side=tk.BOTTOM)

    def _add_traces(self):
        trace_vars = [
            self.state.area_var,
            self.state.offset_correction_var,
            self.state.show_legend_var,
            self.state.marker_size_var,
            self.state.line_width_var,
            self.state.axis_label_fontsize_var,
            self.state.tick_label_fontsize_var,
            self.state.legend_fontsize_var,
            self.state.xlim_min_var,
            self.state.xlim_max_var,
            self.state.ylim_min_var,
            self.state.ylim_max_var,
            self.state.show_grid_var,
            self.state.show_zero_lines_var,
            self.state.unit_mode_var,
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
                row_frame,
                text="↑",
                width=3,
                command=lambda idx=i: self.event_handlers.move_file_up(idx),
            )
            up_button.pack(side=tk.LEFT, padx=(0, 2))
            if i == 0:
                up_button.config(state=tk.DISABLED)

            down_button = ttk.Button(
                row_frame,
                text="↓",
                width=3,
                command=lambda idx=i: self.event_handlers.move_file_down(idx),
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
                command=lambda idx=i: self.event_handlers.choose_individual_color(idx),
            ).pack(side=tk.RIGHT)

    def _schedule_update(self, *args):
        if self._update_job:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(250, self.graph_manager.update_graph)

    def log_message(self, message):
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _update_demag_settings_ui(self):
        # Clear existing widgets
        for widget in self.demag_scrollable_frame.winfo_children():
            widget.destroy()

        if not self.vsm_data:
            ttk.Label(
                self.demag_scrollable_frame, text="ファイルが読み込まれていません。"
            ).pack()
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
                    "pos_range": ("0.5", "1.2"),
                    "neg_range": ("-1.2", "-0.5"),
                    "link_ranges": True,
                }

            # Store vars in the data dict to keep them alive
            if "demag_vars" not in data:
                data["demag_vars"] = {}

            current_settings = data["demag_settings"]

            enabled_var = tk.BooleanVar(value=current_settings.get("enabled", True))
            manual_var = tk.BooleanVar(value=current_settings.get("manual", False))
            pos_min_var = tk.StringVar(
                value=current_settings.get("pos_range", ["1.5", "2.0"])[0]
            )
            pos_max_var = tk.StringVar(
                value=current_settings.get("pos_range", ["1.5", "2.0"])[1]
            )
            neg_min_var = tk.StringVar(
                value=current_settings.get("neg_range", ["-2.0", "-1.5"])[0]
            )
            neg_max_var = tk.StringVar(
                value=current_settings.get("neg_range", ["-2.0", "-1.5"])[1]
            )
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
            frame = ttk.LabelFrame(
                self.demag_scrollable_frame, text=data["path"].name, padding=10
            )
            frame.pack(fill=tk.X, expand=True, pady=5, padx=5)
            frame.grid_columnconfigure(1, weight=1)
            frame.grid_columnconfigure(3, weight=1)

            enabled_check = ttk.Checkbutton(
                frame, text="反磁性補正", variable=enabled_var
            )
            enabled_check.grid(row=0, column=0, sticky="w")

            manual_check = ttk.Checkbutton(frame, text="手動範囲", variable=manual_var)
            manual_check.grid(row=0, column=1, sticky="w")

            link_check = ttk.Checkbutton(
                frame, text="正負の範囲を連動", variable=link_var
            )
            link_check.grid(row=0, column=2, columnspan=2, sticky="w")

            pos_min_entry = ttk.Entry(frame, textvariable=pos_min_var, width=7)
            pos_min_entry.grid(row=1, column=1, sticky="ew", padx=(0, 2))
            ttk.Label(frame, text="～").grid(row=1, column=2)
            pos_max_entry = ttk.Entry(frame, textvariable=pos_max_var, width=7)
            pos_max_entry.grid(row=1, column=3, sticky="ew", padx=(2, 0))
            ttk.Label(frame, text=" (正 H)").grid(row=1, column=4, sticky="w")

            neg_min_entry = ttk.Entry(frame, textvariable=neg_min_var, width=7)
            neg_min_entry.grid(row=2, column=1, sticky="ew", pady=(5, 0), padx=(0, 2))
            ttk.Label(frame, text="～").grid(row=2, column=2, pady=(5, 0))
            neg_max_entry = ttk.Entry(frame, textvariable=neg_max_var, width=7)
            neg_max_entry.grid(row=2, column=3, sticky="ew", pady=(5, 0), padx=(2, 0))
            ttk.Label(frame, text=" (負 H)").grid(
                row=2, column=4, sticky="w", pady=(5, 0)
            )

            manual_entries = [
                pos_min_entry,
                pos_max_entry,
                neg_min_entry,
                neg_max_entry,
            ]

            # --- Traces for real-time update ---
            # Pass the internal var name to the callback to identify the trigger
            enabled_var.trace_add(
                "write",
                lambda *a, idx=i, v=enabled_var: self._on_demag_setting_change(idx, v),
            )
            manual_var.trace_add(
                "write",
                lambda *a, idx=i, v=manual_var: self._on_demag_setting_change(idx, v),
            )
            link_var.trace_add(
                "write",
                lambda *a, idx=i, v=link_var: self._on_demag_setting_change(idx, v),
            )
            pos_min_var.trace_add(
                "write",
                lambda *a, idx=i, v=pos_min_var: self._on_demag_setting_change(idx, v),
            )
            pos_max_var.trace_add(
                "write",
                lambda *a, idx=i, v=pos_max_var: self._on_demag_setting_change(idx, v),
            )
            neg_min_var.trace_add(
                "write",
                lambda *a, idx=i, v=neg_min_var: self._on_demag_setting_change(idx, v),
            )
            neg_max_var.trace_add(
                "write",
                lambda *a, idx=i, v=neg_max_var: self._on_demag_setting_change(idx, v),
            )

            # Add trace to toggle entries and also trigger update
            manual_var.trace_add(
                "write",
                lambda *args, w=manual_entries, v=manual_var: _toggle_manual_entries(
                    w, v
                ),
            )
            _toggle_manual_entries(manual_entries, manual_var)

        # Update button state
        num_files = len(self.vsm_data)
        self.apply_to_all_button.config(
            state=tk.NORMAL if num_files > 1 else tk.DISABLED
        )
        self.ms_settings_button.config(
            state=tk.NORMAL if num_files > 0 else tk.DISABLED
        )

    def _update_thickness_settings_ui(self):
        # Clear existing widgets
        for widget in self.thickness_scrollable_frame.winfo_children():
            widget.destroy()

        if not self.vsm_data:
            ttk.Label(
                self.thickness_scrollable_frame, text="ファイルが読み込まれていません。"
            ).pack()
            return

        for i, data in enumerate(self.vsm_data):
            if "thickness_var" not in data:
                data["thickness_var"] = tk.StringVar(value="100.0")

            thickness_var = data["thickness_var"]

            frame = ttk.Frame(self.thickness_scrollable_frame, padding=(0, 0, 10, 0))
            frame.pack(fill=tk.X, expand=True, pady=2, padx=5)
            frame.grid_columnconfigure(1, weight=1)

            display_name = (
                (data["path"].name[:25] + "..")
                if len(data["path"].name) > 27
                else data["path"].name
            )

            ttk.Label(frame, text=display_name).grid(row=0, column=0, sticky="w")

            entry = ttk.Entry(frame, textvariable=thickness_var, width=10)
            entry.grid(row=0, column=1, sticky="e", padx=5)

            ttk.Label(frame, text="nm").grid(row=0, column=2, sticky="w")

            thickness_var.trace_add(
                "write",
                lambda *a, idx=i: self._on_thickness_change(idx),
            )

    def _on_thickness_change(self, file_index):
        if file_index >= len(self.vsm_data):
            return
        # This just triggers a graph update. The value is already in the StringVar.
        self._schedule_update()

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
                pass  # Ignore errors during typing

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

    def _create_results_tab(self):
        """Creates the structure of the results table tab."""
        frame = ttk.Frame(self.results_tab, padding="10")
        frame.pack(expand=True, fill="both")

        # --- Treeview (Table) ---
        columns = ("filename", "ms", "mr", "hc", "sq")
        self.results_tree = ttk.Treeview(frame, columns=columns, show="headings")

        # Define headings
        self.results_tree.heading("filename", text="ファイル名")
        self.results_tree.heading("ms", text="飽和磁化 Ms (kA/m)")
        self.results_tree.heading("mr", text="残留磁化 Mr (kA/m)")
        self.results_tree.heading("hc", text="保磁力 Hc (Oe)")
        self.results_tree.heading("sq", text="角形比 (S = Mr/Ms)")

        # Define column properties
        self.results_tree.column("filename", anchor="w", width=200)
        self.results_tree.column("ms", anchor="e", width=150)
        self.results_tree.column("mr", anchor="e", width=150)
        self.results_tree.column("hc", anchor="e", width=120)
        self.results_tree.column("sq", anchor="e", width=150)

        # Scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.results_tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.results_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # --- Action Buttons ---
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        copy_button = ttk.Button(
            button_frame,
            text="クリップボードにコピー",
            command=self._copy_results_to_clipboard,
        )
        copy_button.pack(side="left", padx=5)

    def _update_results_table(self):
        """Clears and repopulates the results table with the latest analysis data."""
        # Clear existing data
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)

        if not self.analysis_results:
            return

        # Insert new data
        for res in self.analysis_results:
            ms_str = f"{res['Ms']:.3f}" if res["Ms"] is not None else "N/A"
            mr_str = f"{res['Mr']:.3f}" if res["Mr"] is not None else "N/A"
            hc_str = f"{res['Hc_Oe']:.2f}" if res["Hc_Oe"] is not None else "N/A"
            sq_str = (
                f"{res['squareness']:.3f}" if res["squareness"] is not None else "N/A"
            )
            self.results_tree.insert(
                "", "end", values=(res["filename"], ms_str, mr_str, hc_str, sq_str)
            )

    def _copy_results_to_clipboard(self):
        if not self.analysis_results:
            return

        try:
            # Create header row
            headers = [
                "ファイル名",
                "飽和磁化 Ms (kA/m)",
                "残留磁化 Mr (kA/m)",
                "保磁力 Hc (Oe)",
                "角形比 (S = Mr/Ms)",
            ]
            tsv_data = "\t".join(headers) + "\n"

            # Create data rows
            for res in self.analysis_results:
                ms_str = f"{res['Ms']:.3f}" if res["Ms"] is not None else ""
                mr_str = f"{res['Mr']:.3f}" if res["Mr"] is not None else ""
                hc_str = f"{res['Hc_Oe']:.2f}" if res["Hc_Oe"] is not None else ""
                sq_str = (
                    f"{res['squareness']:.3f}" if res["squareness"] is not None else ""
                )
                row = [res["filename"], ms_str, mr_str, hc_str, sq_str]
                tsv_data += "\t".join(row) + "\n"

            self.root.clipboard_clear()
            self.root.clipboard_append(tsv_data)

            messagebox.showinfo(
                "成功", "クリップボードにコピーしました。", parent=self.root
            )

        except Exception as e:
            messagebox.showerror(
                "エラー", f"コピー中にエラーが発生しました: {e}", parent=self.root
            )
