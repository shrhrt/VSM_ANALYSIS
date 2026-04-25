# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.vsm_app import VSMApp


class AnalysisTab:
    """
    解析タブ内のコントロール（ファイル操作、解析設定など）を構築・管理するクラス。
    vsm_app.py の肥大化を防ぐためにUI構築を分離しています。
    """

    def __init__(self, parent: ttk.Frame, app: "VSMApp") -> None:
        self.parent = parent
        self.app = app

        # 他のモジュールからアクセスされる主要なウィジェットの参照を保持
        self.info_button: ttk.Button
        self.thickness_scrollable_frame: ttk.Frame
        self.demag_scrollable_frame: ttk.Frame
        self.apply_to_all_button: ttk.Button
        self.ms_settings_button: ttk.Button

        self._build_ui()

    def _build_ui(self) -> None:
        parent = self.parent

        # --- ファイル操作フレーム ---
        file_frame = ttk.LabelFrame(parent, text=" ファイル ", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(
            file_frame,
            text="ファイルを選択 (新規)",
            command=self.app.event_handlers.load_files,
            padding="10 5",
            style="Accent.TButton",
        ).pack(fill=tk.X)
        ttk.Button(
            file_frame,
            text="ファイルを追加...",
            command=self.app.event_handlers.add_files,
            padding="10 5",
        ).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(
            file_frame,
            text="ファイルを全て削除",
            command=self.app.event_handlers.clear_all_files,
            padding="10 5",
        ).pack(fill=tk.X, pady=(5, 0))
        self.info_button = ttk.Button(
            file_frame,
            text="測定情報を表示",
            command=self.app.event_handlers.show_metadata_window,
            state=tk.DISABLED,
        )
        self.info_button.pack(fill=tk.X, pady=(5, 0))

        # --- 解析設定フレーム ---
        settings_frame = ttk.LabelFrame(parent, text=" 解析設定 ", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        settings_frame.grid_columnconfigure(1, weight=1)

        unit_label = ttk.Label(settings_frame, text="表示単位系:")
        unit_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        unit_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.app.state.unit_mode_var,
            values=["SI (T, kA/m)", "CGS (Oe, emu/cm³)", "Normalized (T, M/Ms)"],
            state="readonly",
        )
        unit_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 5))

        ttk.Checkbutton(
            settings_frame,
            text="磁化オフセット補正",
            variable=self.app.state.offset_correction_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Checkbutton(
            settings_frame, text="凡例を表示", variable=self.app.state.show_legend_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # --- 膜厚・面積設定フレーム ---
        thickness_outer_frame = ttk.LabelFrame(
            parent, text=" 膜厚・面積設定 ", padding="10"
        )
        thickness_outer_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        thickness_canvas = tk.Canvas(
            thickness_outer_frame,
            borderwidth=0,
            background=self.app.get_bg_color(),
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

        # --- 反磁性補正フレーム ---
        demag_outer_frame = ttk.LabelFrame(parent, text=" 反磁性補正 ", padding="10")
        demag_outer_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        demag_canvas = tk.Canvas(
            demag_outer_frame,
            borderwidth=0,
            background=self.app.get_bg_color(),
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
            command=self.app.event_handlers.apply_first_file_settings_to_all,
            state=tk.DISABLED,
        )
        self.apply_to_all_button.pack(fill=tk.X, pady=(5, 0), padx=5)

        # --- Ms計算フレーム ---
        ms_frame = ttk.LabelFrame(parent, text=" 飽和磁化 (Ms) 計算 ", padding="10")
        ms_frame.pack(fill=tk.X, pady=(0, 10))
        self.ms_settings_button = ttk.Button(
            ms_frame,
            text="計算範囲を手動指定...",
            command=self.app.event_handlers.show_ms_settings_window,
            state=tk.DISABLED,
        )
        self.ms_settings_button.pack(fill=tk.X, pady=5)
