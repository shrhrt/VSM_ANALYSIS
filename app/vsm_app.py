# -*- coding: utf-8 -*-
import platform
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import sv_ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from tkinterdnd2 import DND_FILES, TkinterDnD

import app.analysis_tab as analysis_tab
import app.event_handlers as event_handlers
import app.graph_manager as graph_manager
import app.state_manager as state_manager
import app.theme as theme
from analysis.tex_utils import tex_to_display


# -----------------------------------------------------------------------------
# ソートユーティリティ
# -----------------------------------------------------------------------------

_NUMERIC_COLS = {"ms", "mr", "hc", "sq"}


def _make_sort_key(value: str, col: str) -> tuple:
    """Treeview のソートキーを生成する。N/A は常に末尾、数値列は数値比較。"""
    if value == "N/A":
        return (1, 0.0, "")
    if col in _NUMERIC_COLS:
        try:
            return (0, float(value), "")
        except ValueError:
            return (1, 0.0, "")
    return (0, 0.0, value)


# -----------------------------------------------------------------------------
# GUIアプリケーションのクラス
# -----------------------------------------------------------------------------
class VSMApp:
    """
    VSMデータ解析アプリケーションのメインGUIクラス。
    UIの構築、各タブの管理、および全体データの保持を行う。
    """

    def __init__(self, root: TkinterDnD.Tk) -> None:
        """
        VSMAppのインスタンスを初期化し、GUIを構築。

        Args:
            root (TkinterDnD.Tk): Tkinterのメインウィンドウインスタンス。
        """
        # --- Matplotlibの日本語フォント設定 ---
        # より確実に日本語フォントを適用するための設定
        # OSに応じてフォントを自動で設定
        os_name = platform.system()
        if os_name == "Windows":
            # 利用可能なフォントを優先順位順に指定
            plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic"]
        elif os_name == "Darwin":  # macOS
            plt.rcParams["font.family"] = "Hiragino Sans"
        else:  # Linuxなど
            # IPAフォントなどがインストールされていることを期待
            # 必要に応じて 'IPAexGothic' などを指定
            plt.rcParams["font.family"] = ["IPAexGothic", "sans-serif"]

        # フォント変更時に数式レンダリングの警告が出るのを防ぐ(Windowsはこれがなくても動作を確認)
        plt.rcParams["axes.unicode_minus"] = False
        # ------------------------------------

        self.vsm_data: List[Dict[str, Any]] = []
        self._update_job: Optional[str] = None  # strまたはNone。
        self._sort_ascending: Dict[str, bool] = {}
        self.all_metadata: Dict[str, Dict[str, str]] = {}
        self.analysis_results: List[Dict[str, Any]] = []
        self.file_color_vars: List[tk.StringVar] = []
        self.state: state_manager.StateManager = state_manager.StateManager()
        self.base_colors: List[str] = [  # 1本目を赤、2本目を青、3本目を黄緑に設定
            "red",
            "blue",
            "limegreen",
            "purple",
            "orange",
            "cyan",
            "magenta",
            "brown",
        ]

        self.root: TkinterDnD.Tk = root
        self.root.title("VSM Data Analyzer")
        self.root.minsize(1100, 700)
        self.root.state('zoomed')

        self.style = ttk.Style(self.root)
        # モダンテーマ「sv_ttk」を適用（デフォルトはライトテーマ）
        sv_ttk.set_theme("light")
        theme.apply_theme(self.style)
        self.root.configure(bg=self.get_bg_color())

        # --- ステータスバー（先にpackしてから main_frame をpackする必要がある）---
        self._create_status_bar()

        # --- メインレイアウトの構築 ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_paned_window = ttk.PanedWindow(
            main_frame, orient=tk.HORIZONTAL
        )  # 境界線をマウスでドラッグしてサイズ調整
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- 左ペイン (タブ管理) ---
        left_pane = ttk.Frame(main_paned_window, padding=0)
        main_paned_window.add(left_pane, weight=2)

        self.main_notebook = ttk.Notebook(left_pane)
        self.main_notebook.pack(fill=tk.BOTH, expand=True)
        self.main_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # --- 各タブのフレーム構築 ---
        tab_analysis = ttk.Frame(self.main_notebook, padding="5")
        tab_style = ttk.Frame(self.main_notebook, padding="5")
        self.results_tab = ttk.Frame(self.main_notebook, padding="5")
        tab_export = ttk.Frame(self.main_notebook, padding="5")
        log_tab = ttk.Frame(self.main_notebook, padding="5")

        self.main_notebook.add(tab_analysis, text="解析")
        self.main_notebook.add(tab_style, text="グラフ設定")
        self.main_notebook.add(self.results_tab, text="解析結果")
        self.main_notebook.add(tab_export, text="保存")
        self.main_notebook.add(log_tab, text="ログ")

        # --- ログタブ: ツールバー＋テキストエリア ---
        log_toolbar = ttk.Frame(log_tab)
        log_toolbar.pack(fill=tk.X, pady=(0, 4))
        theme.danger_button(
            log_toolbar, text="ログをクリア", command=self._clear_log,
            padx=8, pady=3,
        ).pack(side=tk.RIGHT)

        self.log_text = scrolledtext.ScrolledText(
            log_tab, wrap=tk.WORD, font=theme.FONT_LOG, state=tk.DISABLED,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log_text.tag_configure("log_ts",      foreground=theme.LOG_FG_TS)
        self.log_text.tag_configure("log_info",    foreground=theme.LOG_FG_INFO)
        self.log_text.tag_configure(
            "log_success", foreground=theme.LOG_FG_SUCCESS,
            font=(theme.FONT_FAMILY, 10, "bold"),
        )
        self.log_text.tag_configure(
            "log_error", foreground=theme.LOG_FG_ERROR,
            font=(theme.FONT_FAMILY, 10, "bold"),
        )

        # --- 右ペイン (グラフ表示領域) --- ※左、右の順にaddされる
        graph_frame = ttk.Frame(main_paned_window, padding=5)
        main_paned_window.add(graph_frame, weight=3)
        graph_frame.grid_rowconfigure(1, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)

        # --- 依存モジュールの初期化 ---
        # Note: UI構築で必要になるため、先にインスタンス化
        self.graph_manager = graph_manager.GraphManager(self)
        self.event_handlers = event_handlers.EventHandlers(self)

        # メニューの作成
        self._create_menu()

        # --- 各タブのUI構築とコントローラーの初期化 ---
        self.analysis_tab_view = analysis_tab.AnalysisTab(tab_analysis, self)

        # 外部モジュールからのアクセスを容易にするため、
        # 主要なUIコンポーネントをAppクラスのプロパティとして公開
        self.info_button = self.analysis_tab_view.info_button
        self.thickness_scrollable_frame = (
            self.analysis_tab_view.thickness_scrollable_frame
        )
        self.demag_scrollable_frame = self.analysis_tab_view.demag_scrollable_frame
        self.apply_to_all_button = self.analysis_tab_view.apply_to_all_button
        self.ms_settings_button = self.analysis_tab_view.ms_settings_button

        # その他のタブ構築
        self._create_style_controls(tab_style)
        self._create_export_controls(tab_export)
        self._create_results_tab()  # self.results_tabのため空欄

        # --- グラフ描画領域のTkinterへの埋め込み ---
        self.fig = plt.figure(figsize=(9, 9), facecolor="white")
        self.ax = self.fig.add_subplot(111)

        # FigureCanvasTkAgg:Matplotlibで作ったfigを、Tkinterの『キャンバス』に変換。
        # master=graph_frame:graph_frameの中にはめ込む。
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)

        # Matplotlib標準のナビゲーションツールバーを追加
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame, pack_toolbar=False)
        toolbar.config(background=self.get_bg_color())

        # メッセージラベルの色の調整
        toolbar._message_label.config(
            background=self.get_bg_color(), foreground="black"
        )
        # --- ツールバーのUI調整 ---
        for button in toolbar.winfo_children():
            if isinstance(button, (tk.Button, tk.Checkbutton)):
                # 設定の上書き
                button.config(
                    background=self.get_bg_color(),
                    foreground="black",
                    highlightbackground=self.get_bg_color(),
                )
        toolbar.update()

        # --- グラフとツールバーの配置 ---
        # ツールバーを上段、グラフキャンバスを下段に配置
        # ウィンドウサイズ変更時にグラフだけが拡大するように設定
        toolbar.grid(row=0, column=0, sticky="ew", padx=5)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # --- グラフの初期化 ---
        self._add_traces()
        self.graph_manager.update_graph()

        # --- ドラッグ＆ドロップの設定 ---
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.event_handlers.on_drop_files)

    def _create_status_bar(self) -> None:
        """ウィンドウ下部にステータスバーを作成する。"""
        bar = ttk.Frame(self.root, relief="sunken", padding="3 2")
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        self._status_files_var = tk.StringVar(value="ファイル未読込")
        self._status_time_var = tk.StringVar(value="")

        ttk.Label(bar, textvariable=self._status_files_var, anchor="w", style="Status.TLabel").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Separator(bar, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2
        )
        ttk.Label(bar, textvariable=self._status_time_var, anchor="w", style="Status.TLabel").pack(
            side=tk.LEFT
        )

    def update_status_bar(self) -> None:
        """ステータスバーの表示を更新する。"""
        n = len(self.vsm_data)
        self._status_files_var.set(
            "ファイル未読込" if n == 0 else f"{n} ファイル読込済"
        )
        self._status_time_var.set(
            f"最終更新: {datetime.now().strftime('%H:%M:%S')}" if n > 0 else ""
        )

    def get_bg_color(self) -> str:
        """現在のテーマに応じた背景色を取得します。"""
        color = self.style.lookup("TFrame", "background")
        if not color:
            color = "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa"
        return color

    def _create_menu(self) -> None:
        """メニューバーを作成してメインウィンドウに配置"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # --- ファイルメニュー ---
        # メニューをウィンドウから切り離す機能（点線）を無効化
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)

        file_menu.add_command(
            label="セッションを読み込み...",
            command=lambda: self.event_handlers.load_session(),
        )
        file_menu.add_command(
            label="セッションを保存...",
            command=lambda: self.event_handlers.save_session(),
        )

        # --- ツールメニュー ---
        tool_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ツール", menu=tool_menu)
        tool_menu.add_command(
            label="PPMS (.dat) を VSM形式に変換...",
            command=lambda: self.event_handlers.convert_dat_file(),
        )

        # --- 表示メニュー ---
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="表示", menu=view_menu)
        # テーマ（ライト/ダーク）を切り替える
        view_menu.add_command(
            label="ライト / ダーク 切り替え",
            command=self._toggle_theme,
        )

        # --- ヘルプメニュー ---
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(
            label="計算ロジックの解説",
            command=self.event_handlers.show_calculation_logic_window,
        )

    def _toggle_theme(self) -> None:
        """テーマを切り替え、カスタムスタイルと tk.Label 背景色を再適用する。"""
        sv_ttk.toggle_theme()
        theme.apply_theme(self.style)
        theme.refresh_section_title_bg(self.root)

    def _on_tab_changed(self, event: tk.Event) -> None:
        """
        タブが切り替わったときのイベントハンドラ。

        Args:
            event (tk.Event): 発生したイベント情報。
        """
        # 型ガード
        notebook = event.widget
        if not isinstance(notebook, ttk.Notebook):
            return

        # Notebookから、現在選択されているタブの内部IDを取得
        selected_tab_id = notebook.select()

        # タブが全く選択されていない場合は早期リターン
        if not selected_tab_id:
            return

        try:
            # 選択されたタブの現在の表示テキストを取得
            tab_text = notebook.tab(selected_tab_id, "text")

            # 解析結果タブが選択され、かつ名前に未読通知マーク「*」が付いている場合、それを消去して既読状態に
            if selected_tab_id == str(self.results_tab) and tab_text.endswith("*"):
                notebook.tab(selected_tab_id, text="解析結果")
        except tk.TclError:
            # タブが破棄されている最中にイベントが発火した場合の例外を安全に無視
            pass

    def _create_style_controls(self, parent: ttk.Frame) -> None:
        """
        グラフ設定タブ内のコントロール（軸、プロット、描画範囲など）を作成。

        Args:
            parent (ttk.Frame): コントロールを配置する親フレーム。
        """
        parent.grid_columnconfigure(1, weight=1)
        axis_grid_section, axis_grid_frame = theme.make_section(parent, "軸とグリッド")
        axis_grid_section.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            axis_grid_frame, text="グリッド線を表示", variable=self.state.show_grid_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            axis_grid_frame,
            text="原点線を表示",
            variable=self.state.show_zero_lines_var,
        ).pack(anchor="w", pady=(5, 0))

        # 原点線の色と線種設定
        zero_line_settings_frame = ttk.Frame(axis_grid_frame)
        zero_line_settings_frame.pack(fill=tk.X, padx=15, pady=(0, 5))
        zero_line_settings_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(zero_line_settings_frame, text="色:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Combobox(
            zero_line_settings_frame,
            textvariable=self.state.zero_line_color_var,
            values=["black", "grey", "red", "blue", "green"],
        ).grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(zero_line_settings_frame, text="線種:").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        ttk.Combobox(
            zero_line_settings_frame,
            textvariable=self.state.zero_line_linestyle_var,
            values=["-", "--", "-.", ":"],  # Solid, Dashed, Dash-dot, Dotted
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 0))

        # --- プロット設定フレーム（マーカーサイズ、線幅など） ---
        plot_section, plot_frame = theme.make_section(parent, "プロット")
        plot_section.pack(fill=tk.X, pady=(0, 10))
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

        # --- フォントサイズ設定フレーム（軸ラベル、目盛り、凡例など） ---
        font_section, font_frame = theme.make_section(parent, "フォントサイズ")
        font_section.pack(fill=tk.X, pady=(0, 10))
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

        # --- 個別ファイル設定フレーム（リスト順、プロット色など） ---
        individual_color_section, self.individual_color_frame = theme.make_section(
            parent, "ファイルリストと描画順"
        )
        individual_color_section.pack(fill=tk.X, pady=(10, 0))
        self.individual_color_frame.grid_columnconfigure(1, weight=1)

        # --- 詳細設定ボタン ---
        adv_settings_section, adv_settings_frame = theme.make_section(parent, "詳細設定")
        adv_settings_section.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            adv_settings_frame,
            text="凡例・線のスタイル・軸を設定...",
            command=self.event_handlers.show_advanced_style_window,
        ).pack(fill=tk.X, pady=5)

        axes_section, axes_frame = theme.make_section(parent, "描画範囲")
        axes_section.pack(fill=tk.X, pady=(0, 10))
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

    def _create_export_controls(self, parent: ttk.Frame) -> None:
        """
        保存タブ内のコントロール（画像サイズ、DPI、保存ボタンなど）を作成。

        Args:
            parent (ttk.Frame): コントロールを配置する親フレーム。
        """
        # --- 1. 画像サイズ設定フレームの構築 ---
        save_settings_section, save_settings_frame = theme.make_section(parent, "画像サイズ設定")
        save_settings_section.pack(fill=tk.X, pady=(0, 10))
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

        # --- 2. 解像度(DPI)設定フレームの構築 ---
        dpi_section, dpi_frame = theme.make_section(parent, "解像度設定")
        dpi_section.pack(fill=tk.X, pady=(0, 10))
        dpi_frame.grid_columnconfigure(1, weight=1)
        # DPIのラベルを左端に配置。
        ttk.Label(dpi_frame, text="DPI:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(dpi_frame, textvariable=self.state.save_dpi_var, width=10).grid(
            row=0, column=1, sticky="ew", padx=5, pady=3
        )

        # --- 3. 保存ボタンフレームの構築 ---
        save_button_frame = ttk.Frame(parent, padding="10 10 10 0")
        save_button_frame.pack(fill=tk.BOTH, expand=True)

        theme.accent_button(
            save_button_frame,
            text="画像を保存",
            command=self.event_handlers.save_figure,
        ).pack(fill=tk.X, expand=True, side=tk.BOTTOM)

    def _add_traces(self) -> None:
        """状態変数(StateManagerの変数)の変更を監視し、変更時にグラフ更新をスケジュール。"""
        trace_vars = [
            self.state.offset_correction_var,
            self.state.show_legend_var,
            self.state.marker_size_var,
            self.state.line_width_var,
            self.state.axis_label_fontsize_var,
            self.state.tick_label_fontsize_var,
            self.state.legend_fontsize_var,
            self.state.legend_location_var,
            self.state.legend_show_frame_var,
            self.state.legend_alpha_var,
            self.state.legend_columns_var,
            self.state.xlim_min_var,
            self.state.xlim_max_var,
            self.state.ylim_min_var,
            self.state.ylim_max_var,
            self.state.show_grid_var,
            self.state.show_zero_lines_var,
            self.state.zero_line_color_var,
            self.state.zero_line_linestyle_var,
            self.state.x_format_si_var,
            self.state.x_format_cgs_var,
            self.state.y_format_si_var,
            self.state.y_format_cgs_var,
            self.state.y_format_norm_var,
            self.state.unit_mode_var,
        ]
        for var in trace_vars:
            var.trace_add("write", self._schedule_update)

    def _update_file_list_ui(self) -> None:
        """
        ファイルリストのUI（描画順の変更、色の選択、削除）を再構築。
        状態の不整合を防ぐため、一度すべてのウィジェットを破棄してから、
        現在の vsm_data の状態に合わせてゼロから生成し直すアプローチ。
        """
        # 既存のウィジェット（各行のフレーム等）をすべて破棄。
        for widget in self.individual_color_frame.winfo_children():
            widget.destroy()

        # 現在保持しているファイルデータごとにUIの行（row_frame）を作成
        for i, data in enumerate(self.vsm_data):
            row_frame = ttk.Frame(self.individual_color_frame)
            row_frame.pack(fill=tk.X, pady=2)

            filename = data["path"].name
            color_var = self.file_color_vars[i]

            # --- 上移動ボタン ---
            # lambda内の idx=i は、ループ変数の遅延評価（すべてのボタンが最後のインデックスを参照してしまうバグ）を防ぐための必須の記述である。
            up_button = ttk.Button(
                row_frame,
                text="↑",
                width=3,
                command=lambda idx=i: self.event_handlers.move_file_up(idx),
            )
            up_button.pack(side=tk.LEFT, padx=(0, 2))
            # 一番上の要素の場合は上移動ボタンを無効化する。
            if i == 0:
                up_button.config(state=tk.DISABLED)

            # --- 下移動ボタン ---
            down_button = ttk.Button(
                row_frame,
                text="↓",
                width=3,
                command=lambda idx=i: self.event_handlers.move_file_down(idx),
            )
            down_button.pack(side=tk.LEFT, padx=(0, 5))
            # 一番下の要素の場合は下移動ボタンを無効化する。
            if i == len(self.vsm_data) - 1:
                down_button.config(state=tk.DISABLED)

            # --- 削除ボタン ---
            theme.danger_button(
                row_frame,
                text="✕",
                width=3,
                command=lambda idx=i: self.event_handlers.remove_file(idx),
                padx=4, pady=2,
            ).pack(side=tk.LEFT, padx=(0, 5))

            # --- ファイル名ラベル ---
            # ファイル名が長すぎる場合はUIが崩れるため、25文字で切り詰めて「..」を付与する。
            display_name = (filename[:25] + "..") if len(filename) > 27 else filename
            ttk.Label(row_frame, text=display_name).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )

            # --- 色プレビューと色選択ボタン ---
            # ttk.Label はテーマによって背景色の変更が難しいため、標準の tk.Label を用いて色付きの四角形を作る。
            preview = tk.Label(row_frame, text="", bg=color_var.get(), width=4)
            preview.pack(side=tk.RIGHT, padx=5)

            # 変数に紐づく古いトレース（監視イベント）が残っていると多重登録されメモリリークの原因になるため、事前にすべて削除する。
            for trace in color_var.trace_info():
                if trace[0] == "write":
                    color_var.trace_remove("write", trace[1])

            # 色変数が変更されたら、プレビューの背景色を更新し、グラフの再描画をスケジュールする。
            color_var.trace_add(
                "write",
                lambda *args, p=preview, cv=color_var: (
                    p.config(bg=cv.get()),
                    self._schedule_update(),
                ),
            )
            # カラーピッカーを開くボタン。
            ttk.Button(
                row_frame,
                text="色選択",
                width=6,
                command=lambda idx=i: self.event_handlers.choose_individual_color(idx),
            ).pack(side=tk.RIGHT)

    def _schedule_update(self, *args: Any) -> None:
        """
        グラフの更新処理を少し遅延させてスケジュール。
        連続した入力による負荷を軽減するための処理。
        """
        # すでに更新の予約（ジョブ）が存在する場合は、それをキャンセル。
        if self._update_job:
            self.root.after_cancel(self._update_job)
        # 新たに500ミリ秒後にグラフ更新処理を実行するよう予約し、そのジョブIDを保存す。
        self._update_job = self.root.after(500, self.graph_manager.update_graph)

    def _clear_log(self) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    def log_message(self, message: str, level: str = "info") -> None:
        """ログタブにタイムスタンプ＋色付きでメッセージを追記する。level: info / success / error"""
        tag = f"log_{level}" if level in ("info", "success", "error") else "log_info"
        labels = {"info": "INFO   ", "success": "SUCCESS", "error": "ERROR  "}
        label = labels.get(level, "INFO   ")
        ts = datetime.now().strftime("%H:%M:%S")

        self.log_text.config(state=tk.NORMAL)
        for line in message.splitlines(keepends=True):
            stripped = line.rstrip("\n")
            if stripped:
                self.log_text.insert(tk.END, f"[{ts}] ", "log_ts")
                self.log_text.insert(tk.END, f"{label}  ", tag)
                self.log_text.insert(tk.END, f"{stripped}\n", tag)
            else:
                self.log_text.insert(tk.END, "\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _update_demag_settings_ui(self) -> None:
        """反磁性補正の設定UI（各ファイルごと）を再構築します。"""
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
            frame_section, frame = theme.make_section(
                self.demag_scrollable_frame, data["path"].name
            )
            frame_section.pack(fill=tk.X, expand=True, pady=5, padx=5)
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

    def _update_thickness_settings_ui(self) -> None:
        """膜厚・面積設定のUI（各ファイルごと）を再構築します。"""
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

            if "area_var" not in data:
                data["area_var"] = tk.StringVar(value="90")  # 初期値 90 mm^2

            thickness_var = data["thickness_var"]
            area_var = data["area_var"]

            frame = ttk.Frame(self.thickness_scrollable_frame, padding=(0, 0, 10, 0))
            frame.pack(fill=tk.X, expand=True, pady=2, padx=5)
            frame.grid_columnconfigure(1, weight=1)

            display_name = (
                (data["path"].name[:25] + "..")
                if len(data["path"].name) > 27
                else data["path"].name
            )

            ttk.Label(frame, text=display_name).grid(row=0, column=0, sticky="w")

            # 膜厚入力
            ttk.Label(frame, text="膜厚:").grid(row=0, column=1, sticky="e")
            entry_t = ttk.Entry(frame, textvariable=thickness_var, width=6)
            entry_t.grid(row=0, column=2, sticky="e", padx=(2, 0))
            ttk.Label(frame, text="nm").grid(row=0, column=3, sticky="w", padx=(2, 10))

            # 面積入力
            ttk.Label(frame, text="面積:").grid(row=0, column=4, sticky="e")
            entry_a = ttk.Entry(frame, textvariable=area_var, width=6)
            entry_a.grid(row=0, column=5, sticky="e", padx=(2, 0))
            ttk.Label(frame, text="mm²").grid(row=0, column=6, sticky="w", padx=(2, 0))

            theme.danger_button(
                frame,
                text="✕",
                width=3,
                command=lambda idx=i: self.event_handlers.remove_file(idx),
                padx=4, pady=2,
            ).grid(row=0, column=7, sticky="w", padx=(10, 0))

            thickness_var.trace_add(
                "write",
                lambda *a, idx=i: self._on_thickness_change(idx),
            )
            area_var.trace_add(
                "write",
                lambda *a, idx=i: self._on_thickness_change(idx),
            )

    def _on_thickness_change(self, file_index: int) -> None:
        """
        膜厚または面積が変更されたときの処理。グラフの更新をスケジュールします。

        Args:
            file_index (int): 変更されたデータのインデックス。
        """
        if file_index >= len(self.vsm_data):
            return
        # This just triggers a graph update. The value is already in the StringVar.
        self._schedule_update()

    def _on_demag_setting_change(self, file_index: int, var_changed: Any) -> None:
        """
        反磁性補正の設定が変更されたときの処理。設定を保存しグラフを更新します。

        Args:
            file_index (int): 変更されたデータのインデックス。
            var_changed (Any): 変更をトリガーしたTkinter変数。
        """
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

    def _create_results_tab(self) -> None:
        """解析結果を表示するテーブル（Treeview）の構造を作成します。"""
        frame = ttk.Frame(self.results_tab, padding="10")
        frame.pack(expand=True, fill="both")

        # --- Treeview (Table) ---
        columns = ("filename", "ms", "mr", "hc", "sq")  # 列の定義
        # selectmode="extended" を追加して複数行選択を可能にする
        self.results_tree = ttk.Treeview(
            frame, columns=columns, show="headings", selectmode="extended"
        )

        # Define headings — command でクリックソートを有効化
        self._col_labels = {
            "filename": "ファイル名",
            "ms": "飽和磁化 Ms (kA/m)",
            "mr": "残留磁化 Mr (kA/m)",
            "hc": "保磁力 Hc (Oe)",
            "sq": "角形比 (S = Mr/Ms)",
        }
        for col, label in self._col_labels.items():
            self.results_tree.heading(
                col, text=label, command=lambda c=col: self._sort_column(c)
            )

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
        theme.setup_treeview_tags(self.results_tree)

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # --- Action Buttons ---
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        copy_text_button = ttk.Button(
            button_frame,
            text="テキストとしてコピー",
            command=self._copy_results_to_clipboard,
        )
        copy_text_button.pack(side="left", padx=5)

        copy_html_button = ttk.Button(
            button_frame,
            text="表としてコピー (PowerPoint等)",
            command=self._copy_results_as_html,
        )
        copy_html_button.pack(side="left", padx=5)

    def _update_results_table(self) -> None:
        """解析結果テーブルを最新のデータでクリアおよび再描画します。"""
        # Clear existing data
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)

        if not self.analysis_results:
            return

        # Insert new data
        for idx, res in enumerate(self.analysis_results):
            ms_str = f"{res['Ms']:.1f}" if res["Ms"] is not None else "N/A"
            mr_str = f"{res['Mr']:.1f}" if res["Mr"] is not None else "N/A"
            hc_str = f"{res['Hc_Oe']:.1f}" if res["Hc_Oe"] is not None else "N/A"
            sq_str = (
                f"{res['squareness']:.3f}" if res["squareness"] is not None else "N/A"
            )
            tag = "evenrow" if idx % 2 == 0 else "oddrow"
            self.results_tree.insert(
                "", "end", values=(tex_to_display(res["filename"]), ms_str, mr_str, hc_str, sq_str),
                tags=(tag,),
            )

    def _sort_column(self, col: str) -> None:
        """解析結果テーブルを指定列でソートし、ヘッダーに矢印を表示します。"""
        ascending = not self._sort_ascending.get(col, True)
        self._sort_ascending[col] = ascending

        rows = [
            (self.results_tree.set(child, col), child)
            for child in self.results_tree.get_children("")
        ]
        rows.sort(key=lambda x: _make_sort_key(x[0], col), reverse=not ascending)
        for i, (_, child) in enumerate(rows):
            self.results_tree.move(child, "", i)
            tag = "evenrow" if i % 2 == 0 else "oddrow"
            self.results_tree.item(child, tags=(tag,))

        arrow = " ▲" if ascending else " ▼"
        for c, label in self._col_labels.items():
            self.results_tree.heading(
                c,
                text=label + (arrow if c == col else ""),
                command=lambda c=c: self._sort_column(c),
            )

    def _copy_results_to_clipboard(self) -> None:
        """解析結果テーブルのデータをTSV形式でクリップボードにコピーします。"""
        if not self.analysis_results:
            return

        try:
            # ヘッダー行を作成
            headers = [
                "ファイル名",
                "飽和磁化 Ms (kA/m)",
                "残留磁化 Mr (kA/m)",
                "保磁力 Hc (Oe)",
                "角形比 (S = Mr/Ms)",
            ]
            tsv_data_lines = ["\t".join(headers)]

            selected_items = self.results_tree.selection()  # 選択された行のIDを取得

            if selected_items:
                # 選択された行のデータをコピー
                for item_id in selected_items:
                    values = self.results_tree.item(item_id, "values")
                    tsv_data_lines.append(
                        "\t".join(map(str, values))
                    )  # 値を文字列に変換してTSV形式で追加
                message_suffix = "選択された行をクリップボードにコピーしました。"
            else:
                # 選択された行がなければ、すべての行のデータをコピー
                for res in self.analysis_results:
                    ms_str = f"{res['Ms']:.1f}" if res["Ms"] is not None else ""
                    mr_str = f"{res['Mr']:.1f}" if res["Mr"] is not None else ""
                    hc_str = f"{res['Hc_Oe']:.1f}" if res["Hc_Oe"] is not None else ""
                    sq_str = (
                        f"{res['squareness']:.3f}"
                        if res["squareness"] is not None
                        else ""
                    )
                    row = [tex_to_display(res["filename"]), ms_str, mr_str, hc_str, sq_str]
                    tsv_data_lines.append("\t".join(row))
                message_suffix = "すべての解析結果をクリップボードにコピーしました。"

            tsv_data = "\n".join(tsv_data_lines)  # 全ての行を改行で結合

            self.root.clipboard_clear()
            self.root.clipboard_append(tsv_data)

            messagebox.showinfo("成功", message_suffix, parent=self.root)
        except Exception as e:
            messagebox.showerror(
                "エラー", f"コピー中にエラーが発生しました: {e}", parent=self.root
            )

    def _copy_results_as_html(self) -> None:
        """解析結果テーブルのデータをCF_HTML形式でクリップボードにコピーします（PowerPoint対応）。"""
        if not self.analysis_results:
            return

        try:
            import win32clipboard

            headers = [
                "ファイル名",
                "飽和磁化 Ms (kA/m)",
                "残留磁化 Mr (kA/m)",
                "保磁力 Hc (Oe)",
                "角形比 (S = Mr/Ms)",
            ]

            # HTML テーブルを構築
            table = '<table border="1" style="border-collapse:collapse; font-family: Arial, sans-serif;">'
            table += "<thead><tr style='background-color:#f2f2f2;'>"
            for h in headers:
                table += f'<th style="padding:8px; border:1px solid #ddd; text-align:left;">{h}</th>'
            table += "</tr></thead><tbody>"

            selected_items = self.results_tree.selection()
            if selected_items:
                items_to_copy = [self.results_tree.item(i, "values") for i in selected_items]
                message_suffix = "選択された行を表形式でクリップボードにコピーしました。"
            else:
                items_to_copy = [self.results_tree.item(i, "values") for i in self.results_tree.get_children()]
                message_suffix = "すべての解析結果を表形式でクリップボードにコピーしました。"

            for values in items_to_copy:
                table += "<tr>"
                for value in values:
                    try:
                        float(value)
                        align = "right"
                    except (ValueError, TypeError):
                        align = "left"
                    table += f'<td style="padding:8px; border:1px solid #ddd; text-align:{align};">{value}</td>'
                table += "</tr>"
            table += "</tbody></table>"

            # CF_HTML 形式のヘッダーを計算して付加
            frag_html = "<!--StartFragment-->" + table + "<!--EndFragment-->"
            body = (
                "<html>\r\n<head>\r\n"
                '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">\r\n'
                "</head>\r\n<body>\r\n"
                + frag_html
                + "\r\n</body>\r\n</html>"
            )
            hdr_template = (
                "Version:0.9\r\n"
                "StartHTML:{sh:08d}\r\nEndHTML:{eh:08d}\r\n"
                "StartFragment:{sf:08d}\r\nEndFragment:{ef:08d}\r\n"
            )
            placeholder = hdr_template.format(sh=0, eh=0, sf=0, ef=0)
            hdr_len = len(placeholder.encode("utf-8"))
            sf = hdr_len + len(body[: body.find("<!--StartFragment-->") + len("<!--StartFragment-->")].encode("utf-8"))
            ef = hdr_len + len(body[: body.find("<!--EndFragment-->")].encode("utf-8"))
            eh = hdr_len + len(body.encode("utf-8"))
            header = hdr_template.format(sh=hdr_len, eh=eh, sf=sf, ef=ef)
            cf_html_bytes = (header + body).encode("utf-8")

            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                cf_fmt = win32clipboard.RegisterClipboardFormat("HTML Format")
                win32clipboard.SetClipboardData(cf_fmt, cf_html_bytes)
            finally:
                win32clipboard.CloseClipboard()

            messagebox.showinfo("成功", message_suffix, parent=self.root)

        except ImportError:
            messagebox.showerror(
                "エラー",
                "pywin32がインストールされていません。\npip install pywin32 を実行してください。",
                parent=self.root,
            )
        except Exception as e:
            messagebox.showerror(
                "エラー",
                f"表形式コピー中にエラーが発生しました: {e}",
                parent=self.root,
            )
