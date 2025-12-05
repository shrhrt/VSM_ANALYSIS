# -*- coding: utf-8 -*-
import sys
import io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, colorchooser
from pathlib import Path
from contextlib import redirect_stdout
from typing import Tuple, Optional, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.stats import linregress

# -----------------------------------------------------------------------------
# 定数・設定
# -----------------------------------------------------------------------------
DEFAULT_COLORS = [
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
ENCODINGS = ["shift-jis", "utf-8"]
FONT_FAMILY = "Arial"


# -----------------------------------------------------------------------------
# 計算・解析用ヘルパー関数
# -----------------------------------------------------------------------------


def find_header_row(file_path: Path, default_row: int = 40) -> int:
    """ファイル内のデータヘッダー行('H(Oe)'と'M(emu)')を検出する"""
    for encoding in ENCODINGS:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for i, line in enumerate(f):
                    if "H(Oe)" in line and "M(emu)" in line:
                        return i
                    if i > 100:  # 探索範囲リミット
                        break
        except (UnicodeDecodeError, IOError):
            continue
    return default_row


def parse_metadata(file_path: Path) -> Dict[str, str]:
    """ファイルのヘッダーから測定条件などのメタデータを抽出する"""
    metadata = {}
    for encoding in ENCODINGS:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for i, line in enumerate(f):
                    if i > 50:  # ヘッダー領域のみ探索
                        break
                    line = line.strip()
                    if "=" in line:
                        parts = line.split("=", 1)
                        key = parts[0].strip()
                        # "key=,value,unit" 形式に対応
                        val_parts = parts[1].split(",")
                        if len(val_parts) > 1:
                            value = val_parts[1].strip()
                            if key and value:
                                metadata[key] = value
            if metadata:
                return metadata
        except (UnicodeDecodeError, IOError):
            continue
    return metadata


def calculate_slope(
    df: pd.DataFrame,
    h_col: str,
    m_col: str,
    manual_range: Optional[Tuple[float, float]] = None,
    auto_ratio: float = 0.15,
) -> Tuple[float, float]:
    """
    指定範囲または自動範囲で線形回帰を行い、傾きとR^2を返す。
    manual_range: (min_H, max_H) のタプル。Noneの場合は自動。
    """
    if manual_range:
        mask = (df[h_col] >= manual_range[0]) & (df[h_col] <= manual_range[1])
        target_df = df.loc[mask]
        if len(target_df) < 2:
            return 0.0, 0.0
    else:
        # 自動: データの端(auto_ratio割合)を使用
        n_points = int(len(df) * auto_ratio)
        if n_points < 2:
            return 0.0, 0.0
        # 正側なら末尾、負側なら先頭を想定（呼び出し元でdfを適切に渡すこと）
        target_df = df

    try:
        res = linregress(target_df[h_col], target_df[m_col])
        return res.slope, res.rvalue**2
    except ValueError:
        return 0.0, 0.0


def get_magnetic_properties(H: np.ndarray, M: np.ndarray) -> Dict[str, float]:
    """保磁力、残留磁化、飽和磁化を計算して辞書で返す"""
    props = {}

    # 前半(Down)と後半(Up)に分割
    idx_min = np.argmin(H)
    H_down, M_down = H[: idx_min + 1], M[: idx_min + 1]
    H_up, M_up = H[idx_min:], M[idx_min:]

    # 残留磁化 Mr
    try:
        mr_down = np.interp(0, H_down[::-1], M_down[::-1])
        mr_up = np.interp(0, H_up, M_up)
        props["Mr"] = (abs(mr_down) + abs(mr_up)) / 2
    except Exception:
        props["Mr"] = 0.0

    # 保磁力 Hc
    try:
        hc_down = np.interp(0, M_down[::-1], H_down[::-1])
        hc_up = np.interp(0, M_up, H_up)
        props["Hc"] = (abs(hc_down) + abs(hc_up)) / 2
    except Exception:
        props["Hc"] = 0.0

    # 飽和磁化 Ms (最大磁場の90%以上の領域の平均)
    try:
        h_max, h_min = np.max(H), np.min(H)
        ms_pos = np.mean(M[H > h_max * 0.9]) if np.any(H > h_max * 0.9) else 0
        ms_neg = np.mean(np.abs(M[H < h_min * 0.9])) if np.any(H < h_min * 0.9) else 0
        props["Ms"] = (ms_pos + ms_neg) / 2
    except Exception:
        props["Ms"] = 0.0

    return props


# -----------------------------------------------------------------------------
# GUIアプリケーション
# -----------------------------------------------------------------------------


class VSMApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VSM Data Analyzer")
        self.root.geometry("1200x850")

        # データ管理
        self.vsm_data: List[Dict[str, Any]] = []
        self.file_color_vars: List[tk.StringVar] = []
        self.all_metadata: Dict[str, Dict[str, str]] = {}
        self._update_job = None

        # スタイルの設定
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._configure_styles()
        self.root.configure(bg=self.style.lookup(".", "background"))

        # 変数の初期化
        self._init_variables()

        # UIの構築
        self._setup_layout()
        self._setup_traces()

        # 初期グラフ描画
        self.toggle_demag_fields()
        self.update_graph()

    def _init_variables(self):
        """Tkinter変数の初期化"""
        # 解析パラメータ
        self.thick_var = tk.StringVar(value="100.0")
        self.area_var = tk.StringVar(value="1.0")
        self.offset_corr_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)

        # 反磁性補正
        self.demag_corr_var = tk.BooleanVar(value=True)
        self.manual_slope_var = tk.BooleanVar(value=False)
        self.pos_h_min_var = tk.StringVar(value="1.5")
        self.pos_h_max_var = tk.StringVar(value="2.0")
        self.neg_h_min_var = tk.StringVar(value="-2.0")
        self.neg_h_max_var = tk.StringVar(value="-1.5")

        # グラフスタイル
        self.marker_size_var = tk.StringVar(value="5")
        self.line_width_var = tk.StringVar(value="1.5")
        self.axis_lbl_size_var = tk.StringVar(value="16")
        self.tick_lbl_size_var = tk.StringVar(value="12")
        self.legend_size_var = tk.StringVar(value="12")
        self.show_grid_var = tk.BooleanVar(value=True)
        self.show_zero_line_var = tk.BooleanVar(value=True)

        # 軸範囲
        self.xlim_min_var = tk.StringVar(value="")
        self.xlim_max_var = tk.StringVar(value="")
        self.ylim_min_var = tk.StringVar(value="")
        self.ylim_max_var = tk.StringVar(value="")

        # 保存設定
        self.save_w_var = tk.StringVar(value="6.0")
        self.save_h_var = tk.StringVar(value="6.0")
        self.save_dpi_var = tk.StringVar(value="300")

    def _configure_styles(self):
        """UIスタイルの定義"""
        bg = "SystemButtonFace"
        fg = "black"
        accent = "#007ACC"

        self.style.configure(".", background=bg, foreground=fg, font=(FONT_FAMILY, 10))
        self.style.configure("TLabel", background=bg, foreground=fg)
        self.style.configure("TFrame", background=bg)
        self.style.configure(
            "TLabelframe", background=bg, bordercolor="#CCCCCC", foreground=fg
        )
        self.style.configure(
            "TLabelframe.Label", foreground=accent, font=(FONT_FAMILY, 11, "bold")
        )
        self.style.configure(
            "TButton",
            background=accent,
            foreground="white",
            font=(FONT_FAMILY, 11, "bold"),
            borderwidth=0,
        )
        self.style.map("TButton", background=[("active", "#005F9E")])

    def _setup_layout(self):
        """メインレイアウトの構築"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左カラム: 設定タブ
        left_panel = ttk.Notebook(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        tab_analysis = ttk.Frame(left_panel, padding="10")
        tab_style = ttk.Frame(left_panel, padding="10")
        tab_export = ttk.Frame(left_panel, padding="10")

        left_panel.add(tab_analysis, text="解析")
        left_panel.add(tab_style, text="グラフ設定")
        left_panel.add(tab_export, text="保存")

        self._create_analysis_tab(tab_analysis)
        self._create_style_tab(tab_style)
        self._create_export_tab(tab_export)

        # 右カラム: グラフとログ
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # グラフエリア
        graph_frame = ttk.LabelFrame(right_panel, text=" グラフ ", padding=10)
        # 修正: weight引数を削除
        graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(6, 6), facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)

        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame, pack_toolbar=False)
        self._customize_toolbar(toolbar)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ログエリア
        log_frame = ttk.LabelFrame(right_panel, text=" ログ ", padding="10")
        # 修正: weight引数を削除
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 9), height=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _customize_toolbar(self, toolbar):
        """Matplotlibツールバーのスタイル調整"""
        bg = self.style.lookup(".", "background")
        toolbar.config(background=bg)
        toolbar._message_label.config(background=bg, foreground="black")
        for button in toolbar.winfo_children():
            if isinstance(button, (tk.Button, tk.Checkbutton)):
                button.config(background=bg, foreground="black")
        toolbar.update()

    def _create_analysis_tab(self, parent):
        # ファイル操作
        frame_file = ttk.LabelFrame(parent, text=" ファイル ", padding="10")
        frame_file.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(frame_file, text="ファイルを選択", command=self.load_files).pack(
            fill=tk.X
        )
        self.btn_info = ttk.Button(
            frame_file,
            text="測定情報を表示",
            command=self.show_metadata,
            state=tk.DISABLED,
        )
        self.btn_info.pack(fill=tk.X, pady=(5, 0))

        # 基本設定
        frame_basic = ttk.LabelFrame(parent, text=" 解析設定 ", padding="10")
        frame_basic.pack(fill=tk.X, pady=(0, 10))

        grid_opts = {"sticky": "ew", "padx": 5, "pady": 2}
        ttk.Label(frame_basic, text="膜厚 (nm):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_basic, textvariable=self.thick_var, width=10).grid(
            row=0, column=1, **grid_opts
        )

        ttk.Label(frame_basic, text="面積 (cm²):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame_basic, textvariable=self.area_var, width=10).grid(
            row=1, column=1, **grid_opts
        )

        ttk.Checkbutton(
            frame_basic, text="磁化オフセット補正", variable=self.offset_corr_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Checkbutton(
            frame_basic, text="凡例を表示", variable=self.show_legend_var
        ).grid(row=3, column=0, columnspan=2, sticky="w")

        # 反磁性補正
        frame_demag = ttk.LabelFrame(parent, text=" 反磁性補正 ", padding="10")
        frame_demag.pack(fill=tk.X)

        ttk.Checkbutton(
            frame_demag, text="補正を有効化", variable=self.demag_corr_var
        ).pack(anchor="w")
        self.chk_manual = ttk.Checkbutton(
            frame_demag, text="範囲を手動指定", variable=self.manual_slope_var
        )
        self.chk_manual.pack(anchor="w", pady=5)

        # 範囲入力用グリッド
        f_range = ttk.Frame(frame_demag)
        f_range.pack(fill=tk.X)

        self.entries_demag = []
        for i, (label, v_min, v_max) in enumerate(
            [
                ("正 H:", self.pos_h_min_var, self.pos_h_max_var),
                ("負 H:", self.neg_h_min_var, self.neg_h_max_var),
            ]
        ):
            ttk.Label(f_range, text=label).grid(row=i, column=0, sticky="w")
            e1 = ttk.Entry(f_range, textvariable=v_min, width=7)
            e1.grid(row=i, column=1, padx=2)
            ttk.Label(f_range, text="~").grid(row=i, column=2)
            e2 = ttk.Entry(f_range, textvariable=v_max, width=7)
            e2.grid(row=i, column=3, padx=2)
            self.entries_demag.extend([e1, e2])

    def _create_style_tab(self, parent):
        # 表示オプション
        frame_opt = ttk.LabelFrame(parent, text=" 表示オプション ", padding="10")
        frame_opt.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(frame_opt, text="グリッド", variable=self.show_grid_var).pack(
            anchor="w"
        )
        ttk.Checkbutton(
            frame_opt, text="原点線", variable=self.show_zero_line_var
        ).pack(anchor="w")

        # プロット・フォント設定
        for title, items in [
            (
                " プロット ",
                [
                    ("マーカーサイズ", self.marker_size_var),
                    ("線幅", self.line_width_var),
                ],
            ),
            (
                " フォント ",
                [
                    ("軸ラベル", self.axis_lbl_size_var),
                    ("目盛り", self.tick_lbl_size_var),
                    ("凡例", self.legend_size_var),
                ],
            ),
        ]:
            frame = ttk.LabelFrame(parent, text=title, padding="10")
            frame.pack(fill=tk.X, pady=(0, 10))
            for i, (label, var) in enumerate(items):
                ttk.Label(frame, text=f"{label}:").grid(
                    row=i, column=0, sticky="w", pady=2
                )
                ttk.Entry(frame, textvariable=var, width=10).grid(
                    row=i, column=1, sticky="ew", padx=5
                )

        # 軸範囲
        frame_axes = ttk.LabelFrame(
            parent, text=" 描画範囲 (空白で自動) ", padding="10"
        )
        frame_axes.pack(fill=tk.X, pady=(0, 10))
        for i, (label, v_min, v_max) in enumerate(
            [
                ("X軸 (T)", self.xlim_min_var, self.xlim_max_var),
                ("Y軸 (kA/m)", self.ylim_min_var, self.ylim_max_var),
            ]
        ):
            ttk.Label(frame_axes, text=label).grid(row=i, column=0, sticky="w")
            ttk.Entry(frame_axes, textvariable=v_min, width=7).grid(
                row=i, column=1, padx=2
            )
            ttk.Label(frame_axes, text="~").grid(row=i, column=2)
            ttk.Entry(frame_axes, textvariable=v_max, width=7).grid(
                row=i, column=3, padx=2
            )

        # 個別色設定コンテナ
        self.frame_colors = ttk.LabelFrame(
            parent, text=" ファイル別カラー ", padding="10"
        )
        self.frame_colors.pack(fill=tk.X, pady=(10, 0))

    def _create_export_tab(self, parent):
        frame = ttk.LabelFrame(parent, text=" 画像保存設定 ", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        for i, (label, var, unit) in enumerate(
            [
                ("幅", self.save_w_var, "inch"),
                ("高さ", self.save_h_var, "inch"),
                ("DPI", self.save_dpi_var, ""),
            ]
        ):
            ttk.Label(frame, text=f"{label}:").grid(row=i, column=0, sticky="w", pady=3)
            ttk.Entry(frame, textvariable=var, width=10).grid(row=i, column=1, padx=5)
            if unit:
                ttk.Label(frame, text=unit).grid(row=i, column=2, sticky="w")

        ttk.Button(
            parent, text="画像を保存", command=self.save_figure, padding=10
        ).pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_traces(self):
        """設定変更時にグラフを更新するトリガーを設定"""
        vars_to_trace = [
            self.thick_var,
            self.area_var,
            self.offset_corr_var,
            self.demag_corr_var,
            self.manual_slope_var,
            self.pos_h_min_var,
            self.pos_h_max_var,
            self.neg_h_min_var,
            self.neg_h_max_var,
            self.show_legend_var,
            self.marker_size_var,
            self.line_width_var,
            self.axis_lbl_size_var,
            self.tick_lbl_size_var,
            self.legend_size_var,
            self.xlim_min_var,
            self.xlim_max_var,
            self.ylim_min_var,
            self.ylim_max_var,
            self.show_grid_var,
            self.show_zero_line_var,
        ]
        for var in vars_to_trace:
            var.trace_add("write", self._schedule_update)

        # 反磁性補正のUI制御
        self.demag_corr_var.trace_add("write", lambda *args: self.toggle_demag_fields())
        self.manual_slope_var.trace_add(
            "write", lambda *args: self.toggle_demag_fields()
        )

    def _schedule_update(self, *args):
        """連続更新を防ぐための遅延実行"""
        if self._update_job:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(300, self.update_graph)

    def toggle_demag_fields(self):
        """反磁性補正の設定に応じて入力欄の有効/無効を切り替え"""
        enable_demag = self.demag_corr_var.get()
        enable_manual = enable_demag and self.manual_slope_var.get()

        state_chk = tk.NORMAL if enable_demag else tk.DISABLED
        state_entry = tk.NORMAL if enable_manual else tk.DISABLED

        self.chk_manual.config(state=state_chk)
        for entry in self.entries_demag:
            entry.config(state=state_entry)

    def log(self, msg: str):
        """ログウィンドウへの出力"""
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    # -------------------------------------------------------------------------
    # ロジック: ファイル読み込み
    # -------------------------------------------------------------------------
    def load_files(self):
        files = filedialog.askopenfilenames(
            title="VSMファイルを選択",
            filetypes=[("VSM files", "*.VSM"), ("All files", "*.*")],
        )
        if not files:
            return

        self.vsm_data.clear()
        self.file_color_vars.clear()
        self.all_metadata.clear()

        # カラー設定エリアのクリア
        for widget in self.frame_colors.winfo_children():
            widget.destroy()

        for i, file_path in enumerate(files):
            path = Path(file_path)
            try:
                header_row = find_header_row(path)
                # 読み込み試行
                try:
                    df = pd.read_csv(path, header=header_row, encoding="shift-jis")
                except UnicodeDecodeError:
                    df = pd.read_csv(path, header=header_row, encoding="utf-8")

                df.dropna(inplace=True)

                if not {"H(Oe)", "M(emu)"}.issubset(df.columns):
                    messagebox.showwarning(
                        "形式エラー",
                        f"{path.name}: 必要な列(H(Oe), M(emu))がありません。",
                    )
                    continue

                self.vsm_data.append({"path": path, "df": df})

                # 色設定UIの追加
                color_var = tk.StringVar(value=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
                self.file_color_vars.append(color_var)
                self._add_color_widget(i, path.name, color_var)

            except Exception as e:
                messagebox.showerror(
                    "エラー", f"{path.name} の読み込みに失敗しました:\n{e}"
                )

        self.btn_info.config(state=tk.NORMAL if self.vsm_data else tk.DISABLED)
        self.update_graph()

    def _add_color_widget(self, idx, name, var):
        f = ttk.Frame(self.frame_colors)
        f.pack(fill=tk.X, pady=2)

        short_name = (name[:20] + "..") if len(name) > 22 else name
        ttk.Label(f, text=short_name).pack(side=tk.LEFT)

        lbl_preview = tk.Label(f, bg=var.get(), width=4)
        lbl_preview.pack(side=tk.RIGHT, padx=5)

        def pick_color(*args):
            c = colorchooser.askcolor(initialcolor=var.get(), title=f"{name} の色")
            if c[1]:
                var.set(c[1])
                lbl_preview.config(bg=c[1])
                self._schedule_update()

        ttk.Button(f, text="色変更", width=6, command=pick_color).pack(side=tk.RIGHT)

    def show_metadata(self):
        """メタデータ表示ウィンドウ"""
        if not self.all_metadata:
            # まだ解析が一度も走っていない場合はここでパースする
            for d in self.vsm_data:
                self.all_metadata[d["path"].name] = parse_metadata(d["path"])

        if not self.all_metadata:
            messagebox.showinfo("情報", "表示できる情報がありません")
            return

        win = tk.Toplevel(self.root)
        win.title("測定情報")
        win.geometry("500x600")

        # ファイル選択コンボ
        names = list(self.all_metadata.keys())
        cb_var = tk.StringVar(value=names[0])
        cb = ttk.Combobox(win, textvariable=cb_var, values=names, state="readonly")
        cb.pack(fill=tk.X, padx=10, pady=10)

        txt = scrolledtext.ScrolledText(win, font=("Arial", 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def update_txt(e=None):
            name = cb_var.get()
            data = self.all_metadata.get(name, {})
            txt.config(state=tk.NORMAL)
            txt.delete(1.0, tk.END)
            content = f"ファイル: {name}\n\n" + "\n".join(
                [f"■ {k}\n  {v}\n" for k, v in data.items()]
            )
            txt.insert(tk.END, content)
            txt.config(state=tk.DISABLED)

        cb.bind("<<ComboboxSelected>>", update_txt)
        update_txt()

    # -------------------------------------------------------------------------
    # ロジック: グラフ更新と解析実行
    # -------------------------------------------------------------------------
    def update_graph(self):
        self.log_text.delete(1.0, tk.END)
        self.ax.clear()

        # パラメータ取得
        try:
            thick = float(self.thick_var.get())
            area = float(self.area_var.get())
            vol = thick * area * 1e-7  # cm^2 * nm -> 単位換算
        except ValueError:
            self.log("エラー: 膜厚または面積に数値以外が含まれています。\n")
            return

        if not self.vsm_data:
            self.ax.text(
                0.5, 0.5, "ファイルを選択してください", ha="center", color="gray"
            )
            self.canvas.draw()
            return

        # 標準出力のキャプチャ（log_textへ流す）
        f_out = io.StringIO()
        with redirect_stdout(f_out):
            print(f"--- 解析開始 (膜厚: {thick} nm, 面積: {area} cm^2) ---\n")

            x_min_global, x_max_global = float("inf"), float("-inf")

            for i, data in enumerate(self.vsm_data):
                path = data["path"]
                df = data["df"].copy()

                # データ準備 (Oe -> T, emu -> kA/m)
                H_raw = df["H(Oe)"].values * 1e-4
                M_raw = df["M(emu)"].values / vol

                # 一時的なDataFrameで処理
                df_proc = (
                    pd.DataFrame({"H": H_raw, "M": M_raw})
                    .sort_values(by="H")
                    .reset_index(drop=True)
                )

                print(f"[{i + 1}] {path.name}")

                # 1. 反磁性補正
                slope = 0.0
                if self.demag_corr_var.get():
                    manual_range = None
                    if self.manual_slope_var.get():
                        try:
                            p_min = float(self.pos_h_min_var.get())
                            p_max = float(self.pos_h_max_var.get())
                            # 負側も考慮する場合はロジック追加が必要だが、元のコードに準拠
                            # 元コードはfind_demag_slope_manualで正負両方見ていたので合わせる
                            n_min = float(self.neg_h_min_var.get())
                            n_max = float(self.neg_h_max_var.get())

                            # 正側計算
                            s_pos, r2_pos = calculate_slope(
                                df_proc, "H", "M", (p_min, p_max)
                            )
                            # 負側計算
                            s_neg, r2_neg = calculate_slope(
                                df_proc, "H", "M", (n_min, n_max)
                            )

                            slope = (s_pos + s_neg) / 2
                            print(
                                f"  手動補正: 傾き={slope:.6f} (R2: Pos={r2_pos:.4f}, Neg={r2_neg:.4f})"
                            )
                        except ValueError:
                            print("  エラー: 範囲指定の値が不正です")
                    else:
                        # 自動
                        df_pos = df_proc.tail(int(len(df_proc) * 0.15))
                        df_neg = df_proc.head(int(len(df_proc) * 0.15))
                        s_pos, r2_pos = calculate_slope(df_pos, "H", "M")
                        s_neg, r2_neg = calculate_slope(df_neg, "H", "M")
                        slope = (s_pos + s_neg) / 2
                        print(f"  自動補正: 傾き={slope:.6f}")

                M_corr = M_raw - (H_raw * slope)

                # 2. オフセット補正
                offset = 0.0
                if self.offset_corr_var.get():
                    props = get_magnetic_properties(H_raw, M_corr)
                    # Ms計算時の中心ズレをオフセットとする簡易手法
                    h_max = np.max(H_raw)
                    ms_pos = (
                        np.mean(M_corr[H_raw > h_max * 0.9])
                        if np.any(H_raw > h_max * 0.9)
                        else 0
                    )
                    ms_neg = (
                        np.mean(M_corr[H_raw < -h_max * 0.9])
                        if np.any(H_raw < -h_max * 0.9)
                        else 0
                    )
                    offset = (ms_pos + ms_neg) / 2
                    print(f"  オフセット補正: {offset:.4f} kA/m")

                M_final = M_corr - offset

                # 3. 磁気特性の計算と表示
                # 往路復路の分割（再計算用）
                idx_split = np.argmin(H_raw)  # 単純な最小値分割
                H_down, M_down = H_raw[: idx_split + 1], M_final[: idx_split + 1]
                H_up, M_up = H_raw[idx_split:], M_final[idx_split:]

                props = get_magnetic_properties(H_raw, M_final)
                print(f"  Ms: {props['Ms']:.2f} kA/m")
                print(f"  Mr: {props['Mr']:.2f} kA/m")
                print(f"  Hc: {props['Hc'] * 10000:.2f} Oe ({props['Hc']:.4f} T)\n")

                # 4. プロット
                color = self.file_color_vars[i].get()
                ms = float(self.marker_size_var.get())
                lw = float(self.line_width_var.get())

                self.ax.plot(
                    H_down,
                    M_down,
                    color=color,
                    marker="o",
                    markersize=ms,
                    linestyle="-",
                    linewidth=lw,
                )
                self.ax.plot(
                    H_up,
                    M_up,
                    color=color,
                    marker="o",
                    markersize=ms,
                    linestyle="-",
                    linewidth=lw,
                    label=path.stem,
                )

                x_min_global = min(x_min_global, H_raw.min())
                x_max_global = max(x_max_global, H_raw.max())

                # メタデータ保存
                if path.name not in self.all_metadata:
                    self.all_metadata[path.name] = parse_metadata(path)

            # 軸設定の適用
            self._apply_axis_settings(x_min_global, x_max_global)
            self.fig.tight_layout()

        self.log(f_out.getvalue())
        self.canvas.draw()

    def _apply_axis_settings(self, auto_min, auto_max):
        """軸ラベル、範囲、グリッド等の適用"""
        fs_lbl = int(self.axis_lbl_size_var.get())
        fs_tick = int(self.tick_lbl_size_var.get())

        self.ax.set_xlabel(r"$\mu_0H$ [T]", fontsize=fs_lbl)
        self.ax.set_ylabel("M [kA/m]", fontsize=fs_lbl)
        self.ax.tick_params(
            axis="both", labelsize=fs_tick, direction="in", top=True, right=True
        )

        # 範囲
        def get_lim(var):
            return float(var.get()) if var.get() else None

        self.ax.set_xlim(
            get_lim(self.xlim_min_var) or auto_min,
            get_lim(self.xlim_max_var) or auto_max,
        )
        if get_lim(self.ylim_min_var) is not None:
            self.ax.set_ylim(get_lim(self.ylim_min_var), get_lim(self.ylim_max_var))

        # グリッド・ゼロ線
        if self.show_grid_var.get():
            self.ax.grid(True, linestyle=":", color="#CCCCCC")
        if self.show_zero_line_var.get():
            self.ax.axhline(0, color="#AAAAAA", lw=1)
            self.ax.axvline(0, color="#AAAAAA", lw=1)

        # 凡例
        if self.show_legend_var.get() and self.vsm_data:
            self.ax.legend(
                fontsize=int(self.legend_size_var.get()),
                frameon=True,
                edgecolor="#DDDDDD",
            )

    def save_figure(self):
        try:
            w = float(self.save_w_var.get())
            h = float(self.save_h_var.get())
            dpi = int(self.save_dpi_var.get())
            if w <= 0 or h <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("エラー", "幅・高さ・DPIは正の数値を入力してください")
            return

        path = filedialog.asksaveasfilename(
            title="保存",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All", "*.*")],
            defaultextension=".png",
        )
        if not path:
            return

        orig_size = self.fig.get_size_inches()
        try:
            self.fig.set_size_inches(w, h)
            self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
            self.log(f"画像を保存しました: {path}\n")
            messagebox.showinfo("完了", "保存しました")
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))
        finally:
            self.fig.set_size_inches(orig_size)
            self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = VSMApp(root)
    root.mainloop()
