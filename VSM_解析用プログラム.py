# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, colorchooser
from scipy.stats import linregress
import sys
from contextlib import redirect_stdout
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# -----------------------------------------------------------------------------
# 解析・処理関数
# -----------------------------------------------------------------------------


def find_header_row(file_path, default_row=40):
    """
    ファイル内のデータヘッダー行を自動的に検出する。
    'H(Oe)'と'M(emu)'を含む行を探し、その行番号（0-indexed）を返す。
    """
    encodings_to_try = ["shift-jis", "utf-8"]
    for encoding in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for i, line in enumerate(f):
                    if "H(Oe)" in line and "M(emu)" in line:
                        print(f"  情報: ヘッダーを {i + 1} 行目で検出。")
                        return i
                    if i > 100:
                        break
        except (UnicodeDecodeError, IOError):
            continue

    print(f"  警告: ヘッダー行を自動検出できず。デフォルト値({default_row + 1}行目)を使用。")
    return default_row


def parse_metadata(file_path):
    """
    ファイルのヘッダーから測定メタデータを抽出し、辞書として返す。
    """
    metadata = {}
    try:
        encodings_to_try = ["shift-jis", "utf-8"]
        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    for i, line in enumerate(f):
                        if i > 40: break
                        try:
                            line = line.strip()
                            if "=" in line:
                                parts = line.split("=", 1)
                                key = parts[0].strip()
                                value_part = parts[1]
                                if value_part.startswith(","):
                                    value_parts = value_part.split(",")
                                    if len(value_parts) > 1:
                                        value = value_parts[1].strip()
                                        if key and value: metadata[key] = value
                        except IndexError:
                            continue
                if metadata: return metadata
            except (UnicodeDecodeError, IOError):
                continue
    except Exception as e:
        print(f"  警告: メタデータ読み取り中に予期せぬエラー発生: {e}。")
    return metadata


def find_diamag_slope_auto(H_data, M_data, segment_ratio=0.15, min_r_squared=0.99):
    """M-Hカーブ両端から線形性の高い領域を自動検出し、反磁性補正の傾きを計算"""
    df = pd.DataFrame({"H": H_data, "M": M_data}).sort_values(by="H").reset_index(drop=True)
    n_points = len(df)
    segment_size = max(5, int(n_points * segment_ratio))

    if n_points < segment_size * 2:
        print("  警告: データ点数不足のため、傾きの自動検出をスキップ。")
        return 0, 0, 0

    slope_pos, r2_pos = 0, 0
    try:
        res_pos = linregress(df.tail(segment_size)["H"], df.tail(segment_size)["M"])
        slope_pos, r2_pos = res_pos.slope, res_pos.rvalue**2
        if r2_pos < min_r_squared: print(f"  警告: 正磁場側の線形性が低い (R^2 = {r2_pos:.4f})。")
    except (ValueError, np.linalg.LinAlgError):
        print("  警告: 正磁場側のフィッティング失敗。")

    slope_neg, r2_neg = 0, 0
    try:
        res_neg = linregress(df.head(segment_size)["H"], df.head(segment_size)["M"])
        slope_neg, r2_neg = res_neg.slope, res_neg.rvalue**2
        if r2_neg < min_r_squared: print(f"  警告: 負磁場側の線形性が低い (R^2 = {r2_neg:.4f})。")
    except (ValueError, np.linalg.LinAlgError):
        print("  警告: 負磁場側のフィッティング失敗。")

    return (slope_pos + slope_neg) / 2, r2_pos, r2_neg


def find_diamag_slope_manual(H_data, M_data, pos_range, neg_range):
    """指定磁場範囲で線形フィッティングを行い、傾きを計算"""
    df = pd.DataFrame({"H": H_data, "M": M_data})
    slope_pos, r2_pos = 0, 0
    try:
        pos_mask = (df["H"] >= pos_range[0]) & (df["H"] <= pos_range[1])
        num_pos = pos_mask.sum()
        print(f"  手動(正): H=[{pos_range[0]:.2f}, {pos_range[1]:.2f}] T, 点数: {num_pos}")
        if num_pos >= 2:
            if num_pos < 5: print(f"  警告: 正磁場範囲のデータ点数僅少({num_pos}点)。")
            res_pos = linregress(df.loc[pos_mask, "H"], df.loc[pos_mask, "M"])
            slope_pos, r2_pos = res_pos.slope, res_pos.rvalue**2
        else:
            print("  エラー: 正磁場範囲のデータ点が2点未満。")
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  警告: 正磁場範囲のフィッティング失敗: {e}")

    slope_neg, r2_neg = 0, 0
    try:
        neg_mask = (df["H"] >= neg_range[0]) & (df["H"] <= neg_range[1])
        num_neg = neg_mask.sum()
        print(f"  手動(負): H=[{neg_range[0]:.2f}, {neg_range[1]:.2f}] T, 点数: {num_neg}")
        if num_neg >= 2:
            if num_neg < 5: print(f"  警告: 負磁場範囲のデータ点数僅少({num_neg}点)。")
            res_neg = linregress(df.loc[neg_mask, "H"], df.loc[neg_mask, "M"])
            slope_neg, r2_neg = res_neg.slope, res_neg.rvalue**2
        else:
            print("  エラー: 負磁場範囲のデータ点が2点未満。")
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  警告: 負磁場範囲のフィッティング失敗: {e}")

    if slope_pos != 0 and slope_neg != 0: return (slope_pos + slope_neg) / 2, r2_pos, r2_neg
    elif slope_pos != 0: print("  情報: 正磁場側の傾きのみ使用。"); return slope_pos, r2_pos, r2_neg
    elif slope_neg != 0: print("  情報: 負磁場側の傾きのみ使用。"); return slope_neg, r2_pos, r2_neg
    else: return 0, 0, 0

def calculate_remanence(H_down, M_down, H_up, M_up):
    """残留磁化(Mr)を往路・復路データから計算"""
    try:
        Mr_down, Mr_up = np.interp(0, H_down[::-1], M_down[::-1]), np.interp(0, H_up, M_up)
        Mr_avg = (abs(Mr_down) + abs(Mr_up)) / 2
        print(f"  残留磁化 Mr: {Mr_avg:.3f} kA/m")
        return Mr_avg
    except Exception as e:
        print(f"  エラー: 残留磁化の計算失敗: {e}"); return None

def calculate_coercivity(H_down, M_down, H_up, M_up):
    """保磁力(Hc)を計算"""
    try:
        Hc_down, Hc_up = np.interp(0, M_down[::-1], H_down[::-1]), np.interp(0, M_up, H_up)
        Hc_avg = (abs(Hc_down) + abs(Hc_up)) / 2
        print(f"  保磁力 Hc: {Hc_avg * 10000:.2f} Oe ({Hc_avg:.4f} T)")
        return Hc_avg
    except Exception as e:
        print(f"  エラー: 保磁力の計算失敗: {e}"); return None

def calculate_saturation_magnetization(H, M):
    """飽和磁化(Ms)を計算"""
    H, M = np.array(H), np.array(M)
    H_max, H_min = np.max(H), np.min(H)
    Ms_pos = np.mean(M[H > H_max * 0.9]) if np.any(H > H_max * 0.9) else 0
    Ms_neg = np.mean(np.abs(M[H < H_min * 0.9])) if np.any(H < H_min * 0.9) else 0
    Ms_avg = (Ms_pos + Ms_neg) / 2
    print(f"  飽和磁化 Ms: {Ms_avg:.3f} kA/m (正側: {Ms_pos:.3f}, 負側: {Ms_neg:.3f})")
    return Ms_avg

def format_axis(ax, fig, style_params):
    """グラフの軸や目盛りなどを整形"""
    ax.set_xlabel(r"$\mu_0H$ [T]", fontsize=style_params.get("axis_label_fontsize", 16))
    ax.set_ylabel(r"M [kA/m]", fontsize=style_params.get("axis_label_fontsize", 16))
    if style_params.get("show_zero_lines", True):
        ax.axhline(0, color="#AAAAAA", linestyle="-", linewidth=1.0)
        ax.axvline(0, color="#AAAAAA", linestyle="-", linewidth=1.0)
    if style_params.get("show_grid", True): ax.grid(True, linestyle=":", color="#CCCCCC")
    if style_params.get("xlim_min") is not None and style_params.get("xlim_max") is not None:
        ax.set_xlim(style_params["xlim_min"], style_params["xlim_max"])
    if style_params.get("ylim_min") is not None and style_params.get("ylim_max") is not None:
        ax.set_ylim(style_params["ylim_min"], style_params["ylim_max"])
    ax.xaxis.set_minor_locator(AutoMinorLocator(5)); ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", which="major", labelsize=style_params.get("tick_label_fontsize", 12), direction="in", top=True, right=True, length=6, width=1.0, colors="black")
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3, width=0.8, colors="black")
    for spine in ax.spines.values(): spine.set_color("black")
    ax.xaxis.label.set_color("black"); ax.yaxis.label.set_color("black"); ax.title.set_color("black")
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["font.sans-serif"] = ["Arial"]


# -----------------------------------------------------------------------------
# GUIアプリケーションのクラス
# -----------------------------------------------------------------------------
class VSMApp:
    def __init__(self, root):
        self.vsm_data = []
        self._update_job = None
        self.all_metadata = {}
        self.file_color_vars = []
        self.base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        self.root = root
        self.root.title("VSM Data Analyzer")
        self.root.geometry("1200x850")

        # --- スタイル変数 ---
        self.thick_var = tk.StringVar(value="100.0")
        self.area_var = tk.StringVar(value="1.0")
        self.offset_correction_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)
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

        # 1. メインの左右分割
        main_paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- 左パネル ---
        left_pane = ttk.Frame(main_paned_window, padding=0)
        main_paned_window.add(left_pane, weight=1)

        # 2. 左パネルの上下分割
        left_paned_window = ttk.PanedWindow(left_pane, orient=tk.VERTICAL)
        left_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- 右パネル (グラフ用) ---
        graph_frame = ttk.LabelFrame(main_paned_window, text=" グラフ ", padding=10)
        main_paned_window.add(graph_frame, weight=2)
        graph_frame.grid_rowconfigure(1, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)

        # --- 左パネル上部 (タブUI用) ---
        notebook_frame = ttk.Frame(left_paned_window, padding=0)
        left_paned_window.add(notebook_frame, weight=1)

        notebook = ttk.Notebook(notebook_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        tab_analysis = ttk.Frame(notebook, padding="10")
        tab_diamag = ttk.Frame(notebook, padding="10")
        tab_style = ttk.Frame(notebook, padding="10")
        tab_export = ttk.Frame(notebook, padding="10")
        notebook.add(tab_analysis, text="解析")
        notebook.add(tab_diamag, text="反磁性補正")
        notebook.add(tab_style, text="グラフ設定")
        notebook.add(tab_export, text="保存")
        
        # --- 左パネル下部 (ログ用) ---
        log_outer_frame = ttk.Frame(left_paned_window, padding=(0, 10, 0, 0)) # 上にスペース
        left_paned_window.add(log_outer_frame, weight=1)
        
        log_frame = ttk.LabelFrame(log_outer_frame, text=" ログ ", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 9), bg="white", fg="black")
        self.log_text.pack(fill=tk.BOTH, expand=True)


        # --- 各タブのコントロールを作成 ---
        self._create_analysis_controls(tab_analysis)
        self._create_diamag_controls(tab_diamag)
        self._create_style_controls(tab_style)
        self._create_export_controls(tab_export)

        # --- グラフ埋め込み ---
        self.fig = plt.figure(figsize=(9, 9), facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame, pack_toolbar=False)
        toolbar.config(background=self.style.lookup(".", "background"))
        toolbar._message_label.config(background=self.style.lookup(".", "background"), foreground="black")
        for button in toolbar.winfo_children():
            if isinstance(button, (tk.Button, tk.Checkbutton)):
                button.config(background=self.style.lookup(".", "background"), foreground="black", highlightbackground=self.style.lookup(".", "background"))
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
        self.style.configure("TNotebook.Tab", background=bg, foreground="#666666", padding=[10, 5], font=("Arial", 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", accent)], foreground=[("selected", "white")])
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=fg, insertcolor=fg)
        self.style.configure("TLabelframe", background=bg, bordercolor=border, foreground=fg)
        self.style.configure("TLabelframe.Label", background=bg, foreground=accent, font=("Arial", 11, "bold"))
        self.style.configure("TButton", background=accent, foreground="white", font=("Arial", 11, "bold"), borderwidth=0)
        self.style.map("TButton", background=[("active", "#005F9E")])
        self.style.configure("TCheckbutton", background=bg, foreground=fg)

    def _create_analysis_controls(self, parent):
        file_frame = ttk.LabelFrame(parent, text=" ファイル ", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(file_frame, text="ファイルを選択", command=self.load_files, padding="10 5").pack(fill=tk.X)
        self.info_button = ttk.Button(file_frame, text="測定情報を表示", command=self.show_metadata_window, state=tk.DISABLED)
        self.info_button.pack(fill=tk.X, pady=(5, 0))

        settings_frame = ttk.LabelFrame(parent, text=" 解析設定 ", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        settings_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(settings_frame, text="膜厚 (nm):").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.thick_var, width=10).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(settings_frame, text="基板面積 (cm²):").grid(row=1, column=0, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.area_var, width=10).grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 0))
        ttk.Checkbutton(settings_frame, text="磁化オフセット補正", variable=self.offset_correction_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Checkbutton(settings_frame, text="凡例を表示", variable=self.show_legend_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))
    
    def _create_diamag_controls(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        
        file_select_frame = ttk.LabelFrame(parent, text=" 対象ファイル ", padding="10")
        file_select_frame.pack(fill=tk.X, pady=(0, 10))
        self.diamag_file_combo = ttk.Combobox(file_select_frame, state="readonly")
        self.diamag_file_combo.pack(fill=tk.X, expand=True)
        self.diamag_file_combo.bind("<<ComboboxSelected>>", self.on_diamag_file_selected)

        settings_frame = ttk.LabelFrame(parent, text=" 補正設定 ", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        settings_frame.grid_columnconfigure(1, weight=1)
        
        self.diamag_check = ttk.Checkbutton(settings_frame, text="反磁性補正を有効化")
        self.diamag_check.grid(row=0, column=0, columnspan=4, sticky="w")
        self.diamag_manual_check = ttk.Checkbutton(settings_frame, text="傾き計算の範囲を手動指定")
        self.diamag_manual_check.grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 5))

        ttk.Label(settings_frame, text="正 H (T):").grid(row=2, column=0, sticky="w")
        self.diamag_pos_min_entry = ttk.Entry(settings_frame, width=7)
        self.diamag_pos_min_entry.grid(row=2, column=1, sticky="ew")
        ttk.Label(settings_frame, text="～").grid(row=2, column=2)
        self.diamag_pos_max_entry = ttk.Entry(settings_frame, width=7)
        self.diamag_pos_max_entry.grid(row=2, column=3, sticky="ew")

        ttk.Label(settings_frame, text="負 H (T):").grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.diamag_neg_min_entry = ttk.Entry(settings_frame, width=7)
        self.diamag_neg_min_entry.grid(row=3, column=1, sticky="ew", pady=(5, 0))
        ttk.Label(settings_frame, text="～").grid(row=3, column=2, pady=(5, 0))
        self.diamag_neg_max_entry = ttk.Entry(settings_frame, width=7)
        self.diamag_neg_max_entry.grid(row=3, column=3, sticky="ew", pady=(5, 0))
        
        self.toggle_diamag_fields(is_init_or_empty=True) # 初期状態を無効化
        
    def _create_style_controls(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        axis_grid_frame = ttk.LabelFrame(parent, text=" 軸とグリッド ", padding="10")
        axis_grid_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(axis_grid_frame, text="グリッド線を表示", variable=self.show_grid_var).pack(anchor="w")
        ttk.Checkbutton(axis_grid_frame, text="原点線を表示", variable=self.show_zero_lines_var).pack(anchor="w", pady=(5, 0))
        plot_frame = ttk.LabelFrame(parent, text=" プロット ", padding="10")
        plot_frame.pack(fill=tk.X, pady=(0, 10))
        plot_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(plot_frame, text="マーカーサイズ:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Entry(plot_frame, textvariable=self.marker_size_var, width=10).grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 5))
        ttk.Label(plot_frame, text="線幅:").grid(row=1, column=0, sticky="w", pady=(0, 5))
        ttk.Entry(plot_frame, textvariable=self.line_width_var, width=10).grid(row=1, column=1, sticky="ew", padx=5, pady=(0, 5))
        font_frame = ttk.LabelFrame(parent, text=" フォントサイズ ", padding="10")
        font_frame.pack(fill=tk.X, pady=(0, 10))
        font_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(font_frame, text="軸ラベル:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Entry(font_frame, textvariable=self.axis_label_fontsize_var, width=10).grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 5))
        ttk.Label(font_frame, text="目盛り:").grid(row=1, column=0, sticky="w", pady=(0, 5))
        ttk.Entry(font_frame, textvariable=self.tick_label_fontsize_var, width=10).grid(row=1, column=1, sticky="ew", padx=5, pady=(0, 5))
        ttk.Label(font_frame, text="凡例:").grid(row=2, column=0, sticky="w", pady=(0, 5))
        ttk.Entry(font_frame, textvariable=self.legend_fontsize_var, width=10).grid(row=2, column=1, sticky="ew", padx=5, pady=(0, 5))
        self.individual_color_frame = ttk.LabelFrame(parent, text=" 各ファイルの色 ", padding="10")
        self.individual_color_frame.pack(fill=tk.X, pady=(10, 0))
        self.individual_color_frame.grid_columnconfigure(1, weight=1)
        axes_frame = ttk.LabelFrame(parent, text=" 描画範囲 ", padding="10")
        axes_frame.pack(fill=tk.X, pady=(0, 10))
        axes_frame.grid_columnconfigure(1, weight=1); axes_frame.grid_columnconfigure(3, weight=1)
        ttk.Label(axes_frame, text="X軸 (T):").grid(row=0, column=0, sticky="w")
        ttk.Entry(axes_frame, textvariable=self.xlim_min_var, width=7).grid(row=0, column=1, sticky="ew")
        ttk.Label(axes_frame, text="～").grid(row=0, column=2)
        ttk.Entry(axes_frame, textvariable=self.xlim_max_var, width=7).grid(row=0, column=3, sticky="ew")
        ttk.Label(axes_frame, text="Y軸 (kA/m):").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(axes_frame, textvariable=self.ylim_min_var, width=7).grid(row=1, column=1, sticky="ew", pady=(5, 0))
        ttk.Label(axes_frame, text="～").grid(row=1, column=2, pady=(5, 0))
        ttk.Entry(axes_frame, textvariable=self.ylim_max_var, width=7).grid(row=1, column=3, sticky="ew", pady=(5, 0))
    
    def _create_export_controls(self, parent):
        save_settings_frame = ttk.LabelFrame(parent, text=" 画像サイズ設定 ", padding="10")
        save_settings_frame.pack(fill=tk.X, pady=(0, 10)); save_settings_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(save_settings_frame, text="幅:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(save_settings_frame, textvariable=self.save_width_var, width=10).grid(row=0, column=1, sticky="ew", padx=5, pady=3)
        ttk.Label(save_settings_frame, text="inch").grid(row=0, column=2, sticky="w")
        ttk.Label(save_settings_frame, text="高さ:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(save_settings_frame, textvariable=self.save_height_var, width=10).grid(row=1, column=1, sticky="ew", padx=5, pady=3)
        ttk.Label(save_settings_frame, text="inch").grid(row=1, column=2, sticky="w")
        dpi_frame = ttk.LabelFrame(parent, text=" 解像度設定 ", padding="10")
        dpi_frame.pack(fill=tk.X, pady=(0, 10)); dpi_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(dpi_frame, text="DPI:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(dpi_frame, textvariable=self.save_dpi_var, width=10).grid(row=0, column=1, sticky="ew", padx=5, pady=3)
        save_button_frame = ttk.Frame(parent, padding="10 10 10 0")
        save_button_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(save_button_frame, text="画像を保存 (Save Image)", command=self.save_figure, padding="10").pack(fill=tk.X, expand=True, side=tk.BOTTOM)

    def _add_traces(self):
        trace_vars = [self.thick_var, self.area_var, self.offset_correction_var, self.show_legend_var, self.marker_size_var,
                      self.line_width_var, self.axis_label_fontsize_var, self.tick_label_fontsize_var, self.legend_fontsize_var,
                      self.xlim_min_var, self.xlim_max_var, self.ylim_min_var, self.ylim_max_var, self.show_grid_var, self.show_zero_lines_var]
        for var in trace_vars: var.trace_add("write", self._schedule_update)

    def on_diamag_file_selected(self, event=None):
        idx = self.diamag_file_combo.current()
        if idx < 0:
            self.toggle_diamag_fields(is_init_or_empty=True)
            return

        settings = self.vsm_data[idx]["settings"]
        self.diamag_check.config(variable=settings["correction_var"])
        self.diamag_manual_check.config(variable=settings["manual_var"])
        self.diamag_pos_min_entry.config(textvariable=settings["pos_h_min_var"])
        self.diamag_pos_max_entry.config(textvariable=settings["pos_h_max_var"])
        self.diamag_neg_min_entry.config(textvariable=settings["neg_h_min_var"])
        self.diamag_neg_max_entry.config(textvariable=settings["neg_h_max_var"])
        self.toggle_diamag_fields()

    def toggle_diamag_fields(self, is_init_or_empty=False):
        if is_init_or_empty:
            self.diamag_file_combo.config(state=tk.DISABLED)
            for widget in [self.diamag_check, self.diamag_manual_check, self.diamag_pos_min_entry, self.diamag_pos_max_entry, self.diamag_neg_min_entry, self.diamag_neg_max_entry]:
                widget.config(state=tk.DISABLED)
            return
        
        self.diamag_file_combo.config(state="readonly")
        self.diamag_check.config(state=tk.NORMAL)
        
        idx = self.diamag_file_combo.current()
        if idx < 0: return # Should not happen if not is_init_or_empty
        
        settings = self.vsm_data[idx]["settings"]
        correction_on = settings["correction_var"].get()
        self.diamag_manual_check.config(state=tk.NORMAL if correction_on else tk.DISABLED)
        
        manual_on = settings["manual_var"].get()
        manual_entry_state = tk.NORMAL if correction_on and manual_on else tk.DISABLED
        for widget in [self.diamag_pos_min_entry, self.diamag_pos_max_entry, self.diamag_neg_min_entry, self.diamag_neg_max_entry]:
            widget.config(state=manual_entry_state)

    def _reset_individual_color_widgets(self):
        if hasattr(self, "individual_color_frame"):
            for widget in self.individual_color_frame.winfo_children(): widget.destroy()

    def _create_individual_color_widget(self, index, filename, color_var):
        row_frame = ttk.Frame(self.individual_color_frame)
        row_frame.pack(fill=tk.X, pady=2)
        display_name = (filename[:25] + "..") if len(filename) > 27 else filename
        ttk.Label(row_frame, text=f"{display_name}:", anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        preview = tk.Label(row_frame, text="", bg=color_var.get(), width=4)
        preview.pack(side=tk.RIGHT, padx=5)
        color_var.trace_add("write", lambda *args: (preview.config(bg=color_var.get()), self._schedule_update()))
        ttk.Button(row_frame, text="選択", width=5, command=lambda idx=index: self.choose_individual_color(idx)).pack(side=tk.RIGHT)

    def choose_individual_color(self, index):
        if index >= len(self.file_color_vars): return
        color_var = self.file_color_vars[index]
        path_name = self.vsm_data[index]["path"].name
        title_name = (path_name[:40] + "..") if len(path_name) > 42 else path_name
        color_code = colorchooser.askcolor(title=f"'{title_name}' の色を選択", initialcolor=color_var.get())
        if color_code and color_code[1]: color_var.set(color_code[1])

    def _schedule_update(self, *args):
        if self._update_job: self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(250, self.update_graph)

    def log_message(self, message):
        self.log_text.insert(tk.END, message); self.log_text.see(tk.END); self.root.update_idletasks()

    def show_metadata_window(self):
        if not self.all_metadata:
            messagebox.showinfo("情報", "表示できる測定情報がありません。")
            return
        info_window = tk.Toplevel(self.root)
        info_window.title("測定情報"); info_window.geometry("500x650")
        info_window.configure(bg=self.style.lookup(".", "background"))
        top_frame = ttk.Frame(info_window, padding="10 10 10 0"); top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="ファイルを選択:").pack(side=tk.LEFT, padx=(0, 10))
        file_names = list(self.all_metadata.keys())
        selected_file = tk.StringVar(value=file_names[0])
        file_menu = ttk.Combobox(top_frame, textvariable=selected_file, values=file_names, state="readonly")
        file_menu.pack(fill=tk.X, expand=True)
        text_widget = scrolledtext.ScrolledText(info_window, wrap=tk.WORD, font=("Arial", 10), bg="white", fg="black")
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        def update_display(event=None):
            filename = selected_file.get()
            metadata = self.all_metadata.get(filename, {})
            text_widget.config(state=tk.NORMAL); text_widget.delete(1.0, tk.END)
            display_keys = {"date":"測定日", "sample name":"サンプル名", "comment":"コメント", "lock-in amp. sensitivity":"感度(mV)", "lock-in amp. time constant":"時定数(msec)", "measuring points":"測定点数", "max magnetic field":"最大磁場(Oe)", "max magnetization":"最大磁化(emu)", "lock-in amp. phase":"位相(deg)", "pole piece gap":"磁極間距離(mm)", "calibration value":"校正値"}
            info_text = f"ファイル: {filename}\n\n--- 測定パラメータ ---\n\n"
            for key, desc in display_keys.items(): info_text += f"■ {desc}\n  {metadata.get(key, '---')}\n\n"
            text_widget.insert(tk.END, info_text); text_widget.config(state=tk.DISABLED)
        file_menu.bind("<<ComboboxSelected>>", update_display)
        update_display()

    def load_files(self):
        files = filedialog.askopenfilenames(title="解析したいVSMファイルを選択", filetypes=[("VSM files", "*.VSM"), ("All files", "*.*")])
        if not files: return

        self.vsm_data, self.file_color_vars = [], []
        self._reset_individual_color_widgets()
        for i, file_path in enumerate(files):
            path = Path(file_path)
            header_row = find_header_row(path)
            try:
                try: df = pd.read_csv(path, header=header_row, encoding="shift-jis")
                except UnicodeDecodeError: df = pd.read_csv(path, header=header_row, encoding="utf-8")
                df.dropna(inplace=True)
                if not {"H(Oe)", "M(emu)"}.issubset(df.columns):
                    messagebox.showwarning("形式エラー", f"ファイル '{path.name}' に必要な列がありません。"); continue
                
                settings = {
                    "correction_var": tk.BooleanVar(value=True, name=f"diamag_{i}"),
                    "manual_var": tk.BooleanVar(value=False, name=f"manual_{i}"),
                    "pos_h_min_var": tk.StringVar(value="1.5", name=f"pos_min_{i}"),
                    "pos_h_max_var": tk.StringVar(value="2.0", name=f"pos_max_{i}"),
                    "neg_h_min_var": tk.StringVar(value="-2.0", name=f"neg_min_{i}"),
                    "neg_h_max_var": tk.StringVar(value="-1.5", name=f"neg_max_{i}")
                }
                for var in settings.values(): var.trace_add("write", self._schedule_update)
                settings["correction_var"].trace_add("write", lambda *a, s=settings: self.toggle_diamag_fields())
                settings["manual_var"].trace_add("write", lambda *a, s=settings: self.toggle_diamag_fields())

                self.vsm_data.append({"path": path, "df": df, "settings": settings})
                color_var = tk.StringVar(value=self.base_colors[i % len(self.base_colors)])
                self.file_color_vars.append(color_var)
                self._create_individual_color_widget(i, path.name, color_var)
            except Exception as e:
                messagebox.showerror("読込エラー", f"'{path.name}'の読込失敗:\n{e}")

        self.info_button.config(state=tk.NORMAL if self.vsm_data else tk.DISABLED)
        if self.vsm_data:
            self.diamag_file_combo['values'] = [d['path'].name for d in self.vsm_data]
            self.diamag_file_combo.current(0)
            self.on_diamag_file_selected()
        else:
            self.diamag_file_combo['values'] = []
            self.diamag_file_combo.set('')
            self.toggle_diamag_fields(is_init_or_empty=True)
        self.update_graph()
        
    def save_figure(self):
        try:
            w, h, dpi = float(self.save_width_var.get()), float(self.save_height_var.get()), int(self.save_dpi_var.get())
            if w <= 0 or h <= 0 or dpi <= 0: raise ValueError("値は正数である必要があります。")
        except ValueError as e:
            messagebox.showerror("入力エラー", f"幅,高さ,DPIには有効な正数を入力してください。\n({e})"); return
        file_path = filedialog.asksaveasfilename(title="画像を保存", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("JPEG", "*.jpg")], defaultextension=".png")
        if not file_path: return
        original_size = self.fig.get_size_inches()
        try:
            self.log_message(f"画像を保存中: {Path(file_path).name}\n  サイズ: {w}x{h} inches, DPI: {dpi}\n")
            self.fig.set_size_inches(w, h)
            self.fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
            self.log_message("保存が完了しました。\n")
            messagebox.showinfo("成功", f"画像を保存しました:\n{file_path}")
        except Exception as e:
            self.log_message(f"エラー: 画像保存失敗 - {e}\n")
            messagebox.showerror("保存エラー", f"画像保存中にエラーが発生:\n{e}")
        finally:
            self.fig.set_size_inches(original_size); self.canvas.draw_idle()

    def update_graph(self):
        self.log_text.delete(1.0, tk.END); self.ax.clear(); self.all_metadata = {}
        try:
            params = {
                "Thick": float(self.thick_var.get()), "Area": float(self.area_var.get()), "marker_size": float(self.marker_size_var.get()),
                "line_width": float(self.line_width_var.get()), "axis_label_fontsize": int(self.axis_label_fontsize_var.get()),
                "tick_label_fontsize": int(self.tick_label_fontsize_var.get()), "legend_fontsize": int(self.legend_fontsize_var.get()),
                "show_grid": self.show_grid_var.get(), "show_zero_lines": self.show_zero_lines_var.get(),
                "xlim_min": float(v) if (v := self.xlim_min_var.get()) else None, "xlim_max": float(v) if (v := self.xlim_max_var.get()) else None,
                "ylim_min": float(v) if (v := self.ylim_min_var.get()) else None, "ylim_max": float(v) if (v := self.ylim_max_var.get()) else None,
            }
        except ValueError:
            # 入力途中（例: 空欄や"-"のみ）の無効な値は無視し、エラー表示せず更新を中断
            return
        if not self.vsm_data:
            format_axis(self.ax, self.fig, params)
            self.ax.text(0.5, 0.5, "ファイルを選択してください", ha="center", va="center", color="gray", fontsize=16, transform=self.ax.transAxes)
            self.canvas.draw(); return
        output_stream = io.StringIO()
        with redirect_stdout(output_stream):
            print(f"解析開始: 膜厚={params['Thick']} nm, 面積={params['Area']} cm²\n")
            self._process_and_plot(params)
        self.log_message(output_stream.getvalue()); self.log_message("\n描画完了。\n")
        self.canvas.draw(); self.info_button.config(state=tk.NORMAL if self.all_metadata else tk.DISABLED)

    def _process_and_plot(self, params):
        Vol = params["Area"] * params["Thick"] * 1e-7
        h_min_global, h_max_global = float("inf"), float("-inf")
        print("読み込みファイル"); [print(f" {i+1}: {d['path'].name}") for i, d in enumerate(self.vsm_data)]
        for idx, data in enumerate(self.vsm_data):
            file, df, settings = data["path"], data["df"], data["settings"]
            try:
                self.all_metadata[file.name] = parse_metadata(file)
                min_H_idx = df["H(Oe)"].idxmin()
                if df["H(Oe)"].iloc[min_H_idx:].empty: raise ValueError("不完全なデータ。復路が見つかりません。")
                max_H_idx2 = min_H_idx + df["H(Oe)"].iloc[min_H_idx:].idxmax()
                df_loop = df.iloc[:max_H_idx2 + 1]
                H_raw, M_raw = df_loop["H(Oe)"] * 1e-4, df_loop["M(emu)"] / Vol
                print(f"\n--- 解析: {file.stem} (データ点: {len(H_raw)}) ---")

                if settings["correction_var"].get():
                    print("  反磁性補正: 有効")
                    if settings["manual_var"].get():
                        print("    傾き計算: 手動設定モード")
                        try:
                            pos_range = (float(settings["pos_h_min_var"].get()), float(settings["pos_h_max_var"].get()))
                            neg_range = (float(settings["neg_h_min_var"].get()), float(settings["neg_h_max_var"].get()))
                            slope, r2_pos, r2_neg = find_diamag_slope_manual(H_raw, M_raw, pos_range, neg_range)
                        except ValueError:
                            print("  エラー: 手動設定の磁場範囲が無効。"); slope, r2_pos, r2_neg = 0, 0, 0
                    else:
                        print("    傾き計算: 自動検出モード")
                        slope, r2_pos, r2_neg = find_diamag_slope_auto(H_raw, M_raw)
                    print(f"    補正傾き S: {slope:.6f}, R^2: [正 {r2_pos:.4f}], [負 {r2_neg:.4f}]")
                else:
                    print("  反磁性補正: 無効"); slope = 0

                M_corrected = M_raw - H_raw * slope
                if self.offset_correction_var.get():
                    print("  磁化オフセット補正: 有効")
                    H_np, M_np = H_raw.values, M_corrected.values
                    Ms_pos = np.mean(M_np[H_np > np.max(H_np)*0.9]) if np.any(H_np > np.max(H_np)*0.9) else 0
                    Ms_neg = np.mean(M_np[H_np < np.min(H_np)*0.9]) if np.any(H_np < np.min(H_np)*0.9) else 0
                    offset = (Ms_pos + Ms_neg) / 2
                    M_final = M_corrected - offset
                    print(f"    補正値: {offset:.4f} kA/m")
                else:
                    print("  磁化オフセット補正: 無効"); M_final = M_corrected

                min_H_idx_loop = H_raw.idxmin()
                H_down, M_down = H_raw.iloc[:min_H_idx_loop+1].values, M_final.iloc[:min_H_idx_loop+1].values
                H_up, M_up = H_raw.iloc[min_H_idx_loop:].values, M_final.iloc[min_H_idx_loop:].values
                color = self.file_color_vars[idx].get() if idx < len(self.file_color_vars) else self.base_colors[idx % len(self.base_colors)]
                plot_kwargs = {"marker":"o", "markersize":params["marker_size"], "linestyle":"-", "linewidth":params["line_width"]}
                self.ax.plot(H_down, M_down, color=color, **plot_kwargs)
                self.ax.plot(H_up, M_up, color=color, label=file.stem, **plot_kwargs)

                if params["xlim_min"] is None and params["xlim_max"] is None:
                    h_min_global, h_max_global = min(h_min_global, H_raw.min()), max(h_max_global, H_raw.max())
                    params["xlim_min"], params["xlim_max"] = h_min_global, h_max_global
                
                calculate_saturation_magnetization(H_raw, M_final)
                calculate_remanence(H_down, M_down, H_up, M_up)
                calculate_coercivity(H_down, M_down, H_up, M_up)
            except Exception as e:
                print(f"\nエラー: '{file.name}'の処理中に問題発生: {e}")
                import traceback; traceback.print_exc(file=sys.stdout)
                continue
        format_axis(self.ax, self.fig, params)
        if self.show_legend_var.get() and any(self.ax.get_legend_handles_labels()[1]):
            self.ax.legend(fontsize=params["legend_fontsize"], loc="best", facecolor="white", edgecolor="#DDDDDD", labelcolor="black")
        self.fig.tight_layout()

if __name__ == "__main__":
    root = tk.Tk()
    app = VSMApp(root)
    root.mainloop()
