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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
                # ファイルの先頭から最大100行までを探索
                for i, line in enumerate(f):
                    if "H(Oe)" in line and "M(emu)" in line:
                        print(f"  情報: ヘッダーを {i + 1} 行目で検出。")
                        return i  # 0-indexedの行番号を返す
                    if i > 100:
                        break
        except (UnicodeDecodeError, IOError):
            continue  # エラーの場合は次のエンコーディングを試す

    print(
        f"  警告: ヘッダー行を自動検出できず。デフォルト値({default_row + 1}行目)を使用。"
    )
    return default_row


def parse_metadata(file_path):
    """
    ファイルのヘッダーから測定メタデータを抽出し、辞書として返す。
    この処理でエラーが発生しても、メインの解析は停止しない。
    """
    metadata = {}
    try:
        encodings_to_try = ["shift-jis", "utf-8"]
        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    # ファイルの先頭から最大40行までを探索（行数指定は暫定対応）
                    for i, line in enumerate(f):
                        if i > 40:
                            break
                        try:
                            line = line.strip()
                            if "=" in line:
                                parts = line.split("=", 1)
                                key = parts[0].strip()
                                value_part = parts[1]
                                # "key=,value,unit" 形式を想定
                                if value_part.startswith(","):
                                    value_parts = value_part.split(",")
                                    if len(value_parts) > 1:
                                        value = value_parts[1].strip()
                                        if key and value:
                                            metadata[key] = value
                        except IndexError:
                            # 行のフォーマットが想定外でも処理を続行
                            continue
                if metadata:
                    return metadata  # 読み取り成功
            except (UnicodeDecodeError, IOError):
                continue  # 別のエンコーディングを試す
    except Exception as e:
        print(
            f"  警告: メタデータ読み取り中に予期せぬエラー発生: {e}。解析は続行します。"
        )
    return metadata


def find_demag_slope_auto(H_data, M_data, segment_ratio=0.15, min_r_squared=0.99):
    """M-Hカーブ両端から線形性の高い領域を自動検出し、反磁性補正の傾きを計算"""
    df = (
        pd.DataFrame({"H": H_data, "M": M_data})
        .sort_values(by="H")
        .reset_index(drop=True)
    )

    n_points = len(df)
    segment_size = max(5, int(n_points * segment_ratio))

    if n_points < segment_size * 2:
        print("  警告: データ点数不足のため、傾きの自動検出をスキップ。")
        return 0, 0, 0

    slope_pos, r2_pos = 0, 0
    try:
        pos_segment = df.tail(segment_size)
        res_pos = linregress(pos_segment["H"], pos_segment["M"])
        slope_pos = res_pos.slope
        r2_pos = res_pos.rvalue**2
        if r2_pos < min_r_squared:
            print(f"  警告: 正磁場側の線形性が低い (R^2 = {r2_pos:.4f})。")
    except (ValueError, np.linalg.LinAlgError):
        print("  警告: 正磁場側のフィッティング失敗。")

    slope_neg, r2_neg = 0, 0
    try:
        neg_segment = df.head(segment_size)
        res_neg = linregress(neg_segment["H"], neg_segment["M"])
        slope_neg = res_neg.slope
        r2_neg = res_neg.rvalue**2
        if r2_neg < min_r_squared:
            print(f"  警告: 負磁場側の線形性が低い (R^2 = {r2_neg:.4f})。")
    except (ValueError, np.linalg.LinAlgError):
        print("  警告: 負磁場側のフィッティング失敗。")

    return (slope_pos + slope_neg) / 2, r2_pos, r2_neg


def find_demag_slope_manual(H_data, M_data, pos_range, neg_range):
    """指定磁場範囲で線形フィッティングを行い、傾きを計算"""
    df = pd.DataFrame({"H": H_data, "M": M_data})

    slope_pos, r2_pos = 0, 0
    try:
        pos_mask = (df["H"] >= pos_range[0]) & (df["H"] <= pos_range[1])
        num_pos_points = pos_mask.sum()
        print(
            f"  手動(正): H=[{pos_range[0]:.2f}, {pos_range[1]:.2f}] T, 点数: {num_pos_points}"
        )
        if num_pos_points < 2:
            print("  エラー: 正磁場範囲のデータ点が2点未満。傾き計算不可。")
        else:
            if num_pos_points < 5:
                print(
                    f"  警告: 正磁場範囲のデータ点数僅少({num_pos_points}点)。結果の信頼性に注意。"
                )
            res_pos = linregress(df.loc[pos_mask, "H"], df.loc[pos_mask, "M"])
            slope_pos = res_pos.slope
            r2_pos = res_pos.rvalue**2
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  警告: 正磁場範囲のフィッティング失敗: {e}")

    slope_neg, r2_neg = 0, 0
    try:
        neg_mask = (df["H"] >= neg_range[0]) & (df["H"] <= neg_range[1])
        num_neg_points = neg_mask.sum()
        print(
            f"  手動(負): H=[{neg_range[0]:.2f}, {neg_range[1]:.2f}] T, 点数: {num_neg_points}"
        )
        if num_neg_points < 2:
            print("  エラー: 負磁場範囲のデータ点が2点未満。傾き計算不可。")
        else:
            if num_neg_points < 5:
                print(
                    f"  警告: 負磁場範囲のデータ点数僅少({num_neg_points}点)。結果の信頼性に注意。"
                )
            res_neg = linregress(df.loc[neg_mask, "H"], df.loc[neg_mask, "M"])
            slope_neg = res_neg.slope
            r2_neg = res_neg.rvalue**2
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  警告: 負磁場範囲のフィッティング失敗: {e}")

    if slope_pos != 0 and slope_neg != 0:
        return (slope_pos + slope_neg) / 2, r2_pos, r2_neg
    elif slope_pos != 0:
        print("  情報: 正磁場側の傾きのみ使用。")
        return slope_pos, r2_pos, r2_neg
    elif slope_neg != 0:
        print("  情報: 負磁場側の傾きのみ使用。")
        return slope_neg, r2_pos, r2_neg
    else:
        return 0, 0, 0


def calculate_remanence(H_down, M_down, H_up, M_up):
    """残留磁化(Mr)を往路・復路データから計算"""
    try:
        Mr_down = np.interp(0, H_down[::-1], M_down[::-1])
        Mr_up = np.interp(0, H_up, M_up)
        Mr_avg = (abs(Mr_down) + abs(Mr_up)) / 2
        print(f"  残留磁化 Mr: {Mr_avg:.3f} kA/m")
        return Mr_avg
    except Exception as e:
        print(f"  エラー: 残留磁化の計算失敗: {e}")
        return None


def calculate_coercivity(H_down, M_down, H_up, M_up):
    """保磁力(Hc)を計算"""
    try:
        Hc_down = np.interp(0, M_down[::-1], H_down[::-1])
        Hc_up = np.interp(0, M_up, H_up)
        Hc_avg = (abs(Hc_down) + abs(Hc_up)) / 2
        print(f"  保磁力 Hc: {Hc_avg * 10000:.2f} Oe ({Hc_avg:.4f} T)")
        return Hc_avg
    except Exception as e:
        print(f"  エラー: 保磁力の計算失敗: {e}")
        return None


def calculate_saturation_magnetization(H, M):
    """飽和磁化(Ms)を計算"""
    H, M = np.array(H), np.array(M)
    H_max_val, H_min_val = np.max(H), np.min(H)
    mask_pos = H > H_max_val * 0.9
    mask_neg = H < H_min_val * 0.9
    Ms_pos = np.mean(M[mask_pos]) if np.any(mask_pos) else 0
    Ms_neg = np.mean(np.abs(M[mask_neg])) if np.any(mask_neg) else 0
    Ms_avg = (Ms_pos + Ms_neg) / 2
    print(f"  飽和磁化 Ms: {Ms_avg:.3f} kA/m (正側: {Ms_pos:.3f}, 負側: {Ms_neg:.3f})")
    return Ms_avg


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
                # ファイルの先頭から最大100行までを探索
                for i, line in enumerate(f):
                    if "H(Oe)" in line and "M(emu)" in line:
                        print(f"  情報: ヘッダーを {i + 1} 行目で検出。")
                        return i  # 0-indexedの行番号を返す
                    if i > 100:
                        break
        except (UnicodeDecodeError, IOError):
            continue  # エラーの場合は次のエンコーディングを試す

    print(
        f"  警告: ヘッダー行を自動検出できず。デフォルト値({default_row + 1}行目)を使用。"
    )
    return default_row


def parse_metadata(file_path):
    """
    ファイルのヘッダーから測定メタデータを抽出し、辞書として返す。
    この処理でエラーが発生しても、メインの解析は停止しない。
    """
    metadata = {}
    try:
        encodings_to_try = ["shift-jis", "utf-8"]
        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    # ファイルの先頭から最大40行までを探索（行数指定は暫定対応）
                    for i, line in enumerate(f):
                        if i > 40:
                            break
                        try:
                            line = line.strip()
                            if "=" in line:
                                parts = line.split("=", 1)
                                key = parts[0].strip()
                                value_part = parts[1]
                                # "key=,value,unit" 形式を想定
                                if value_part.startswith(","):
                                    value_parts = value_part.split(",")
                                    if len(value_parts) > 1:
                                        value = value_parts[1].strip()
                                        if key and value:
                                            metadata[key] = value
                        except IndexError:
                            # 行のフォーマットが想定外でも処理を続行
                            continue
                if metadata:
                    return metadata  # 読み取り成功
            except (UnicodeDecodeError, IOError):
                continue  # 別のエンコーディングを試す
    except Exception as e:
        print(
            f"  警告: メタデータ読み取り中に予期せぬエラー発生: {e}。解析は続行します。"
        )
    return metadata


def find_demag_slope_auto(H_data, M_data, segment_ratio=0.15, min_r_squared=0.99):
    """M-Hカーブ両端から線形性の高い領域を自動検出し、反磁性補正の傾きを計算"""
    df = (
        pd.DataFrame({"H": H_data, "M": M_data})
        .sort_values(by="H")
        .reset_index(drop=True)
    )

    n_points = len(df)
    segment_size = max(5, int(n_points * segment_ratio))

    if n_points < segment_size * 2:
        print("  警告: データ点数不足のため、傾きの自動検出をスキップ。")
        return 0, 0, 0

    slope_pos, r2_pos = 0, 0
    try:
        pos_segment = df.tail(segment_size)
        res_pos = linregress(pos_segment["H"], pos_segment["M"])
        slope_pos = res_pos.slope
        r2_pos = res_pos.rvalue**2
        if r2_pos < min_r_squared:
            print(f"  警告: 正磁場側の線形性が低い (R^2 = {r2_pos:.4f})。")
    except (ValueError, np.linalg.LinAlgError):
        print("  警告: 正磁場側のフィッティング失敗。")

    slope_neg, r2_neg = 0, 0
    try:
        neg_segment = df.head(segment_size)
        res_neg = linregress(neg_segment["H"], neg_segment["M"])
        slope_neg = res_neg.slope
        r2_neg = res_neg.rvalue**2
        if r2_neg < min_r_squared:
            print(f"  警告: 負磁場側の線形性が低い (R^2 = {r2_neg:.4f})。")
    except (ValueError, np.linalg.LinAlgError):
        print("  警告: 負磁場側のフィッティング失敗。")

    return (slope_pos + slope_neg) / 2, r2_pos, r2_neg


def find_demag_slope_manual(H_data, M_data, pos_range, neg_range):
    """指定磁場範囲で線形フィッティングを行い、傾きを計算"""
    df = pd.DataFrame({"H": H_data, "M": M_data})

    slope_pos, r2_pos = 0, 0
    try:
        pos_mask = (df["H"] >= pos_range[0]) & (df["H"] <= pos_range[1])
        num_pos_points = pos_mask.sum()
        print(
            f"  手動(正): H=[{pos_range[0]:.2f}, {pos_range[1]:.2f}] T, 点数: {num_pos_points}"
        )
        if num_pos_points < 2:
            print("  エラー: 正磁場範囲のデータ点が2点未満。傾き計算不可。")
        else:
            if num_pos_points < 5:
                print(
                    f"  警告: 正磁場範囲のデータ点数僅少({num_pos_points}点)。結果の信頼性に注意。"
                )
            res_pos = linregress(df.loc[pos_mask, "H"], df.loc[pos_mask, "M"])
            slope_pos = res_pos.slope
            r2_pos = res_pos.rvalue**2
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  警告: 正磁場範囲のフィッティング失敗: {e}")

    slope_neg, r2_neg = 0, 0
    try:
        neg_mask = (df["H"] >= neg_range[0]) & (df["H"] <= neg_range[1])
        num_neg_points = neg_mask.sum()
        print(
            f"  手動(負): H=[{neg_range[0]:.2f}, {neg_range[1]:.2f}] T, 点数: {num_neg_points}"
        )
        if num_neg_points < 2:
            print("  エラー: 負磁場範囲のデータ点が2点未満。傾き計算不可。")
        else:
            if num_neg_points < 5:
                print(
                    f"  警告: 負磁場範囲のデータ点数僅少({num_neg_points}点)。結果の信頼性に注意。"
                )
            res_neg = linregress(df.loc[neg_mask, "H"], df.loc[neg_mask, "M"])
            slope_neg = res_neg.slope
            r2_neg = res_neg.rvalue**2
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  警告: 負磁場範囲のフィッティング失敗: {e}")

    if slope_pos != 0 and slope_neg != 0:
        return (slope_pos + slope_neg) / 2, r2_pos, r2_neg
    elif slope_pos != 0:
        print("  情報: 正磁場側の傾きのみ使用。")
        return slope_pos, r2_pos, r2_neg
    elif slope_neg != 0:
        print("  情報: 負磁場側の傾きのみ使用。")
        return slope_neg, r2_pos, r2_neg
    else:
        return 0, 0, 0


def calculate_remanence(H_down, M_down, H_up, M_up):
    """残留磁化(Mr)を往路・復路データから計算"""
    try:
        Mr_down = np.interp(0, H_down[::-1], M_down[::-1])
        Mr_up = np.interp(0, H_up, M_up)
        Mr_avg = (abs(Mr_down) + abs(Mr_up)) / 2
        print(f"  残留磁化 Mr: {Mr_avg:.3f} kA/m")
        return Mr_avg
    except Exception as e:
        print(f"  エラー: 残留磁化の計算失敗: {e}")
        return None


def calculate_coercivity(H_down, M_down, H_up, M_up):
    """保磁力(Hc)を計算"""
    try:
        Hc_down = np.interp(0, M_down[::-1], H_down[::-1])
        Hc_up = np.interp(0, M_up, H_up)
        Hc_avg = (abs(Hc_down) + abs(Hc_up)) / 2
        print(f"  保磁力 Hc: {Hc_avg * 10000:.2f} Oe ({Hc_avg:.4f} T)")
        return Hc_avg
    except Exception as e:
        print(f"  エラー: 保磁力の計算失敗: {e}")
        return None


def calculate_saturation_magnetization(H, M):
    """飽和磁化(Ms)を計算"""
    H, M = np.array(H), np.array(M)
    H_max_val, H_min_val = np.max(H), np.min(H)
    mask_pos = H > H_max_val * 0.9
    mask_neg = H < H_min_val * 0.9
    Ms_pos = np.mean(M[mask_pos]) if np.any(mask_pos) else 0
    Ms_neg = np.mean(np.abs(M[mask_neg])) if np.any(mask_neg) else 0
    Ms_avg = (Ms_pos + Ms_neg) / 2
    print(f"  飽和磁化 Ms: {Ms_avg:.3f} kA/m (正側: {Ms_pos:.3f}, 負側: {Ms_neg:.3f})")
    return Ms_avg


def format_axis(ax, fig, style_params):
    """グラフの軸や目盛りなどを整形"""
    ax.set_xlabel(r"$\mu_0H$ [T]", fontsize=style_params.get("axis_label_fontsize", 16))
    ax.set_ylabel(r"M [kA/m]", fontsize=style_params.get("axis_label_fontsize", 16))

    # 原点線を設定
    if style_params.get("show_zero_lines", True):
        ax.axhline(0, color="#888888", linestyle="-", linewidth=1.0)
        ax.axvline(0, color="#888888", linestyle="-", linewidth=1.0)

    # グリッドを設定
    if style_params.get("show_grid", True):
        ax.grid(True, linestyle=":", color="#555555")

    # 軸範囲を設定
    if (
        style_params.get("xlim_min") is not None
        and style_params.get("xlim_max") is not None
    ):
        ax.set_xlim(style_params["xlim_min"], style_params["xlim_max"])
    if (
        style_params.get("ylim_min") is not None
        and style_params.get("ylim_max") is not None
    ):
        ax.set_ylim(style_params["ylim_min"], style_params["ylim_max"])

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=style_params.get("tick_label_fontsize", 12),
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.0,
        colors="#EAEAEA",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8,
        colors="#EAEAEA",
    )
    ax.spines["top"].set_color("#EAEAEA")
    ax.spines["bottom"].set_color("#EAEAEA")
    ax.spines["left"].set_color("#EAEAEA")
    ax.spines["right"].set_color("#EAEAEA")
    ax.xaxis.label.set_color("#EAEAEA")
    ax.yaxis.label.set_color("#EAEAEA")
    ax.title.set_color("#EAEAEA")

    fig.patch.set_facecolor("#2E2E2E")
    ax.set_facecolor("#222222")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]


# GUIアプリケーションのクラス (リファクタリング後)
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
        self.root.configure(bg="#2E2E2E")

        # --- スタイル変数 ---
        # 解析タブ
        self.thick_var = tk.StringVar(value="100.0")
        self.area_var = tk.StringVar(value="1.0")
        self.offset_correction_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)
        self.demag_correction_var = tk.BooleanVar(value=True)
        self.manual_slope_var = tk.BooleanVar(value=False)
        self.pos_h_min_var = tk.StringVar(value="1.5")
        self.pos_h_max_var = tk.StringVar(value="2.0")
        self.neg_h_min_var = tk.StringVar(value="-2.0")
        self.neg_h_max_var = tk.StringVar(value="-1.5")

        # グラフ設定タブ
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

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self._configure_styles()

        # --- メインレイアウト ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # --- 左パネルをタブUIに変更 ---
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        tab_analysis = ttk.Frame(notebook, padding="10")
        tab_style = ttk.Frame(notebook, padding="10")
        # tab_export = ttk.Frame(notebook, padding="10") # 保存タブはツールバーに統合

        notebook.add(tab_analysis, text="解析")
        notebook.add(tab_style, text="グラフ設定")
        # notebook.add(tab_export, text="保存")

        graph_frame = ttk.LabelFrame(main_frame, text=" グラフ ", padding=10)
        graph_frame.grid(row=0, column=1, sticky="nsew")
        graph_frame.grid_rowconfigure(1, weight=1)  # canvas用
        graph_frame.grid_columnconfigure(0, weight=1)

        # --- 各タブのコントロールを作成 ---
        self._create_analysis_controls(tab_analysis)
        self._create_style_controls(tab_style)
        # self._create_export_controls(tab_export)

        # --- ログフレームをタブUIの外に配置 ---
        log_frame = ttk.LabelFrame(main_frame, text=" ログ ", padding="10", height=150)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        log_frame.grid_propagate(False)  # 高さが固定されるように
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 9), bg="#1E1E1E", fg="#D4D4D4"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- グラフの埋め込みとツールバー ---
        self.fig = plt.figure(figsize=(9, 9), facecolor="#2E2E2E")
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)

        # ツールバーを生成して配置
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame, pack_toolbar=False)
        toolbar.config(background="#2E2E2E")
        toolbar._message_label.config(background="#2E2E2E", foreground="#EAEAEA")
        # ボタンのテーマ追従のため
        for button in toolbar.winfo_children():
            if isinstance(button, (tk.Button, tk.Checkbutton)):
                button.config(
                    background="#2E2E2E",
                    foreground="#EAEAEA",
                    highlightbackground="#2E2E2E",
                )

        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="ew", padx=5)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # --- 変数のトレース設定 ---
        self._add_traces()

        self.toggle_demag_fields()
        self.update_graph()  # 初期グラフ描画

    def _configure_styles(self):
        """UIコンポーネントのスタイルを設定"""
        self.style.configure(
            ".", background="#2E2E2E", foreground="#EAEAEA", font=("Arial", 10)
        )
        self.style.configure("TLabel", background="#2E2E2E", foreground="#EAEAEA")
        self.style.configure("TFrame", background="#2E2E2E")
        self.style.configure("TNotebook", background="#2E2E2E", borderwidth=0)
        self.style.configure(
            "TNotebook.Tab",
            background="#2E2E2E",
            foreground="#AAAAAA",
            padding=[10, 5],
            font=("Arial", 10, "bold"),
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", "#007ACC")],
            foreground=[("selected", "white")],
        )

        self.style.configure(
            "TEntry", fieldbackground="#3C3C3C", foreground="white", insertcolor="white"
        )
        self.style.configure(
            "TLabelframe",
            background="#2E2E2E",
            bordercolor="#555555",
            foreground="#EAEAEA",
        )
        self.style.configure(
            "TLabelframe.Label",
            background="#2E2E2E",
            foreground="#00AEEF",
            font=("Arial", 11, "bold"),
        )
        self.style.configure(
            "TButton",
            background="#007ACC",
            foreground="white",
            font=("Arial", 11, "bold"),
            borderwidth=0,
        )
        self.style.map("TButton", background=[("active", "#005F9E")])
        self.style.configure("TCheckbutton", background="#2E2E2E", foreground="#EAEAEA")

    def _create_analysis_controls(self, parent):
        """「解析」タブのウィジェットを作成"""
        # ファイル操作フレーム
        file_frame = ttk.LabelFrame(parent, text=" ファイル ", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        self.load_button = ttk.Button(
            file_frame, text="ファイルを選択", command=self.load_files, padding="10 5"
        )
        self.load_button.pack(fill=tk.X)
        self.info_button = ttk.Button(
            file_frame,
            text="測定情報を表示",
            command=self.show_metadata_window,
            state=tk.DISABLED,
        )
        self.info_button.pack(fill=tk.X, pady=(5, 0))

        # 解析設定フレーム
        settings_frame = ttk.LabelFrame(parent, text=" 解析設定 ", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        settings_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="膜厚 (nm):").grid(row=0, column=0, sticky="w")
        self.thick_var = tk.StringVar(value="100.0")
        self.thick_entry = ttk.Entry(
            settings_frame, textvariable=self.thick_var, width=10
        )
        self.thick_entry.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(settings_frame, text="基板面積 (cm²):").grid(
            row=1, column=0, sticky="w"
        )
        self.area_var = tk.StringVar(value="1.0")
        self.area_entry = ttk.Entry(
            settings_frame, textvariable=self.area_var, width=10
        )
        self.area_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 0))

        self.offset_correction_var = tk.BooleanVar(value=True)
        self.offset_check = ttk.Checkbutton(
            settings_frame,
            text="磁化オフセット補正",
            variable=self.offset_correction_var,
        )
        self.offset_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.show_legend_var = tk.BooleanVar(value=True)
        self.legend_check = ttk.Checkbutton(
            settings_frame, text="凡例を表示", variable=self.show_legend_var
        )
        self.legend_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # 反磁性補正フレーム
        demag_frame = ttk.LabelFrame(parent, text=" 反磁性補正 ", padding="10")
        demag_frame.pack(fill=tk.X, pady=(0, 10))
        demag_frame.grid_columnconfigure(1, weight=1)

        self.demag_correction_var = tk.BooleanVar(value=True)
        self.demag_check = ttk.Checkbutton(
            demag_frame, text="反磁性補正を有効化", variable=self.demag_correction_var
        )
        self.demag_check.grid(row=0, column=0, columnspan=4, sticky="w")

        self.manual_slope_var = tk.BooleanVar(value=False)
        self.manual_check = ttk.Checkbutton(
            demag_frame, text="傾き計算の範囲を手動指定", variable=self.manual_slope_var
        )
        self.manual_check.grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 5))

        ttk.Label(demag_frame, text="正 H (T):").grid(row=2, column=0, sticky="w")
        self.pos_h_min_var = tk.StringVar(value="1.5")
        self.pos_h_min_entry = ttk.Entry(
            demag_frame, textvariable=self.pos_h_min_var, width=7
        )
        self.pos_h_min_entry.grid(row=2, column=1, sticky="ew")
        ttk.Label(demag_frame, text="～").grid(row=2, column=2)
        self.pos_h_max_var = tk.StringVar(value="2.0")
        self.pos_h_max_entry = ttk.Entry(
            demag_frame, textvariable=self.pos_h_max_var, width=7
        )
        self.pos_h_max_entry.grid(row=2, column=3, sticky="ew")

        ttk.Label(demag_frame, text="負 H (T):").grid(
            row=3, column=0, sticky="w", pady=(5, 0)
        )
        self.neg_h_min_var = tk.StringVar(value="-2.0")
        self.neg_h_min_entry = ttk.Entry(
            demag_frame, textvariable=self.neg_h_min_var, width=7
        )
        self.neg_h_min_entry.grid(row=3, column=1, sticky="ew", pady=(5, 0))
        ttk.Label(demag_frame, text="～").grid(row=3, column=2, pady=(5, 0))
        self.neg_h_max_var = tk.StringVar(value="-1.5")
        self.neg_h_max_entry = ttk.Entry(
            demag_frame, textvariable=self.neg_h_max_var, width=7
        )
        self.neg_h_max_entry.grid(row=3, column=3, sticky="ew", pady=(5, 0))

    def _create_style_controls(self, parent):
        """「グラフ設定」タブのウィジェットを作成"""
        parent.grid_columnconfigure(1, weight=1)

        # --- 軸・グリッド設定 ---
        axis_grid_frame = ttk.LabelFrame(parent, text=" 軸とグリッド ", padding="10")
        axis_grid_frame.pack(fill=tk.X, pady=(0, 10))

        self.grid_check = ttk.Checkbutton(
            axis_grid_frame, text="グリッド線を表示", variable=self.show_grid_var
        )
        self.grid_check.pack(anchor="w")

        self.zero_lines_check = ttk.Checkbutton(
            axis_grid_frame, text="原点線を表示", variable=self.show_zero_lines_var
        )
        self.zero_lines_check.pack(anchor="w", pady=(5, 0))

        # --- プロット設定 ---
        plot_frame = ttk.LabelFrame(parent, text=" プロット ", padding="10")
        plot_frame.pack(fill=tk.X, pady=(0, 10))
        plot_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(plot_frame, text="マーカーサイズ:").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        marker_size_entry = ttk.Entry(
            plot_frame, textvariable=self.marker_size_var, width=10
        )
        marker_size_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 5))

        ttk.Label(plot_frame, text="線幅:").grid(
            row=1, column=0, sticky="w", pady=(0, 5)
        )
        line_width_entry = ttk.Entry(
            plot_frame, textvariable=self.line_width_var, width=10
        )
        line_width_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=(0, 5))

        # --- フォント設定 ---
        font_frame = ttk.LabelFrame(parent, text=" フォントサイズ ", padding="10")
        font_frame.pack(fill=tk.X, pady=(0, 10))
        font_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(font_frame, text="軸ラベル:").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        axis_label_fs_entry = ttk.Entry(
            font_frame, textvariable=self.axis_label_fontsize_var, width=10
        )
        axis_label_fs_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 5))

        ttk.Label(font_frame, text="目盛り:").grid(
            row=1, column=0, sticky="w", pady=(0, 5)
        )
        tick_label_fs_entry = ttk.Entry(
            font_frame, textvariable=self.tick_label_fontsize_var, width=10
        )
        tick_label_fs_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=(0, 5))

        ttk.Label(font_frame, text="凡例:").grid(
            row=2, column=0, sticky="w", pady=(0, 5)
        )
        legend_fs_entry = ttk.Entry(
            font_frame, textvariable=self.legend_fontsize_var, width=10
        )
        legend_fs_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=(0, 5))

        # --- ファイルごとの色設定 ---
        self.individual_color_frame = ttk.LabelFrame(
            parent, text=" 各ファイルの色 ", padding="10"
        )
        self.individual_color_frame.pack(fill=tk.X, pady=(10, 0))
        self.individual_color_frame.grid_columnconfigure(1, weight=1)

        # --- 描画範囲 ---
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

    def _add_traces(self):
        """ウィジェットの変数変更を監視し、更新関数を呼び出す"""
        trace_vars = [
            self.thick_var,
            self.area_var,
            self.offset_correction_var,
            self.demag_correction_var,
            self.manual_slope_var,
            self.pos_h_min_var,
            self.pos_h_max_var,
            self.neg_h_min_var,
            self.neg_h_max_var,
            self.show_legend_var,
            # Style vars
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
        ]
        for var in trace_vars:
            var.trace_add("write", self._schedule_update)

        self.demag_correction_var.trace_add(
            "write", lambda *args: self.toggle_demag_fields()
        )
        self.manual_slope_var.trace_add(
            "write", lambda *args: self.toggle_manual_fields()
        )

    def _reset_individual_color_widgets(self):
        """個別の色設定ウィジェットをクリアする"""
        if not hasattr(self, "individual_color_frame"):
            return
        for widget in self.individual_color_frame.winfo_children():
            widget.destroy()

    def _create_individual_color_widget(self, index, filename, color_var):
        """ファイルごとの色設定ウィジェットを作成する"""
        row_frame = ttk.Frame(self.individual_color_frame)
        row_frame.pack(fill=tk.X, pady=2)

        display_name = (filename[:25] + "..") if len(filename) > 27 else filename
        label = ttk.Label(row_frame, text=f"{display_name}:", anchor="w")
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        preview = tk.Label(row_frame, text="", bg=color_var.get(), width=4)
        preview.pack(side=tk.RIGHT, padx=5)

        def on_color_change(*args):
            new_color = color_var.get()
            preview.config(bg=new_color)
            self._schedule_update()

        color_var.trace_add("write", on_color_change)

        button = ttk.Button(
            row_frame,
            text="選択",
            width=5,
            command=lambda idx=index: self.choose_individual_color(idx),
        )
        button.pack(side=tk.RIGHT)

    def choose_individual_color(self, index):
        """個別のファイルに対応するカラーピーカーを開く"""
        if index >= len(self.file_color_vars):
            return

        color_var = self.file_color_vars[index]
        initial_color = color_var.get()

        # ファイル名が長すぎる場合があるため、表示用に短縮
        path_name = self.vsm_data[index]["path"].name
        title_name = (path_name[:40] + "..") if len(path_name) > 42 else path_name

        color_code = colorchooser.askcolor(
            title=f"'{title_name}' の色を選択", initialcolor=initial_color
        )
        if color_code and color_code[1]:
            color_var.set(color_code[1])

    def _schedule_update(self, *args):
        """更新処理を予約する（連続的なイベントを間引く）"""
        if self._update_job:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(250, self.update_graph)

    def toggle_demag_fields(self):
        """反磁性補正の有効/無効に応じて、関連ウィジェットの状態を切り替える"""
        state = tk.NORMAL if self.demag_correction_var.get() else tk.DISABLED
        self.manual_check.config(state=state)
        self.toggle_manual_fields()

    def toggle_manual_fields(self):
        """手動設定の有効/無効に応じて、入力フィールドの状態を切り替える"""
        state = (
            tk.NORMAL
            if self.demag_correction_var.get() and self.manual_slope_var.get()
            else tk.DISABLED
        )
        for widget in [
            self.pos_h_min_entry,
            self.pos_h_max_entry,
            self.neg_h_min_entry,
            self.neg_h_max_entry,
        ]:
            widget.config(state=state)

    def log_message(self, message):
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def show_metadata_window(self):
        if not self.all_metadata:
            messagebox.showinfo("情報", "表示できる測定情報がありません。")
            return

        info_window = tk.Toplevel(self.root)
        info_window.title("測定情報")
        info_window.geometry("500x650")
        info_window.configure(bg="#2E2E2E")

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
            info_window, wrap=tk.WORD, font=("Arial", 10), bg="#1E1E1E", fg="#D4D4D4"
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
                "lock-in amp. sensitivity": "ロックインアンプ感度 (mV)",
                "lock-in amp. time constant": "ロックインアンプ時定数 (msec)",
                "measuring points": "測定点数",
                "max magnetic field": "最大印加磁場 (Oe)",
                "max magnetization": "最大磁化 (emu)",
                "lock-in amp. phase": "ロックインアンプ位相 (degree)",
                "pole piece gap": "磁極間距離 (mm)",
                "calibration value": "校正値",
            }

            info_text = f"ファイル: {filename}\n\n--- 測定パラメータ ---\n\n"
            for key, description in display_keys.items():
                value = metadata.get(key, "---")
                info_text += f"■ {description}\n  {value}\n\n"
            text_widget.insert(tk.END, info_text)
            text_widget.config(state=tk.DISABLED)

        file_menu.bind("<<ComboboxSelected>>", update_display)
        update_display()

    def load_files(self):
        """ファイルを選択し、データを読み込んで解析を開始する"""
        files = filedialog.askopenfilenames(
            title="解析したいVSMファイルを選択 (Ctrl/Shiftで複数選択)",
            filetypes=[("VSM files", "*.VSM"), ("All files", "*.*")],
        )
        if not files:
            return

        self.vsm_data = []
        self.file_color_vars = []
        self._reset_individual_color_widgets()

        for i, file_path in enumerate(files):
            path = Path(file_path)
            header_row = find_header_row(path)
            try:
                try:
                    df = pd.read_csv(path, header=header_row, encoding="shift-jis")
                except UnicodeDecodeError:
                    df = pd.read_csv(path, header=header_row, encoding="utf-8")
                df.dropna(inplace=True)

                if not {"H(Oe)", "M(emu)"}.issubset(df.columns):
                    messagebox.showwarning(
                        "形式エラー",
                        f"ファイル '{path.name}' に必要な列(H(Oe), M(emu))がありません。",
                    )
                    continue

                self.vsm_data.append({"path": path, "df": df})
                # ファイルごとの色設定UIを生成
                color_var = tk.StringVar(
                    value=self.base_colors[i % len(self.base_colors)]
                )
                self.file_color_vars.append(color_var)
                self._create_individual_color_widget(i, path.name, color_var)

            except Exception as e:
                messagebox.showerror(
                    "読み込みエラー",
                    f"ファイル '{path.name}' の読み込みに失敗しました:\n{e}",
                )

        if self.vsm_data:
            self.info_button.config(state=tk.NORMAL)
        else:
            self.info_button.config(state=tk.DISABLED)

        self.update_graph()

    def update_graph(self):
        """現在の設定に基づいてグラフを再描画および再計算する"""
        self.log_text.delete(1.0, tk.END)
        self.ax.clear()
        self.all_metadata = {}

        # --- 入力値の検証 ---
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
            }

            def to_float_or_none(val):
                return float(val) if val else None

            params["xlim_min"] = to_float_or_none(self.xlim_min_var.get())
            params["xlim_max"] = to_float_or_none(self.xlim_max_var.get())
            params["ylim_min"] = to_float_or_none(self.ylim_min_var.get())
            params["ylim_max"] = to_float_or_none(self.ylim_max_var.get())

        except ValueError:
            messagebox.showerror(
                "入力エラー",
                "数値フィールドに無効な値が入力されています。確認してください。",
            )
            # 不正な値があった場合、グラフをクリアしてデフォルト状態で表示
            format_axis(self.ax, self.fig, {})
            self.canvas.draw()
            return

        if not self.vsm_data:
            format_axis(self.ax, self.fig, params)
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
            return

        output_stream = io.StringIO()
        with redirect_stdout(output_stream):
            print(f"解析開始: 膜厚={params['Thick']} nm, 面積={params['Area']} cm²\n")
            self._process_and_plot(params)

        self.log_message(output_stream.getvalue())
        self.log_message("\n描画完了。\n")

        self.canvas.draw()
        if self.all_metadata:
            self.info_button.config(state=tk.NORMAL)
        else:
            self.info_button.config(state=tk.DISABLED)

    def _process_and_plot(self, params):
        """データ処理とプロットのメインロジック"""
        Vol = params["Area"] * params["Thick"] * 1e-7

        # グローバルな軸範囲をリセット
        h_min_global, h_max_global = float("inf"), float("-inf")

        print("読み込みファイル:")
        for i, data in enumerate(self.vsm_data):
            print(f" {i + 1}: {data['path'].name}")

        for idx, data in enumerate(self.vsm_data):
            file = data["path"]
            df = data["df"]

            try:
                metadata = parse_metadata(file)
                self.all_metadata[file.name] = metadata

                H_full = df["H(Oe)"] * 1e-4
                M_full = df["M(emu)"] / Vol

                min_H_index = H_full.idxmin()
                remaining_H = H_full.iloc[min_H_index:]
                if remaining_H.empty:
                    raise ValueError("データ不完全。復路が見つからない。")
                max_H_index_2 = min_H_index + remaining_H.idxmax()

                df_loop = df.iloc[: max_H_index_2 + 1]
                H_raw = df_loop["H(Oe)"] * 1e-4
                M_raw = df_loop["M(emu)"] / Vol
                print(f"\n--- 解析: {file.stem} (データ点: {len(H_raw)}) ---")

                if self.demag_correction_var.get():
                    print("  反磁性補正: 有効")
                    if self.manual_slope_var.get():
                        print("    傾き計算: 手動設定モード")
                        try:
                            pos_range = (
                                float(self.pos_h_min_var.get()),
                                float(self.pos_h_max_var.get()),
                            )
                            neg_range = (
                                float(self.neg_h_min_var.get()),
                                float(self.neg_h_max_var.get()),
                            )
                            slope, r2_pos, r2_neg = find_demag_slope_manual(
                                H_raw, M_raw, pos_range, neg_range
                            )
                        except ValueError:
                            print("  エラー: 手動設定の磁場範囲が無効。")
                            slope, r2_pos, r2_neg = 0, 0, 0
                    else:
                        print("    傾き計算: 自動検出モード")
                        slope, r2_pos, r2_neg = find_demag_slope_auto(H_raw, M_raw)
                    print(
                        f"    補正傾き S: {slope:.6f}, R^2: [正 {r2_pos:.4f}], [負 {r2_neg:.4f}]"
                    )
                else:
                    print("  反磁性補正: 無効")
                    slope = 0

                M_corrected = M_raw - H_raw * slope

                if self.offset_correction_var.get():
                    print("  磁化オフセット補正: 有効")
                    H_np, M_np = H_raw.values, M_corrected.values
                    Ms_pos = np.mean(M_np[H_np > np.max(H_np) * 0.9])
                    Ms_neg = np.mean(M_np[H_np < np.min(H_np) * 0.9])
                    offset = (Ms_pos + Ms_neg) / 2
                    M_final = M_corrected - offset
                    print(f"    補正値: {offset:.4f} kA/m")
                else:
                    print("  磁化オフセット補正: 無効")
                    M_final = M_corrected

                min_H_index_loop = H_raw.idxmin()
                H_down, M_down = (
                    H_raw.iloc[: min_H_index_loop + 1].values,
                    M_final.iloc[: min_H_index_loop + 1].values,
                )
                H_up, M_up = (
                    H_raw.iloc[min_H_index_loop:].values,
                    M_final.iloc[min_H_index_loop:].values,
                )

                # プロット色を決定
                if idx < len(self.file_color_vars):
                    color = self.file_color_vars[idx].get()
                else:  # フォールバック
                    color = self.base_colors[idx % len(self.base_colors)]

                plot_kwargs = {
                    "marker": "o",
                    "markersize": params["marker_size"],
                    "linestyle": "-",
                    "linewidth": params["line_width"],
                }

                self.ax.plot(H_down, M_down, color=color, **plot_kwargs)
                self.ax.plot(H_up, M_up, color=color, label=file.stem, **plot_kwargs)

                # 自動スケール用の範囲を更新
                if not self.xlim_min_var.get() and not self.xlim_max_var.get():
                    h_min_global = min(h_min_global, H_raw.min())
                    h_max_global = max(h_max_global, H_raw.max())
                    params["xlim_min"], params["xlim_max"] = h_min_global, h_max_global

                calculate_saturation_magnetization(H_raw, M_final)
                calculate_remanence(H_down, M_down, H_up, M_up)
                calculate_coercivity(H_down, M_down, H_up, M_up)

            except Exception as e:
                print(f"\nエラー: ファイル '{file.name}' の処理中に問題発生: {e}")
                import traceback

                traceback.print_exc(file=sys.stdout)
                continue

        format_axis(self.ax, self.fig, params)
        if self.show_legend_var.get() and any(self.ax.get_legend_handles_labels()[1]):
            self.ax.legend(
                fontsize=params["legend_fontsize"],
                loc="best",
                facecolor="#3C3C3C",
                edgecolor="none",
                labelcolor="#EAEAEA",
            )
        self.fig.tight_layout()


if __name__ == "__main__":
    root = tk.Tk()
    app = VSMApp(root)
    root.mainloop()
