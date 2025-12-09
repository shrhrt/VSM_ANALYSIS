# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import linregress
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt

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

    print(
        f"  警告: ヘッダー行を自動検出できず。デフォルト値({default_row + 1}行目)を使用。"
    )
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
                        if i > 40:
                            break
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
                                        if key and value:
                                            metadata[key] = value
                        except IndexError:
                            continue
                if metadata:
                    return metadata
            except (UnicodeDecodeError, IOError):
                continue
    except Exception as e:
        print(f"  警告: メタデータ読み取り中に予期せぬエラー発生: {e}。")
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
        res_pos = linregress(df.tail(segment_size)["H"], df.tail(segment_size)["M"])
        slope_pos, r2_pos = res_pos.slope, res_pos.rvalue**2
        if r2_pos < min_r_squared:
            print(f"  警告: 正磁場側の線形性が低い (R^2 = {r2_pos:.4f})。")
    except (ValueError, np.linalg.LinAlgError):
        print("  警告: 正磁場側のフィッティング失敗。")

    slope_neg, r2_neg = 0, 0
    try:
        res_neg = linregress(df.head(segment_size)["H"], df.head(segment_size)["M"])
        slope_neg, r2_neg = res_neg.slope, res_neg.rvalue**2
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
        num_pos = pos_mask.sum()
        print(
            f"  手動(正): H=[{pos_range[0]:.2f}, {pos_range[1]:.2f}] T, 点数: {num_pos}"
        )
        if num_pos >= 2:
            if num_pos < 5:
                print(f"  警告: 正磁場範囲のデータ点数僅少({num_pos}点)。")
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
        print(
            f"  手動(負): H=[{neg_range[0]:.2f}, {neg_range[1]:.2f}] T, 点数: {num_neg}"
        )
        if num_neg >= 2:
            if num_neg < 5:
                print(f"  警告: 負磁場範囲のデータ点数僅少({num_neg}点)。")
            res_neg = linregress(df.loc[neg_mask, "H"], df.loc[neg_mask, "M"])
            slope_neg, r2_neg = res_neg.slope, res_neg.rvalue**2
        else:
            print("  エラー: 負磁場範囲のデータ点が2点未満。")
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
        Mr_down, Mr_up = (
            np.interp(0, H_down[::-1], M_down[::-1]),
            np.interp(0, H_up, M_up),
        )
        Mr_avg = (abs(Mr_down) + abs(Mr_up)) / 2
        # print(f"  残留磁化 Mr: {Mr_avg:.3f} kA/m")
        return Mr_avg
    except Exception as e:
        print(f"  エラー: 残留磁化の計算失敗: {e}")
        return None


def calculate_coercivity(H_down, M_down, H_up, M_up):
    """保磁力(Hc)を計算"""
    try:
        Hc_down, Hc_up = (
            np.interp(0, M_down[::-1], H_down[::-1]),
            np.interp(0, M_up, H_up),
        )
        Hc_avg = (abs(Hc_down) + abs(Hc_up)) / 2
        # print(f"  保磁力 Hc: {Hc_avg * 10000:.2f} Oe ({Hc_avg:.4f} T)")
        return {"T": Hc_avg, "Oe": Hc_avg * 10000}
    except Exception as e:
        print(f"  エラー: 保磁力の計算失敗: {e}")
        return None


def calculate_saturation_magnetization(H, M, pos_range=None, neg_range=None):
    """飽和磁化(Ms)を計算。手動範囲が指定されていればそれを使用する。"""
    H, M = np.array(H), np.array(M)
    Ms_pos, Ms_neg = 0, 0

    if pos_range and neg_range:
        # Manual range calculation
        print("    Ms計算: 手動範囲を使用")
        pos_mask = (H >= pos_range[0]) & (H <= pos_range[1])
        neg_mask = (H >= neg_range[0]) & (H <= neg_range[1])

        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()
        print(
            f"    正磁場範囲 H=[{pos_range[0]:.2f}, {pos_range[1]:.2f}] T, 点数: {num_pos}"
        )
        if num_pos < 2:
            print(f"    警告: 正磁場範囲のデータ点数僅少({num_pos}点)。")

        print(
            f"    負磁場範囲 H=[{neg_range[0]:.2f}, {neg_range[1]:.2f}] T, 点数: {num_neg}"
        )
        if num_neg < 2:
            print(f"    警告: 負磁場範囲のデータ点数僅少({num_neg}点)。")

        if num_pos > 0:
            Ms_pos = np.mean(M[pos_mask])
        if num_neg > 0:
            Ms_neg = np.mean(np.abs(M[neg_mask]))

    else:
        # Automatic range calculation
        print("    Ms計算: 自動範囲を使用")
        H_max, H_min = np.max(H), np.min(H)
        pos_mask = H > H_max * 0.9
        neg_mask = H < H_min * 0.9
        if np.any(pos_mask):
            Ms_pos = np.mean(M[pos_mask])
        if np.any(neg_mask):
            Ms_neg = np.mean(np.abs(M[neg_mask]))

    if Ms_pos != 0 and Ms_neg != 0:
        Ms_avg = (Ms_pos + Ms_neg) / 2
    elif Ms_pos != 0:
        Ms_avg = Ms_pos
    else:
        Ms_avg = Ms_neg

    print(f"  飽和磁化 Ms: {Ms_avg:.3f} kA/m (正側: {Ms_pos:.3f}, 負側: {Ms_neg:.3f})")
    return {"avg": Ms_avg, "pos": Ms_pos, "neg": Ms_neg}


def format_axis(ax, fig, style_params, unit_mode="SI (T, kA/m)"):
    """グラフの軸や目盛りなどを整形"""
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    # 単位モードに応じて軸ラベルを設定
    if "CGS" in unit_mode:
        ax.set_xlabel(
            r"$H$ ($\mathrm{Oe}$)", fontsize=style_params.get("axis_label_fontsize", 16)
        )
        ax.set_ylabel(
            r"$M$ ($\mathrm{emu/cm^3}$)",
            fontsize=style_params.get("axis_label_fontsize", 16),
        )
    elif "Normalized" in unit_mode:
        ax.set_xlabel(
            r"$\mu_0H$ ($\mathrm{T}$)",
            fontsize=style_params.get("axis_label_fontsize", 16),
        )
        ax.set_ylabel(r"$M/M_s$", fontsize=style_params.get("axis_label_fontsize", 16))
    else:  # SI (T, kA/m)
        ax.set_xlabel(
            r"$\mu_0H$ ($\mathrm{T}$)",
            fontsize=style_params.get("axis_label_fontsize", 16),
        )
        ax.set_ylabel(
            r"$M$ ($\mathrm{kA/m}$)",
            fontsize=style_params.get("axis_label_fontsize", 16),
        )

    if style_params.get("show_zero_lines", True):
        ax.axhline(0, color="#AAAAAA", linestyle="-", linewidth=1.0)
        ax.axvline(0, color="#AAAAAA", linestyle="-", linewidth=1.0)
    if style_params.get("show_grid", True):
        ax.grid(
            True, 
            linestyle=style_params.get("grid_style", ":"), 
            color=style_params.get("grid_color", "#CCCCCC")
        )
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

    # --- 新しい軸設定 ---
    if style_params.get("xaxis_step"):
        try:
            step = float(style_params["xaxis_step"])
            if step > 0:
                ax.xaxis.set_major_locator(MultipleLocator(step))
        except (ValueError, TypeError):
            pass  # 無効な値は無視
    if style_params.get("yaxis_step"):
        try:
            step = float(style_params["yaxis_step"])
            if step > 0:
                ax.yaxis.set_major_locator(MultipleLocator(step))
        except (ValueError, TypeError):
            pass

    if style_params.get("xaxis_format"):
        try:
            ax.xaxis.set_major_formatter(FormatStrFormatter(style_params["xaxis_format"]))
        except (ValueError, TypeError):
             print(f"警告: 無効なX軸フォーマットです: {style_params['xaxis_format']}")
    if style_params.get("yaxis_format"):
        try:
            ax.yaxis.set_major_formatter(FormatStrFormatter(style_params["yaxis_format"]))
        except (ValueError, TypeError):
            print(f"警告: 無効なY軸フォーマットです: {style_params['yaxis_format']}")


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
        colors="black",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8,
        colors="black",
    )
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
