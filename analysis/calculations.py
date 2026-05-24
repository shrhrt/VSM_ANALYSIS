# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import linregress
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
from typing import Any, Tuple, Optional, Dict

# -----------------------------------------------------------------------------
# 解析・処理関数
# -----------------------------------------------------------------------------


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


def find_demag_slope_manual(
    H_data: Any,
    M_data: Any,
    pos_range: Tuple[float, float],
    neg_range: Tuple[float, float],
) -> Tuple[float, float, float]:
    """
    指定された磁場範囲で線形フィッティングを行い、反磁性補正の傾きを計算します。

    Args:
        H_data (Any): 磁場(H)のデータ配列。
        M_data (Any): 磁化(M)のデータ配列。
        pos_range (Tuple[float, float]): 正磁場側のフィッティング範囲 (min, max)。
        neg_range (Tuple[float, float]): 負磁場側のフィッティング範囲 (min, max)。

    Returns:
        Tuple[float, float, float]: (計算された平均の傾き, 正磁場側のR^2, 負磁場側のR^2)。
    """
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


def calculate_remanence(
    H_down: Any, M_down: Any, H_up: Any, M_up: Any
) -> Optional[float]:
    """
    残留磁化(Mr)を往路・復路データから計算します (H=0のときのM)。

    Args:
        H_down (Any): 往路の磁場データ。
        M_down (Any): 往路の磁化データ。
        H_up (Any): 復路の磁場データ。
        M_up (Any): 復路の磁化データ。

    Returns:
        Optional[float]: 計算された残留磁化の平均値。計算に失敗した場合はNone。
    """
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


def calculate_coercivity(
    H_down: Any, M_down: Any, H_up: Any, M_up: Any
) -> Optional[Dict[str, float]]:
    """
    保磁力(Hc)を計算します (M=0のときのH)。

    Args:
        H_down (Any): 往路の磁場データ。
        M_down (Any): 往路の磁化データ。
        H_up (Any): 復路の磁場データ。
        M_up (Any): 復路の磁化データ。

    Returns:
        Optional[Dict[str, float]]: 'T'と'Oe'をキーとする保磁力の辞書。計算失敗時はNone。
    """
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


def calculate_saturation_magnetization(
    H: Any,
    M: Any,
    pos_range: Optional[Tuple[float, float]] = None,
    neg_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    飽和磁化(Ms)を計算します。手動範囲が指定されていればそれを使用します。

    Args:
        H (Any): 磁場データ。
        M (Any): 磁化データ。
        pos_range (Optional[Tuple[float, float]], optional): 正側の計算範囲 (min, max)。
        neg_range (Optional[Tuple[float, float]], optional): 負側の計算範囲 (min, max)。

    Returns:
        Dict[str, float]: 'avg', 'pos', 'neg' をキーとする飽和磁化の計算結果辞書。
    """
    H, M = np.array(H), np.array(M)
    Ms_pos, Ms_neg = 0, 0

    if pos_range and neg_range:
        # Manual range calculation
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
