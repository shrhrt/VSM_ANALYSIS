# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import linregress
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


def antisymmetrize_loop(
    H_down: Any, M_down: Any, H_up: Any, M_up: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M-H ループを反対称化し、原点対称 M_down(H) = -M_up(-H) を厳密に満たすよう補正します。

    磁場に対して偶な成分（定数オフセットや偶な背景などの測定アーティファクト）を除去します。
    強磁性ループ本体や反磁性の傾き（いずれも奇成分）は保存されます。

    式:
        M_down_as(H) = ( M_down(H) - M_up(-H) ) / 2
        M_up_as(H)   = ( M_up(H)   - M_down(-H) ) / 2
    反対の枝の -H での値は線形補間で求めます。

    注意: 原点対称を強制するため、交換バイアス（ループの水平シフト）も 0 になります。
          交換バイアス試料には適用しないでください。

    Args:
        H_down: 往路（降磁場）の磁場データ。
        M_down: 往路の磁化データ。
        H_up:   復路（昇磁場）の磁場データ。
        M_up:   復路の磁化データ。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 反対称化後の (M_down_as, M_up_as)。
    """
    H_down = np.asarray(H_down, dtype=float)
    M_down = np.asarray(M_down, dtype=float)
    H_up = np.asarray(H_up, dtype=float)
    M_up = np.asarray(M_up, dtype=float)

    # np.interp は x が昇順である必要があるため、各枝を H 昇順に並べ替えてから補間する
    du = np.argsort(H_up)
    dd = np.argsort(H_down)

    def M_up_at(x):
        return np.interp(x, H_up[du], M_up[du])

    def M_down_at(x):
        return np.interp(x, H_down[dd], M_down[dd])

    M_down_as = (M_down - M_up_at(-H_down)) / 2.0
    M_up_as = (M_up - M_down_at(-H_up)) / 2.0
    return M_down_as, M_up_as


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
        Optional[Dict[str, float]]: 保磁力とループ形状に関する辞書。計算失敗時はNone。
            'T', 'Oe'          : 保磁力 Hc = (|Hc_down| + |Hc_up|) / 2
            'down_T', 'up_T'   : 各ブランチが M=0 を横切る符号付き磁場 (T)
            'Heb_T', 'Heb_Oe'  : 交換バイアス磁場（ループの水平シフト）
                                 Heb = (Hc_down + Hc_up) / 2。対称ループなら 0。
    """
    try:
        Hc_down, Hc_up = (
            float(np.interp(0, M_down[::-1], H_down[::-1])),
            float(np.interp(0, M_up, H_up)),
        )
        Hc_avg = (abs(Hc_down) + abs(Hc_up)) / 2
        Heb = (Hc_down + Hc_up) / 2
        # print(f"  保磁力 Hc: {Hc_avg * 10000:.2f} Oe ({Hc_avg:.4f} T)")
        return {
            "T": Hc_avg,
            "Oe": Hc_avg * 10000,
            "down_T": Hc_down,
            "up_T": Hc_up,
            "Heb_T": Heb,
            "Heb_Oe": Heb * 10000,
        }
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


def calculate_saturation_field(
    H_down: Any,
    M_down: Any,
    H_up: Any,
    M_up: Any,
    Ms: float,
    tolerance_pct: float = 2.0,
    min_consecutive: int = 3,
) -> Optional[Dict[str, float]]:
    """
    飽和磁場(Hs)を計算します。

    正側(降磁場ブランチのH>0領域)と負側(昇磁場ブランチのH<0領域)で
    別々に算出し、平均を返します。

    アルゴリズム:
        |H|昇順でスキャンし、|M| < (1-ε)*Ms となる点が
        min_consecutive点以上連続する「最後のまとまり」の直後の点の|H|を Hs とする。
        連続性条件により高磁場側の孤立した外れ値を無視できる。

    Args:
        H_down: 往路の磁場データ（正Hmaxから負Hmaxへ）
        M_down: 往路の磁化データ
        H_up: 復路の磁場データ（負Hmaxから正Hmaxへ）
        M_up: 復路の磁化データ
        Ms: 飽和磁化 (kA/m)
        tolerance_pct: 許容誤差 (%)。Ms*(1-tolerance_pct/100) を閾値とする
        min_consecutive: 連続して閾値以下である必要な最小点数

    Returns:
        {'T', 'Oe', 'pos', 'neg'} をキーとする辞書。計算失敗時は None。
    """
    if Ms <= 0:
        return None

    threshold = (1.0 - tolerance_pct / 100.0) * Ms

    def _find_hs_branch(H_branch: np.ndarray, M_branch: np.ndarray) -> Optional[float]:
        abs_H = np.abs(H_branch)
        sort_idx = np.argsort(abs_H)
        abs_H_sorted = abs_H[sort_idx]
        abs_M_sorted = np.abs(M_branch[sort_idx])

        n = len(abs_H_sorted)
        if n < min_consecutive:
            return None

        below = abs_M_sorted < threshold

        # |H|昇順に走査し、連続 min_consecutive 点以上の below-threshold ランの
        # 最後のもの（最も高|H|側）の末端インデックスを記録する
        last_run_end = -1
        run_len = 0
        for i in range(n):
            if below[i]:
                run_len += 1
                if run_len >= min_consecutive:
                    last_run_end = i
            else:
                run_len = 0

        if last_run_end == -1:
            # 連続した below-threshold ランなし → 最低磁場から飽和とみなす
            return float(abs_H_sorted[0])

        next_idx = last_run_end + 1
        if next_idx >= n:
            # 測定範囲内で飽和に到達しなかった
            return None

        return float(abs_H_sorted[next_idx])

    try:
        H_down_np = np.asarray(H_down, dtype=float)
        M_down_np = np.asarray(M_down, dtype=float)
        H_up_np = np.asarray(H_up, dtype=float)
        M_up_np = np.asarray(M_up, dtype=float)

        # 正側: 降磁場ブランチの H > 0 領域
        pos_mask = H_down_np > 0
        Hs_pos: Optional[float] = None
        if np.sum(pos_mask) >= min_consecutive:
            Hs_pos = _find_hs_branch(H_down_np[pos_mask], M_down_np[pos_mask])

        # 負側: 昇磁場ブランチの H < 0 領域
        neg_mask = H_up_np < 0
        Hs_neg: Optional[float] = None
        if np.sum(neg_mask) >= min_consecutive:
            Hs_neg = _find_hs_branch(H_up_np[neg_mask], M_up_np[neg_mask])

        if Hs_pos is not None and Hs_neg is not None:
            Hs_avg = (Hs_pos + Hs_neg) / 2.0
        elif Hs_pos is not None:
            Hs_avg = Hs_pos
        elif Hs_neg is not None:
            Hs_avg = Hs_neg
        else:
            return None

        return {
            "T": Hs_avg,
            "Oe": Hs_avg * 10000.0,
            "pos": Hs_pos,
            "neg": Hs_neg,
        }

    except Exception as e:
        print(f"  エラー: 飽和磁場の計算失敗: {e}")
        return None
