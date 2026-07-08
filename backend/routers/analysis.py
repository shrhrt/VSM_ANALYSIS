import sys, io, contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from analysis import calculations as vsm_logic
from analysis import file_io
import tempfile, os
import json
import numpy as np

router = APIRouter(prefix="/api", tags=["analysis"])


def _r(v, ndigits):
    """None 安全な round。値が None ならそのまま None を返す。"""
    return round(float(v), ndigits) if v is not None else None


@router.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    # グローバル設定
    thickness: float = Form(50.0),
    area: float = Form(100.0),
    demag_mode: str = Form("auto"),          # "auto" | "none"
    offset_correction: bool = Form(False),
    hs_tolerance: float = Form(2.0),
    hs_min_consecutive: int = Form(3),
    # ファイル別反磁性補正
    per_demag_mode: str = Form(""),          # "" = グローバル設定を使用, "auto"|"manual"|"none"
    demag_pos_min: float = Form(0.5),
    demag_pos_max: float = Form(2.0),
    demag_neg_min: float = Form(-2.0),
    demag_neg_max: float = Form(-0.5),
    # ファイル別Ms計算範囲
    ms_manual: bool = Form(False),
    ms_pos_min: float = Form(0.5),
    ms_pos_max: float = Form(2.0),
    ms_neg_min: float = Form(-2.0),
    ms_neg_max: float = Form(-0.5),
    ms_link_ranges: bool = Form(True),
    # 除外点: 元データの行番号の JSON 配列（例: "[3, 17, 42]"）。該当点を全計算から除外する
    excluded_indices: str = Form(""),
):
    suffix = Path(file.filename or "data.VSM").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df, load_error = file_io.load_vsm_file(Path(tmp_path))
        if df is None:
            raise HTTPException(status_code=422, detail=load_error or "読み込み失敗")

        metadata = file_io.parse_metadata(Path(tmp_path))

        # ファイル別設定が指定されていればそちらを優先
        effective_demag_mode = per_demag_mode if per_demag_mode else demag_mode

        # 除外点の解析: JSON 配列 → 行番号の集合
        try:
            excluded_set = (
                set(int(i) for i in json.loads(excluded_indices))
                if excluded_indices.strip() else set()
            )
        except (ValueError, TypeError):
            excluded_set = set()

        log_buffer = io.StringIO()
        with contextlib.redirect_stdout(log_buffer):
            result = _run_analysis(
                df=df,
                thickness=thickness,
                area=area,
                demag_mode=effective_demag_mode,
                demag_pos_range=(demag_pos_min, demag_pos_max),
                demag_neg_range=(demag_neg_min, demag_neg_max),
                offset_correction=offset_correction,
                hs_tolerance=hs_tolerance,
                hs_min_consecutive=hs_min_consecutive,
                ms_manual=ms_manual,
                ms_pos_range=(ms_pos_min, ms_pos_max),
                ms_neg_range=(ms_neg_min, ms_neg_max),
                excluded_set=excluded_set,
            )

        result["filename"] = file.filename
        result["metadata"] = metadata
        result["logs"]     = [l for l in log_buffer.getvalue().splitlines() if l.strip()]
        return result

    finally:
        os.unlink(tmp_path)


def _run_analysis(
    df, thickness, area,
    demag_mode, demag_pos_range, demag_neg_range,
    offset_correction,
    hs_tolerance, hs_min_consecutive,
    ms_manual, ms_pos_range, ms_neg_range,
    excluded_set=None,
):
    vol_cm3 = area * 1e-2 * thickness * 1e-7
    if vol_cm3 <= 0:
        raise HTTPException(status_code=422, detail="膜厚・面積は正の値を入力してください")

    excluded_set = excluded_set or set()

    # 元データ全点（体積磁化へ換算）。元の行番号を保持したまま除外点をマスクする
    H_all = df["H(Oe)"].values * 1e-4
    M_all = df["M(emu)"].values / vol_cm3
    n_total = len(H_all)
    orig_idx_all = np.arange(n_total)

    keep = np.array([i not in excluded_set for i in range(n_total)], dtype=bool)
    H_raw = H_all[keep]
    M_raw = M_all[keep]
    orig_idx = orig_idx_all[keep]
    n_excluded = int((~keep).sum())

    print(f"--- 解析 (膜厚: {thickness} nm, 面積: {area} mm², 体積: {vol_cm3:.4e} cm³, "
          f"有効点: {len(H_raw)}, 除外点: {n_excluded}) ---")

    if len(H_raw) < 3:
        raise HTTPException(status_code=422, detail="有効なデータ点が不足しています（除外点が多すぎます）")

    # ループ抽出（元の行番号 orig_idx も同じスライスで持ち回る）
    min_idx = int(np.argmin(H_raw))
    if min_idx >= len(H_raw) - 1:
        min_idx = len(H_raw) // 2
    max_idx2 = min_idx + int(np.argmax(H_raw[min_idx:]))
    H_loop   = H_raw[: max_idx2 + 1]
    M_loop   = M_raw[: max_idx2 + 1]
    idx_loop = orig_idx[: max_idx2 + 1]

    # 反磁性補正
    slope = 0.0
    r2_pos = r2_neg = None
    if demag_mode == "auto":
        slope, r2_pos, r2_neg = vsm_logic.find_demag_slope_auto(H_loop, M_loop)
        print(f"  反磁性補正 (自動): 傾き={slope:.6f}, R²=[正 {r2_pos:.4f}, 負 {r2_neg:.4f}]")
    elif demag_mode == "manual":
        slope, r2_pos, r2_neg = vsm_logic.find_demag_slope_manual(
            H_loop, M_loop, demag_pos_range, demag_neg_range
        )
        print(f"  反磁性補正 (手動): 傾き={slope:.6f}, R²=[正 {r2_pos:.4f}, 負 {r2_neg:.4f}]")
        print(f"    正側範囲: {demag_pos_range[0]}～{demag_pos_range[1]} T")
        print(f"    負側範囲: {demag_neg_range[0]}～{demag_neg_range[1]} T")
    else:
        print("  反磁性補正: なし")

    M_corrected = M_loop - H_loop * slope

    # オフセット補正
    offset = 0.0
    if offset_correction:
        H_np = H_loop
        Ms_pos_o = float(np.mean(M_corrected[H_np > np.max(H_np) * 0.9])) \
            if np.any(H_np > np.max(H_np) * 0.9) else 0.0
        Ms_neg_o = float(np.mean(M_corrected[H_np < np.min(H_np) * 0.9])) \
            if np.any(H_np < np.min(H_np) * 0.9) else 0.0
        offset = (Ms_pos_o + Ms_neg_o) / 2
        M_corrected = M_corrected - offset
        print(f"  オフセット補正: {offset:.4f} kA/m")

    # 往路・復路分割
    split = int(np.argmin(H_loop))
    H_down, M_down = H_loop[: split + 1], M_corrected[: split + 1]
    H_up,   M_up   = H_loop[split:],       M_corrected[split:]
    idx_down = idx_loop[: split + 1]
    idx_up   = idx_loop[split:]

    # Ms計算（可視化用に使用した磁場範囲も記録）
    if ms_manual:
        print(f"  Ms計算 (手動): 正側 {ms_pos_range}, 負側 {ms_neg_range} T")
        Ms_result = vsm_logic.calculate_saturation_magnetization(
            H_loop, M_corrected,
            pos_range=ms_pos_range,
            neg_range=ms_neg_range,
        )
        ms_pos_win = [float(ms_pos_range[0]), float(ms_pos_range[1])]
        ms_neg_win = [float(ms_neg_range[0]), float(ms_neg_range[1])]
    else:
        Ms_result = vsm_logic.calculate_saturation_magnetization(H_loop, M_corrected)
        h_max, h_min = float(np.max(H_loop)), float(np.min(H_loop))
        ms_pos_win = [h_max * 0.9, h_max]
        ms_neg_win = [h_min, h_min * 0.9]
    Ms     = Ms_result.get("avg") if Ms_result else None
    Ms_pos = Ms_result.get("pos") if Ms_result else None
    Ms_neg = Ms_result.get("neg") if Ms_result else None

    Mr        = vsm_logic.calculate_remanence(H_down, M_down, H_up, M_up)
    Hc_result = vsm_logic.calculate_coercivity(H_down, M_down, H_up, M_up)
    Hc_T      = Hc_result.get("T")      if Hc_result else None
    Hc_Oe     = Hc_result.get("Oe")     if Hc_result else None
    Hc_down_T = Hc_result.get("down_T") if Hc_result else None
    Hc_up_T   = Hc_result.get("up_T")   if Hc_result else None
    Heb_T     = Hc_result.get("Heb_T")  if Hc_result else None
    Heb_Oe    = Hc_result.get("Heb_Oe") if Hc_result else None
    Hs_result = vsm_logic.calculate_saturation_field(
        H_down, M_down, H_up, M_up, Ms=Ms or 0,
        tolerance_pct=hs_tolerance, min_consecutive=hs_min_consecutive,
    ) if Ms else None
    Hs_Oe     = Hs_result.get("Oe")  if Hs_result else None
    Hs_pos_T  = Hs_result.get("pos") if Hs_result else None
    Hs_neg_T  = Hs_result.get("neg") if Hs_result else None
    squareness = (Mr / Ms) if (Mr is not None and Ms and Ms > 0) else None

    print(f"  Ms={Ms:.1f} kA/m" if Ms is not None else "  Ms=N/A")
    print(f"  Mr={Mr:.1f} kA/m" if Mr is not None else "  Mr=N/A")
    print(f"  Hc={Hc_T*1000:.2f} mT" if Hc_T is not None else "  Hc=N/A")
    print(f"  Heb={Heb_T*1000:.2f} mT" if Heb_T is not None else "  Heb=N/A")
    print(f"  Hs={Hs_Oe*0.1:.2f} mT" if Hs_Oe is not None else "  Hs=N/A")

    # 除外点（表示用: 補正後グラフ上の座標に同じ補正を適用）
    excl_mask = ~keep
    H_excl = H_all[excl_mask]
    M_excl = M_all[excl_mask] - H_excl * slope - offset
    idx_excl = orig_idx_all[excl_mask]

    return {
        "Ms":          _r(Ms,  3),
        "Mr":          _r(Mr,  3),
        "Hc_T":        _r(Hc_T,  6),
        "Hc_Oe":       _r(Hc_Oe, 2),
        "Hs_Oe":       _r(Hs_Oe, 2),
        "Heb_T":       _r(Heb_T,  6),
        "Heb_Oe":      _r(Heb_Oe, 2),
        "squareness":  _r(squareness, 4),
        "demag_slope": round(float(slope), 6),
        "plot": {
            "H_down": H_down.tolist(),
            "M_down": M_down.tolist(),
            "H_up":   H_up.tolist(),
            "M_up":   M_up.tolist(),
            "idx_down": idx_down.tolist(),
            "idx_up":   idx_up.tolist(),
        },
        "excluded": {
            "H":   H_excl.tolist(),
            "M":   M_excl.tolist(),
            "idx": idx_excl.tolist(),
        },
        # 解析注釈（可視化用。すべて内部単位 T / kA/m）
        "annot": {
            "hc_down_T":    _r(Hc_down_T, 6),
            "hc_up_T":      _r(Hc_up_T, 6),
            "mr":           _r(Mr, 3),
            "ms_pos":       _r(Ms_pos, 3),
            "ms_neg":       _r(Ms_neg, 3),
            "ms_pos_range": [round(ms_pos_win[0], 4), round(ms_pos_win[1], 4)],
            "ms_neg_range": [round(ms_neg_win[0], 4), round(ms_neg_win[1], 4)],
            "hs_pos_T":     _r(Hs_pos_T, 6),
            "hs_neg_T":     _r(Hs_neg_T, 6),
            "demag_r2_pos": _r(r2_pos, 4),
            "demag_r2_neg": _r(r2_neg, 4),
        },
    }
