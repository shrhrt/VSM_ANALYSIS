import sys, io, contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from analysis import calculations as vsm_logic
from analysis import file_io
import tempfile, os
import numpy as np

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    thickness: float = Form(50.0),
    area: float = Form(100.0),
    demag_mode: str = Form("auto"),
    offset_correction: bool = Form(False),
    hs_tolerance: float = Form(2.0),
    hs_min_consecutive: int = Form(3),
):
    suffix = Path(file.filename or "data.VSM").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df, load_error = file_io.load_vsm_file(Path(tmp_path))
        if df is None:
            raise HTTPException(status_code=422, detail=load_error or "読み込み失敗")

        # メタデータ取得
        metadata = file_io.parse_metadata(Path(tmp_path))

        # ログをキャプチャしながら解析実行
        log_buffer = io.StringIO()
        with contextlib.redirect_stdout(log_buffer):
            result = _run_analysis(
                df, thickness, area, demag_mode, offset_correction,
                hs_tolerance, hs_min_consecutive,
            )

        result["filename"] = file.filename
        result["metadata"] = metadata
        result["logs"]     = [l for l in log_buffer.getvalue().splitlines() if l.strip()]
        return result

    finally:
        os.unlink(tmp_path)


def _run_analysis(df, thickness, area, demag_mode, offset_correction,
                  hs_tolerance, hs_min_consecutive):
    vol_cm3 = area * 1e-2 * thickness * 1e-7
    if vol_cm3 <= 0:
        raise HTTPException(status_code=422, detail="膜厚・面積は正の値を入力してください")

    H_raw = df["H(Oe)"].values * 1e-4
    M_raw = df["M(emu)"].values / vol_cm3

    print(f"--- 解析 (膜厚: {thickness} nm, 面積: {area} mm², 体積: {vol_cm3:.4e} cm³, データ点: {len(H_raw)}) ---")

    # ループ抽出
    min_idx = int(np.argmin(H_raw))
    if min_idx >= len(H_raw) - 1:
        min_idx = len(H_raw) // 2
    max_idx2 = min_idx + int(np.argmax(H_raw[min_idx:]))
    H_loop = H_raw[: max_idx2 + 1]
    M_loop = M_raw[: max_idx2 + 1]

    # 反磁性補正
    slope = 0.0
    if demag_mode == "auto":
        slope, r2_pos, r2_neg = vsm_logic.find_demag_slope_auto(H_loop, M_loop)
        print(f"  反磁性補正 (自動): 傾き={slope:.6f}, R²=[正 {r2_pos:.4f}, 負 {r2_neg:.4f}]")
    else:
        print("  反磁性補正: なし")

    M_corrected = M_loop - H_loop * slope

    # オフセット補正
    if offset_correction:
        H_np = H_loop
        Ms_pos = float(np.mean(M_corrected[H_np > np.max(H_np) * 0.9])) \
            if np.any(H_np > np.max(H_np) * 0.9) else 0.0
        Ms_neg = float(np.mean(M_corrected[H_np < np.min(H_np) * 0.9])) \
            if np.any(H_np < np.min(H_np) * 0.9) else 0.0
        offset = (Ms_pos + Ms_neg) / 2
        M_corrected = M_corrected - offset
        print(f"  オフセット補正: {offset:.4f} kA/m")

    # 往路・復路分割
    split = int(np.argmin(H_loop))
    H_down, M_down = H_loop[: split + 1], M_corrected[: split + 1]
    H_up,   M_up   = H_loop[split:],       M_corrected[split:]

    # 物性値計算
    Ms_result  = vsm_logic.calculate_saturation_magnetization(H_loop, M_corrected)
    Ms         = Ms_result.get("avg") if Ms_result else None

    Mr         = vsm_logic.calculate_remanence(H_down, M_down, H_up, M_up)

    Hc_result  = vsm_logic.calculate_coercivity(H_down, M_down, H_up, M_up)
    Hc_T       = Hc_result.get("T")  if Hc_result else None
    Hc_Oe      = Hc_result.get("Oe") if Hc_result else None

    Hs_result  = vsm_logic.calculate_saturation_field(
        H_down, M_down, H_up, M_up, Ms=Ms or 0,
        tolerance_pct=hs_tolerance, min_consecutive=hs_min_consecutive,
    ) if Ms else None
    Hs_Oe      = Hs_result.get("Oe") if Hs_result else None

    squareness = (Mr / Ms) if (Mr is not None and Ms and Ms > 0) else None

    print(f"  Ms={Ms:.1f} kA/m" if Ms is not None else "  Ms=N/A")
    print(f"  Mr={Mr:.1f} kA/m" if Mr is not None else "  Mr=N/A")
    print(f"  Hc={Hc_T*1000:.2f} mT" if Hc_T is not None else "  Hc=N/A")
    print(f"  Hs={Hs_Oe*0.1:.2f} mT" if Hs_Oe is not None else "  Hs=N/A")

    return {
        "Ms":          round(float(Ms),  3) if Ms  is not None else None,
        "Mr":          round(float(Mr),  3) if Mr  is not None else None,
        "Hc_T":        round(float(Hc_T),  6) if Hc_T  is not None else None,
        "Hc_Oe":       round(float(Hc_Oe), 2) if Hc_Oe is not None else None,
        "Hs_Oe":       round(float(Hs_Oe), 2) if Hs_Oe is not None else None,
        "squareness":  round(float(squareness), 4) if squareness is not None else None,
        "demag_slope": round(float(slope), 6),
        "plot": {
            "H_down": H_down.tolist(),
            "M_down": M_down.tolist(),
            "H_up":   H_up.tolist(),
            "M_up":   M_up.tolist(),
        },
    }
