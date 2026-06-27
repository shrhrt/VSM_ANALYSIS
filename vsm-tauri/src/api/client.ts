const BASE = "http://localhost:8000";

export interface AnalysisResult {
  filename:    string;
  Ms:          number | null;
  Mr:          number | null;
  Hc_T:        number | null;
  Hc_Oe:       number | null;
  Hs_Oe:       number | null;
  squareness:  number | null;
  demag_slope: number;
  logs:        string[];
  metadata:    Record<string, string>;
  plot: { H_down: number[]; M_down: number[]; H_up: number[]; M_up: number[] };
}

export interface AnalysisParams {
  thickness:        number;
  area:             number;
  demagMode:       "auto" | "none";
  offsetCorrection: boolean;
  hsTolerance:      number;
  hsMinConsecutive: number;
}

export interface FileCalcSettings {
  thickness?:    number;
  area?:         number;
  // ファイル別反磁性補正
  perDemagMode?: "auto" | "manual" | "none";
  demagPosMin?:  number;
  demagPosMax?:  number;
  demagNegMin?:  number;
  demagNegMax?:  number;
  // ファイル別Ms計算範囲
  msManual?:     boolean;
  msPosMin?:     number;
  msPosMax?:     number;
  msNegMin?:     number;
  msNegMax?:     number;
  msLinkRanges?: boolean;
}

export async function analyzeFile(
  file: File,
  params: AnalysisParams,
  fileSettings?: FileCalcSettings,
): Promise<AnalysisResult> {
  const s = fileSettings ?? {};
  const form = new FormData();
  form.append("file",              file);
  form.append("thickness",         String(s.thickness    ?? params.thickness));
  form.append("area",              String(s.area         ?? params.area));
  form.append("demag_mode",        params.demagMode);
  form.append("offset_correction", String(params.offsetCorrection));
  form.append("hs_tolerance",      String(params.hsTolerance));
  form.append("hs_min_consecutive", String(params.hsMinConsecutive));
  // ファイル別設定
  form.append("per_demag_mode",    s.perDemagMode  ?? "");
  form.append("demag_pos_min",     String(s.demagPosMin  ?? 0.5));
  form.append("demag_pos_max",     String(s.demagPosMax  ?? 2.0));
  form.append("demag_neg_min",     String(s.demagNegMin  ?? -2.0));
  form.append("demag_neg_max",     String(s.demagNegMax  ?? -0.5));
  form.append("ms_manual",         String(s.msManual     ?? false));
  form.append("ms_pos_min",        String(s.msPosMin     ?? 0.5));
  form.append("ms_pos_max",        String(s.msPosMax     ?? 2.0));
  form.append("ms_neg_min",        String(s.msNegMin     ?? -2.0));
  form.append("ms_neg_max",        String(s.msNegMax     ?? -0.5));
  form.append("ms_link_ranges",    String(s.msLinkRanges ?? true));

  const res = await fetch(`${BASE}/api/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch { return false; }
}
