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
  plot: {
    H_down: number[];
    M_down: number[];
    H_up:   number[];
    M_up:   number[];
  };
}

export interface AnalysisParams {
  thickness:         number;   // nm
  area:              number;   // mm²
  demagMode:        "auto" | "none";
  offsetCorrection:  boolean;
  hsTolerance:       number;   // %
  hsMinConsecutive:  number;
}

export async function analyzeFile(
  file: File,
  params: AnalysisParams,
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("thickness",         String(params.thickness));
  form.append("area",              String(params.area));
  form.append("demag_mode",          params.demagMode);
  form.append("offset_correction",   String(params.offsetCorrection));
  form.append("hs_tolerance",        String(params.hsTolerance));
  form.append("hs_min_consecutive",  String(params.hsMinConsecutive));

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
  } catch {
    return false;
  }
}
