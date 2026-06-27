import { open as tauriOpen } from "@tauri-apps/plugin-dialog";
import { readFile } from "@tauri-apps/plugin-fs";

const BASE = "http://localhost:8000";

// ── 型定義 ─────────────────────────────────────────────────────

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
  perDemagMode?: "auto" | "manual" | "none";
  demagPosMin?:  number;
  demagPosMax?:  number;
  demagNegMin?:  number;
  demagNegMax?:  number;
  demagLinkRanges?: boolean;
  msManual?:        boolean;
  msPosMin?:        number;
  msPosMax?:        number;
  msNegMin?:        number;
  msNegMax?:        number;
  msLinkRanges?:    boolean;
}

/** ファイルパスと File オブジェクトのペア（Tauri ダイアログ経由で開いたファイル） */
export interface FileWithPath {
  file: File;
  path: string;
}

// ── ファイル操作 ────────────────────────────────────────────────

/** 絶対パスから File + path ペアを作成 */
export async function pathToFileWithPath(path: string): Promise<FileWithPath> {
  const bytes = await readFile(path);
  const name  = path.replace(/\\/g, "/").split("/").pop()!;
  return { file: new File([bytes], name), path };
}

/** Tauri ダイアログで VSM ファイルを選択して開く */
export async function openVSMFiles(multiple = true): Promise<FileWithPath[]> {
  const selected = await tauriOpen({
    multiple,
    filters: [{ name: "VSM Files", extensions: ["VSM", "vsm"] }],
  });
  if (!selected) return [];
  const paths = Array.isArray(selected) ? selected : [selected];
  return Promise.all(paths.map(pathToFileWithPath));
}

/** Tauri ダイアログでセッションファイルを選択して開く */
export async function openSessionFilePicker(): Promise<FileWithPath | null> {
  const selected = await tauriOpen({
    multiple: false,
    filters: [{ name: "VSM Session", extensions: ["vsm_session"] }],
  });
  if (!selected || Array.isArray(selected)) return null;
  return pathToFileWithPath(selected);
}

// ── 解析 API ───────────────────────────────────────────────────

export async function analyzeFile(
  file: File,
  params: AnalysisParams,
  fileSettings?: FileCalcSettings,
): Promise<AnalysisResult> {
  const s = fileSettings ?? {};
  const form = new FormData();
  form.append("file",               file);
  form.append("thickness",          String(s.thickness    ?? params.thickness));
  form.append("area",               String(s.area         ?? params.area));
  form.append("demag_mode",         params.demagMode);
  form.append("offset_correction",  String(params.offsetCorrection));
  form.append("hs_tolerance",       String(params.hsTolerance));
  form.append("hs_min_consecutive", String(params.hsMinConsecutive));
  form.append("per_demag_mode",     s.perDemagMode  ?? "");
  form.append("demag_pos_min",      String(s.demagPosMin  ?? 0.5));
  form.append("demag_pos_max",      String(s.demagPosMax  ?? 2.0));
  form.append("demag_neg_min",      String(s.demagNegMin  ?? -2.0));
  form.append("demag_neg_max",      String(s.demagNegMax  ?? -0.5));
  form.append("ms_manual",          String(s.msManual     ?? false));
  form.append("ms_pos_min",         String(s.msPosMin     ?? 0.5));
  form.append("ms_pos_max",         String(s.msPosMax     ?? 2.0));
  form.append("ms_neg_min",         String(s.msNegMin     ?? -2.0));
  form.append("ms_neg_max",         String(s.msNegMax     ?? -0.5));
  form.append("ms_link_ranges",     String(s.msLinkRanges ?? true));

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

// ── セッション パス解決 API ────────────────────────────────────

export interface SessionEnv {
  onedrive_commercial: string;
  onedrive:            string;
}

export interface SessionEntryMeta {
  filename:     string;
  absolutePath: string;
  relativePath: string;
  onedrivePath: string;
}

export interface ResolveResponse {
  resolved: { filename: string; resolved_path: string }[];
  missing:  string[];
}

export async function getSessionEnv(): Promise<SessionEnv> {
  const res = await fetch(`${BASE}/api/session/env`);
  if (!res.ok) throw new Error("session env fetch failed");
  return res.json();
}

export async function resolveSessionPaths(
  sessionPath: string,
  entries: SessionEntryMeta[],
): Promise<ResolveResponse> {
  const res = await fetch(`${BASE}/api/session/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_path: sessionPath, entries }),
  });
  if (!res.ok) throw new Error(`session resolve failed: HTTP ${res.status}`);
  return res.json();
}
