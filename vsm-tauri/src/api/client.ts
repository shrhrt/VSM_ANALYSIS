import { open as tauriOpen } from "@tauri-apps/plugin-dialog";
import { readFile } from "@tauri-apps/plugin-fs";
import { fetch } from "@tauri-apps/plugin-http";

const BASE = "http://localhost:8000";

// ── 型定義 ─────────────────────────────────────────────────────

/** 解析注釈（可視化用。すべて内部単位 T / kA/m） */
export interface AnalysisAnnotations {
  hc_down_T:    number | null;   // 往路が M=0 を横切る符号付き磁場
  hc_up_T:      number | null;   // 復路が M=0 を横切る符号付き磁場
  mr:           number | null;   // 残留磁化の大きさ (kA/m)
  ms_pos:       number | null;   // 正側飽和磁化 (kA/m)
  ms_neg:       number | null;   // 負側飽和磁化の大きさ (kA/m)
  ms_pos_range: [number, number];// Ms 平均に使った正側の磁場範囲 (T)
  ms_neg_range: [number, number];// Ms 平均に使った負側の磁場範囲 (T)
  hs_pos_T:     number | null;
  hs_neg_T:     number | null;
  demag_r2_pos: number | null;   // 反磁性フィットの決定係数
  demag_r2_neg: number | null;
}

export interface AnalysisResult {
  filename:    string;
  Ms:          number | null;
  Mr:          number | null;
  Hc_T:        number | null;
  Hc_Oe:       number | null;
  Hs_Oe:       number | null;
  Heb_T:       number | null;   // 交換バイアス磁場（ループの水平シフト）
  Heb_Oe:      number | null;
  squareness:  number | null;
  demag_slope: number;
  logs:        string[];
  metadata:    Record<string, string>;
  // idx_down/idx_up は各プロット点の「元データ行番号」。除外点のクリック対応に使う
  plot: { H_down: number[]; M_down: number[]; H_up: number[]; M_up: number[];
          idx_down?: number[]; idx_up?: number[] };
  // 除外された点の座標（補正後）と元行番号。灰色×で表示する
  excluded?: { H: number[]; M: number[]; idx: number[] };
  annot?: AnalysisAnnotations;
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
  // 除外点: 元データの行番号リスト。全計算から除外する
  excludedIndices?: number[];
  // 反対称化: 往路/復路を原点対称に補正し、磁場に偶な成分（定数オフセット等）を除去
  antisymmetrize?:  boolean;
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
  form.append("excluded_indices",   JSON.stringify(s.excludedIndices ?? []));
  form.append("antisymmetrize",      String(s.antisymmetrize ?? false));

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
