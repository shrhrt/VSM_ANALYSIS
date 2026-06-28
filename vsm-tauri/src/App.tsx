import { useState, useCallback, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { writeTextFile } from "@tauri-apps/plugin-fs";
import "./App.css";
import Sidebar from "./components/Sidebar";
import Graph from "./components/Graph";
import ResultsTable from "./components/ResultsTable";
import StatusBar from "./components/StatusBar";
import MenuBar from "./components/MenuBar";
import MissingFilesDialog from "./components/MissingFilesDialog";
import {
  analyzeFile, AnalysisResult, AnalysisParams, FileCalcSettings, FileWithPath,
  openSessionFilePicker, getSessionEnv, resolveSessionPaths, pathToFileWithPath,
} from "./api/client";
import { downloadGraphImage, copyGraphToClipboard } from "./utils/graphExport";
import { computeRelativePath, computeOnedrivePath } from "./utils/sessionPaths";
import type { ExportOptions } from "./utils/graphExport";

export type FileEntry = {
  file:          File;
  filePath:      string;        // Tauri ダイアログで開いた絶対パス（空文字 = 不明）
  result:        AnalysisResult | null;
  error:         string | null;
  loading:       boolean;
  color:         string;
  legendName?:   string;
  markerSymbol?: string;
  calcSettings?: FileCalcSettings;
};

export type UnitMode = "SI" | "CGS" | "Normalized";

export type PaperColorScheme = "current" | "journal" | "grayscale";

export type GraphSettings = {
  showLegend:     boolean;
  showGrid:       boolean;
  showZeroLines:  boolean;
  lineWidth:      number;
  markerSize:     number;
  markerSymbol:   string;
  legendPosition: "top-right" | "top-left" | "bottom-right" | "bottom-left";
  legendFontSize: number;
  legendColumns:  number;
  xLabelOverride: string;
  yLabelOverride: string;
  axisLabelSize:  number;
  tickLabelSize:  number;
  xTickFormat:    string;
  yTickFormat:    string;
  xMin: string; xMax: string;
  yMin: string; yMax: string;
  zeroLineColor:  string;
  zeroLineStyle:  string;
  gridColor:      string;
  gridStyle:      string;
  // 目盛り間隔 (空文字 = 自動)
  xDtick: string;
  yDtick: string;
  // 論文モード
  paperMode:        boolean;
  paperColorScheme: PaperColorScheme;
};

export const FILE_COLORS = [
  "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
  "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
];

// セッションファイルの内部型 (バージョン問わず共通)
type SessionEntryV1 = {
  filename: string; color: string; legendName?: string; markerSymbol?: string;
  calcSettings?: FileCalcSettings; fileData: string;
};
type SessionEntryV2 = {
  filename: string; color: string; legendName?: string; markerSymbol?: string;
  calcSettings?: FileCalcSettings;
  absolutePath: string; relativePath: string; onedrivePath: string;
};
type SessionData = {
  version?: number;
  params: AnalysisParams;
  graphSettings?: GraphSettings;
  unitMode?: string;
  fieldUnit?: string;
  entries: (SessionEntryV1 | SessionEntryV2)[];
};

const DEFAULT_PARAMS: AnalysisParams = {
  thickness: 30, area: 90,
  demagMode: "auto", offsetCorrection: false,
  hsTolerance: 2.0, hsMinConsecutive: 3,
};

const DEFAULT_GRAPH: GraphSettings = {
  showLegend: true, showGrid: false, showZeroLines: true,
  lineWidth: 1.5, markerSize: 5, markerSymbol: "circle",
  legendPosition: "top-left", legendFontSize: 16, legendColumns: 1,
  xLabelOverride: "", yLabelOverride: "",
  axisLabelSize: 24, tickLabelSize: 16,
  xTickFormat: ".1f", yTickFormat: ".0f",
  xMin: "-1.0", xMax: "1.0", yMin: "", yMax: "",
  xDtick: "0.5", yDtick: "",
  zeroLineColor: "grey", zeroLineStyle: "dot",
  gridColor: "#CCCCCC", gridStyle: "dot",
  paperMode: true, paperColorScheme: "current",
};

function App() {
  const [entries,       setEntries]       = useState<FileEntry[]>([]);
  const [params,        setParams]        = useState<AnalysisParams>(DEFAULT_PARAMS);
  const [unitMode,      setUnitMode]      = useState<UnitMode>("SI");
  const [fieldUnit,     setFieldUnit]     = useState<"mT" | "Oe">("mT");
  const [graphSettings, setGraphSettings] = useState<GraphSettings>(DEFAULT_GRAPH);
  const [backendStatus, setBackendStatus] = useState<"starting" | "ready" | "error">("starting");

  // 欠損ファイルダイアログ用の保留セッション
  const [missingDialog, setMissingDialog] = useState<{
    missing: string[];
    session: SessionData;
    sessionPath: string;
    foundMap: Map<string, string>;
  } | null>(null);

  useEffect(() => {
    invoke("start_backend").catch(() => {});
    let fast = true;
    const check = async () => {
      try {
        const res = await fetch("http://localhost:8000/health");
        if (res.ok) {
          setBackendStatus("ready");
          if (fast) { fast = false; clearInterval(poll); poll = setInterval(check, 5000); }
        } else throw new Error();
      } catch {
        if (!fast) setBackendStatus("error");
      }
    };
    let poll = setInterval(check, 500);
    setTimeout(() => {
      if (fast) { fast = false; clearInterval(poll); poll = setInterval(check, 5000); }
    }, 20000);
    return () => clearInterval(poll);
  }, []);

  const runAnalysis = useCallback(async (items: FileWithPath[], replace: boolean) => {
    setEntries((prev) => {
      const base = replace ? [] : prev;
      const added: FileEntry[] = items.map((item, i) => ({
        file: item.file, filePath: item.path,
        result: null, error: null, loading: true,
        color: FILE_COLORS[(base.length + i) % FILE_COLORS.length],
      }));
      return [...base, ...added];
    });

    const results = await Promise.all(
      items.map(({ file }) =>
        analyzeFile(file, params)
          .then((r) => ({ result: r, error: null }))
          .catch((e: Error) => ({ result: null, error: e.message }))
      )
    );

    setEntries((prev) => {
      const next = [...prev];
      const offset = replace ? 0 : next.length - items.length;
      items.forEach((_item, i) => {
        next[offset + i] = { ...next[offset + i], result: results[i].result, error: results[i].error, loading: false };
      });
      return next;
    });
  }, [params]);

  const updateParams = useCallback((next: Partial<AnalysisParams>) => {
    setParams((prev) => {
      const merged = { ...prev, ...next };
      setEntries((cur) => {
        if (cur.length === 0) return cur;
        const loading: FileEntry[] = cur.map((e) => ({ ...e, loading: true }));
        Promise.all(
          cur.map((e) =>
            analyzeFile(e.file, merged, e.calcSettings)
              .then((r) => ({ result: r, error: null }))
              .catch((e: Error) => ({ result: null, error: e.message }))
          )
        ).then((results) => {
          setEntries((c) => c.map((e, i) => ({ ...e, result: results[i].result, error: results[i].error, loading: false })));
        });
        return loading;
      });
      return merged;
    });
  }, []);

  const updateEntryCalcSettings = useCallback((index: number, patch: Partial<FileCalcSettings>) => {
    setEntries((prev) => {
      const next = prev.map((e, i) => {
        if (i !== index) return e;
        return { ...e, calcSettings: { ...e.calcSettings, ...patch }, loading: true };
      });
      const entry = next[index];
      analyzeFile(entry.file, params, entry.calcSettings)
        .then((r) => setEntries((c) => c.map((e, i) => i === index ? { ...e, result: r, error: null, loading: false } : e)))
        .catch((err: Error) => setEntries((c) => c.map((e, i) => i === index ? { ...e, result: null, error: err.message, loading: false } : e)));
      return next;
    });
  }, [params]);

  const updateEntryDisplay = useCallback((index: number, patch: Partial<Pick<FileEntry, "legendName" | "color" | "markerSymbol">>) => {
    setEntries((prev) => prev.map((e, i) => i === index ? { ...e, ...patch } : e));
  }, []);

  const removeEntry  = useCallback((i: number) => setEntries((p) => p.filter((_, j) => j !== i)), []);

  const moveEntry = useCallback((from: number, to: number) => {
    setEntries((prev) => {
      if (to < 0 || to >= prev.length) return prev;
      const next = [...prev];
      [next[from], next[to]] = [next[to], next[from]];
      return next;
    });
  }, []);

  // 1番目ファイルの calcSettings を全ファイルに適用
  const applyFirstToAll = useCallback(() => {
    setEntries((prev) => {
      if (prev.length < 2) return prev;
      const firstCalc = { ...prev[0].calcSettings };
      const updated = prev.map((e, i) =>
        i === 0 ? e : { ...e, calcSettings: firstCalc, loading: true }
      );
      updated.slice(1).forEach((e, i) => {
        const idx = i + 1;
        analyzeFile(e.file, params, e.calcSettings)
          .then((r) => setEntries((c) => c.map((ce, j) => j === idx ? { ...ce, result: r, error: null, loading: false } : ce)))
          .catch((err: Error) => setEntries((c) => c.map((ce, j) => j === idx ? { ...ce, result: null, error: err.message, loading: false } : ce)));
      });
      return updated;
    });
  }, [params]);

  // グラフ画像保存・クリップボードコピー
  const saveGraph = useCallback(async (opts: ExportOptions): Promise<boolean> => {
    return downloadGraphImage(opts);
  }, []);

  const copyGraph = useCallback(async (scale: number) => {
    await copyGraphToClipboard(scale);
  }, []);

  // ── セッション保存 (v2: パス参照方式) ─────────────────────────
  const saveSession = useCallback(async (): Promise<boolean> => {
    if (entries.length === 0) return false;

    const defaultName = `session_${new Date().toISOString().slice(0, 10)}.vsm_session`;
    const savePath = await save({
      defaultPath: defaultName,
      filters: [{ name: "VSM Session", extensions: ["vsm_session"] }],
    });
    if (!savePath) return false;

    let sessionEnv: { onedrive_commercial: string; onedrive: string } | null = null;
    try { sessionEnv = await getSessionEnv(); } catch { /* バックエンド未起動時は無視 */ }

    const entriesData = entries.map((e) => {
      const absPath = e.filePath;
      let relPath = "";
      let odPath  = "";
      if (absPath) {
        relPath = computeRelativePath(savePath, absPath);
        if (sessionEnv) {
          odPath = computeOnedrivePath(absPath, sessionEnv.onedrive_commercial)
                ?? computeOnedrivePath(absPath, sessionEnv.onedrive)
                ?? "";
        }
      }
      return {
        filename:     e.file.name,
        absolutePath: absPath,
        relativePath: relPath,
        onedrivePath: odPath,
        color:        e.color,
        legendName:   e.legendName,
        markerSymbol: e.markerSymbol,
        calcSettings: e.calcSettings,
      };
    });

    await writeTextFile(savePath, JSON.stringify({
      version: 2,
      savedAt: new Date().toISOString(),
      params, graphSettings, unitMode, fieldUnit,
      entries: entriesData,
    }, null, 2));
    return true;
  }, [entries, params, graphSettings, unitMode, fieldUnit]);

  // ── v2 セッション適用 (パスから File を読み込んで解析) ─────────
  const applySessionV2 = useCallback(async (
    session: SessionData,
    foundMap: Map<string, string>,
  ) => {
    const toLoad = (session.entries as SessionEntryV2[]).filter(
      (e) => foundMap.has(e.filename)
    );
    const restoredEntries: FileEntry[] = await Promise.all(
      toLoad.map(async (e, i) => {
        const { file } = await pathToFileWithPath(foundMap.get(e.filename)!);
        return {
          file, filePath: foundMap.get(e.filename)!,
          result: null, error: null, loading: true,
          color:        e.color ?? FILE_COLORS[i % FILE_COLORS.length],
          legendName:   e.legendName,
          markerSymbol: e.markerSymbol,
          calcSettings: e.calcSettings,
        };
      })
    );
    const rp: AnalysisParams = session.params;
    setParams(rp);
    setGraphSettings(session.graphSettings ?? DEFAULT_GRAPH);
    setUnitMode((session.unitMode as UnitMode) ?? "SI");
    setFieldUnit((session.fieldUnit as "mT" | "Oe") ?? "mT");
    setEntries(restoredEntries);
    const results = await Promise.all(
      restoredEntries.map((e) =>
        analyzeFile(e.file, rp, e.calcSettings)
          .then((r) => ({ result: r, error: null }))
          .catch((err: Error) => ({ result: null, error: err.message }))
      )
    );
    setEntries((prev) => prev.map((e, i) => ({
      ...e, result: results[i].result, error: results[i].error, loading: false,
    })));
  }, []);

  // ── v1 セッション適用 (後方互換: Base64 埋め込み) ──────────────
  const applySessionV1 = useCallback(async (session: SessionData) => {
    const restoredEntries: FileEntry[] = (session.entries as SessionEntryV1[]).map((e, i) => {
      const binary = atob(e.fileData ?? "");
      const bytes  = Uint8Array.from(binary, (c) => c.charCodeAt(0));
      return {
        file: new File([bytes], e.filename), filePath: "",
        result: null, error: null, loading: true,
        color:        e.color ?? FILE_COLORS[i % FILE_COLORS.length],
        legendName:   e.legendName,
        markerSymbol: e.markerSymbol,
        calcSettings: e.calcSettings,
      };
    });
    const rp: AnalysisParams = session.params;
    setParams(rp);
    setGraphSettings(session.graphSettings ?? DEFAULT_GRAPH);
    setUnitMode((session.unitMode as UnitMode) ?? "SI");
    setFieldUnit((session.fieldUnit as "mT" | "Oe") ?? "mT");
    setEntries(restoredEntries);
    const results = await Promise.all(
      restoredEntries.map((e) =>
        analyzeFile(e.file, rp, e.calcSettings)
          .then((r) => ({ result: r, error: null }))
          .catch((err: Error) => ({ result: null, error: err.message }))
      )
    );
    setEntries((prev) => prev.map((e, i) => ({
      ...e, result: results[i].result, error: results[i].error, loading: false,
    })));
  }, []);

  // ── セッション読み込みエントリーポイント ───────────────────────
  const loadSession = useCallback(async () => {
    const picked = await openSessionFilePicker();
    if (!picked) return;

    let session: SessionData;
    try { session = JSON.parse(await picked.file.text()); }
    catch { return; }

    // v1 (Base64) 後方互換
    if (!session.version || session.version < 2) {
      await applySessionV1(session);
      return;
    }

    // v2: パス解決
    const metas = (session.entries as SessionEntryV2[]).map((e) => ({
      filename:     e.filename     ?? "",
      absolutePath: e.absolutePath ?? "",
      relativePath: e.relativePath ?? "",
      onedrivePath: e.onedrivePath ?? "",
    }));
    const { resolved, missing } = await resolveSessionPaths(picked.path, metas);
    const foundMap = new Map(resolved.map((r) => [r.filename, r.resolved_path]));

    if (missing.length > 0) {
      setMissingDialog({ missing, session, sessionPath: picked.path, foundMap });
    } else {
      await applySessionV2(session, foundMap);
    }
  }, [applySessionV1, applySessionV2]);

  // 欠損ファイルダイアログ: 確定
  const handleMissingConfirm = useCallback(async (located: Map<string, string>) => {
    if (!missingDialog) return;
    const combined = new Map([...missingDialog.foundMap, ...located]);
    setMissingDialog(null);
    await applySessionV2(missingDialog.session, combined);
  }, [missingDialog, applySessionV2]);

  // ── サイドバーリサイズ ──────────────────────────────────────
  const SIDEBAR_DEFAULT = 288;
  const SIDEBAR_MIN     = 160;
  const SIDEBAR_MAX     = 900;
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT);
  const dragging  = useRef(false);
  const dragStartX = useRef(0);
  const dragStartW = useRef(0);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const delta = e.clientX - dragStartX.current;
      setSidebarWidth(Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, dragStartW.current + delta)));
    };
    const onUp = () => { dragging.current = false; document.body.style.cursor = ""; document.body.style.userSelect = ""; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);

  const handleDragStart = (e: React.MouseEvent) => {
    dragging.current   = true;
    dragStartX.current = e.clientX;
    dragStartW.current = sidebarWidth;
    document.body.style.cursor     = "col-resize";
    document.body.style.userSelect = "none";
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-zinc-950 text-zinc-100 overflow-hidden">
      {/* 欠損ファイルダイアログ */}
      {missingDialog && (
        <MissingFilesDialog
          missing={missingDialog.missing}
          onConfirm={handleMissingConfirm}
          onCancel={() => setMissingDialog(null)}
        />
      )}

      {/* メニューバー */}
      <MenuBar
        hasEntries={entries.length > 0}
        unitMode={unitMode}
        graphSettings={graphSettings}
        onOpenFiles={(f) => runAnalysis(f, true)}
        onAddFiles={(f)  => runAnalysis(f, false)}
        onClearAll={() => setEntries([])}
        onSaveSession={saveSession}
        onLoadSession={loadSession}
        onUnitMode={setUnitMode}
        onGraphSettings={(s) => setGraphSettings((p) => ({ ...p, ...s }))}
        onSaveGraph={saveGraph}
        onCopyGraph={copyGraph}
      />

      {/* メインコンテンツ */}
      <div className="flex flex-1 overflow-hidden">
        {/* サイドバー */}
        <Sidebar
          style={{ width: sidebarWidth, minWidth: sidebarWidth, maxWidth: sidebarWidth }}
          entries={entries}
          params={params}
          unitMode={unitMode}
          graphSettings={graphSettings}
          onLoadFiles={(f) => runAnalysis(f, true)}
          onAddFiles={(f)  => runAnalysis(f, false)}
          onClearAll={() => setEntries([])}
          onParamsChange={updateParams}
          onUnitModeChange={setUnitMode}
          onGraphSettingsChange={(s) => setGraphSettings((p) => ({ ...p, ...s }))}
          onEntryDisplayChange={updateEntryDisplay}
          onEntryCalcChange={updateEntryCalcSettings}
          onEntryRemove={removeEntry}
          onEntryMove={moveEntry}
          onApplyFirstToAll={applyFirstToAll}
          onSaveSession={saveSession}
          onLoadSession={loadSession}
          onSaveGraph={saveGraph}
          onCopyGraph={copyGraph}
        />

        {/* ドラッグハンドル */}
        <div
          className="w-1 shrink-0 cursor-col-resize bg-zinc-800 hover:bg-indigo-500/40 active:bg-indigo-500/70 transition-colors"
          onMouseDown={handleDragStart}
        />

        {/* グラフ・結果エリア */}
        <div className="flex flex-col flex-1 overflow-hidden min-w-0">
          <header className="h-10 flex items-center px-4 border-b border-zinc-700 bg-zinc-900 shrink-0">
            <span className="text-sm font-semibold text-zinc-200">VSM Data Analyzer</span>
            <div className="ml-auto flex items-center gap-2">
              {graphSettings.paperMode && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-900/60 text-indigo-300 border border-indigo-700/40">
                  論文モード
                </span>
              )}
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                backendStatus === "ready" ? "bg-green-900 text-green-300" :
                backendStatus === "error" ? "bg-red-900 text-red-300" :
                                            "bg-zinc-800 text-zinc-400 animate-pulse"
              }`}>
                {backendStatus === "ready" ? "● API 接続中" :
                 backendStatus === "error" ? "● API エラー" : "● API 起動中..."}
              </span>
            </div>
          </header>
          <Graph entries={entries} unitMode={unitMode} graphSettings={graphSettings} />
          <ResultsTable
            entries={entries}
            fieldUnit={fieldUnit}
            onToggleUnit={() => setFieldUnit((u) => u === "mT" ? "Oe" : "mT")}
          />
          <StatusBar entries={entries} backendStatus={backendStatus} />
        </div>
      </div>
    </div>
  );
}

export default App;
