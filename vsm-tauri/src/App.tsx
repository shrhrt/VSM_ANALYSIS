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
import { analyzeFile, AnalysisResult, AnalysisParams, FileCalcSettings } from "./api/client";
import { downloadGraphImage, copyGraphToClipboard } from "./utils/graphExport";
import type { ExportOptions } from "./utils/graphExport";

export type FileEntry = {
  file:          File;
  result:        AnalysisResult | null;
  error:         string | null;
  loading:       boolean;
  color:         string;
  legendName?:   string;
  markerSymbol?: string;       // per-file マーカー形状
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
  "#6366f1", "#f59e0b", "#10b981", "#ef4444",
  "#8b5cf6", "#06b6d4", "#f97316", "#84cc16",
];

const DEFAULT_PARAMS: AnalysisParams = {
  thickness: 30, area: 90,
  demagMode: "auto", offsetCorrection: false,
  hsTolerance: 2.0, hsMinConsecutive: 3,
};

const DEFAULT_GRAPH: GraphSettings = {
  showLegend: true, showGrid: false, showZeroLines: true,
  lineWidth: 1.5, markerSize: 0, markerSymbol: "circle",
  legendPosition: "top-left", legendFontSize: 16, legendColumns: 1,
  xLabelOverride: "", yLabelOverride: "",
  axisLabelSize: 24, tickLabelSize: 16,
  xTickFormat: ".1f", yTickFormat: ".0f",
  xMin: "-1.0", xMax: "1.0", yMin: "", yMax: "",
  xDtick: "0.5", yDtick: "",
  zeroLineColor: "grey", zeroLineStyle: "dot",
  gridColor: "#CCCCCC", gridStyle: "dot",
  paperMode: false, paperColorScheme: "journal",
};

// ファイルをBase64文字列に変換
function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const buf = reader.result as ArrayBuffer;
      const bytes = new Uint8Array(buf);
      let binary = "";
      for (const b of bytes) binary += String.fromCharCode(b);
      resolve(btoa(binary));
    };
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
}

function App() {
  const [entries,       setEntries]       = useState<FileEntry[]>([]);
  const [params,        setParams]        = useState<AnalysisParams>(DEFAULT_PARAMS);
  const [unitMode,      setUnitMode]      = useState<UnitMode>("SI");
  const [fieldUnit,     setFieldUnit]     = useState<"mT" | "Oe">("mT");
  const [graphSettings, setGraphSettings] = useState<GraphSettings>(DEFAULT_GRAPH);
  const [backendStatus, setBackendStatus] = useState<"starting" | "ready" | "error">("starting");

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

  const runAnalysis = useCallback(async (files: File[], replace: boolean) => {
    setEntries((prev) => {
      const base = replace ? [] : prev;
      const added: FileEntry[] = files.map((f, i) => ({
        file: f, result: null, error: null, loading: true,
        color: FILE_COLORS[(base.length + i) % FILE_COLORS.length],
      }));
      return [...base, ...added];
    });

    const results = await Promise.all(
      files.map((f) =>
        analyzeFile(f, params)
          .then((r) => ({ result: r, error: null }))
          .catch((e: Error) => ({ result: null, error: e.message }))
      )
    );

    setEntries((prev) => {
      const next = [...prev];
      const offset = replace ? 0 : next.length - files.length;
      files.forEach((_f, i) => {
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
        const loading = cur.map((e) => ({ ...e, loading: true }));
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

  // セッション保存 (.vsm_session)
  const saveSession = useCallback(async (): Promise<boolean> => {
    if (entries.length === 0) return false;
    const entriesData = await Promise.all(entries.map(async (e) => ({
      filename:     e.file.name,
      color:        e.color,
      legendName:   e.legendName,
      markerSymbol: e.markerSymbol,
      calcSettings: e.calcSettings,
      fileData:     await toBase64(e.file),
    })));

    const json = JSON.stringify({
      version: 1,
      savedAt: new Date().toISOString(),
      params,
      graphSettings,
      unitMode,
      fieldUnit,
      entries: entriesData,
    }, null, 2);

    const defaultName = `session_${new Date().toISOString().slice(0, 10)}.vsm_session`;
    const filePath = await save({
      defaultPath: defaultName,
      filters: [{ name: "VSM Session", extensions: ["vsm_session"] }],
    });
    if (!filePath) return false; // キャンセル

    await writeTextFile(filePath, json);
    return true;
  }, [entries, params, graphSettings, unitMode, fieldUnit]);

  // セッション読み込み
  const loadSession = useCallback(async (file: File) => {
    const text    = await file.text();
    const session = JSON.parse(text);

    const restoredEntries: FileEntry[] = session.entries.map((e: {
      filename: string; color: string; legendName?: string; markerSymbol?: string;
      calcSettings?: FileCalcSettings; fileData: string;
    }, i: number) => {
      const binary = atob(e.fileData);
      const bytes  = Uint8Array.from(binary, (c) => c.charCodeAt(0));
      const vsFile = new File([bytes], e.filename);
      return {
        file: vsFile, result: null, error: null, loading: true,
        color:        e.color ?? FILE_COLORS[i % FILE_COLORS.length],
        legendName:   e.legendName,
        markerSymbol: e.markerSymbol,
        calcSettings: e.calcSettings,
      };
    });

    const restoredParams: AnalysisParams = session.params;
    setParams(restoredParams);
    setGraphSettings(session.graphSettings ?? DEFAULT_GRAPH);
    setUnitMode(session.unitMode ?? "SI");
    setFieldUnit(session.fieldUnit ?? "mT");
    setEntries(restoredEntries);

    const results = await Promise.all(
      restoredEntries.map((e) =>
        analyzeFile(e.file, restoredParams, e.calcSettings)
          .then((r) => ({ result: r, error: null }))
          .catch((err: Error) => ({ result: null, error: err.message }))
      )
    );

    setEntries((prev) => prev.map((e, i) => ({
      ...e,
      result:  results[i].result,
      error:   results[i].error,
      loading: false,
    })));
  }, []);

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
