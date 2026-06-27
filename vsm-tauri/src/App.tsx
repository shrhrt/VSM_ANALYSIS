import { useState, useCallback, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";
import Sidebar from "./components/Sidebar";
import Graph from "./components/Graph";
import ResultsTable from "./components/ResultsTable";
import { analyzeFile, AnalysisResult, AnalysisParams } from "./api/client";

export type FileEntry = {
  file:      File;
  result:    AnalysisResult | null;
  error:     string | null;
  loading:   boolean;
  color:     string;
  thickness?: number;   // per-file override (未設定ならグローバル値を使用)
  area?:      number;
};

export type UnitMode = "SI" | "CGS" | "Normalized";

export type GraphSettings = {
  // 表示
  showLegend:      boolean;
  showGrid:        boolean;
  showZeroLines:   boolean;
  // プロット
  lineWidth:       number;
  markerSize:      number;    // 0 = マーカーなし
  markerSymbol:    string;    // "circle" | "square" | "diamond" | "triangle-up" | "cross" | "x"
  // 凡例
  legendPosition:  "top-right" | "top-left" | "bottom-right" | "bottom-left";
  legendFontSize:  number;
  legendColumns:   number;
  // 軸ラベル
  xLabelOverride:  string;
  yLabelOverride:  string;
  axisLabelSize:   number;
  tickLabelSize:   number;
  // 目盛り書式 (d3 format: ".1f" / ".2g" / ".2e" / "" = auto)
  xTickFormat:     string;
  yTickFormat:     string;
  // 描画範囲 (空文字 = auto)
  xMin: string;
  xMax: string;
  yMin: string;
  yMax: string;
  // 原点線
  zeroLineColor:   string;
  zeroLineStyle:   string;    // "solid" | "dot" | "dash"
  // グリッド
  gridColor:       string;
  gridStyle:       string;    // "solid" | "dot" | "dash"
};

export const FILE_COLORS = [
  "#6366f1", "#f59e0b", "#10b981", "#ef4444",
  "#8b5cf6", "#06b6d4", "#f97316", "#84cc16",
];

const DEFAULT_PARAMS: AnalysisParams = {
  thickness:        50,
  area:             100,
  demagMode:       "auto",
  offsetCorrection: false,
  hsTolerance:      2.0,
  hsMinConsecutive: 3,
};

const DEFAULT_GRAPH: GraphSettings = {
  showLegend:     true,
  showGrid:       false,
  showZeroLines:  true,
  lineWidth:      1.5,
  markerSize:     0,
  markerSymbol:   "circle",
  legendPosition: "top-right",
  legendFontSize: 16,
  legendColumns:  1,
  xLabelOverride: "",
  yLabelOverride: "",
  axisLabelSize:  24,
  tickLabelSize:  16,
  xTickFormat:    ".1f",
  yTickFormat:    ".0f",
  xMin: "-1.0",
  xMax:  "1.0",
  yMin: "",
  yMax: "",
  zeroLineColor: "grey",
  zeroLineStyle: "dot",
  gridColor:     "#CCCCCC",
  gridStyle:     "dot",
};

// エントリのファイル別設定を考慮した実効パラメータを生成
function effectiveParams(entry: FileEntry, global: AnalysisParams): AnalysisParams {
  return {
    ...global,
    thickness: entry.thickness ?? global.thickness,
    area:      entry.area      ?? global.area,
  };
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
    const poll = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8000/health");
        if (res.ok) { setBackendStatus("ready"); clearInterval(poll); }
      } catch {}
    }, 500);
    setTimeout(() => {
      clearInterval(poll);
      setBackendStatus((s) => s === "ready" ? "ready" : "error");
    }, 15000);
  }, []);

  // 新規ファイル読み込み
  const runAnalysis = useCallback(
    async (files: File[], replace: boolean) => {
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
        files.forEach((f, i) => {
          next[offset + i] = { ...next[offset + i], file: f, result: results[i].result, error: results[i].error, loading: false };
        });
        return next;
      });
    },
    [params]
  );

  // グローバルパラメータ変更 → 全ファイル再解析
  const updateParams = useCallback((next: Partial<AnalysisParams>) => {
    setParams((prev) => {
      const merged = { ...prev, ...next };
      setEntries((cur) => {
        if (cur.length === 0) return cur;
        const loading = cur.map((e) => ({ ...e, loading: true }));
        Promise.all(
          cur.map((e) =>
            analyzeFile(e.file, effectiveParams(e, merged))
              .then((r) => ({ result: r, error: null }))
              .catch((e: Error) => ({ result: null, error: e.message }))
          )
        ).then((results) => {
          setEntries((cur2) =>
            cur2.map((e, i) => ({ ...e, result: results[i].result, error: results[i].error, loading: false }))
          );
        });
        return loading;
      });
      return merged;
    });
  }, []);

  // ファイル別設定変更 → そのファイルだけ再解析
  const updateEntrySettings = useCallback((index: number, patch: { thickness?: number; area?: number }) => {
    setEntries((prev) => {
      const next = prev.map((e, i) => i === index ? { ...e, ...patch, loading: true } : e);
      const entry = next[index];
      analyzeFile(entry.file, effectiveParams(entry, params))
        .then((r) => setEntries((cur) => cur.map((e, i) => i === index ? { ...e, result: r, error: null, loading: false } : e)))
        .catch((err: Error) => setEntries((cur) => cur.map((e, i) => i === index ? { ...e, result: null, error: err.message, loading: false } : e)));
      return next;
    });
  }, [params]);

  const updateEntryColor  = useCallback((i: number, c: string) => setEntries((p) => p.map((e, j) => j === i ? { ...e, color: c } : e)), []);
  const removeEntry       = useCallback((i: number) => setEntries((p) => p.filter((_, j) => j !== i)), []);

  return (
    <div className="flex h-screen w-screen bg-zinc-950 text-zinc-100 overflow-hidden">
      <Sidebar
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
        onEntryColorChange={updateEntryColor}
        onEntryRemove={removeEntry}
        onEntrySettingsChange={updateEntrySettings}
      />
      <div className="flex flex-col flex-1 overflow-hidden">
        <header className="h-10 flex items-center px-4 border-b border-zinc-700 bg-zinc-900 shrink-0">
          <span className="text-sm font-semibold text-zinc-200">VSM Data Analyzer</span>
          <span className="ml-3 text-xs text-zinc-500">
            {entries.length === 0 ? "ファイル未読み込み" : `${entries.length} ファイル`}
          </span>
          <span className={`ml-auto text-xs px-2 py-0.5 rounded-full ${
            backendStatus === "ready" ? "bg-green-900 text-green-300" :
            backendStatus === "error" ? "bg-red-900 text-red-300" :
                                        "bg-zinc-800 text-zinc-400 animate-pulse"
          }`}>
            {backendStatus === "ready" ? "● API 接続中" :
             backendStatus === "error" ? "● API エラー" : "● API 起動中..."}
          </span>
        </header>
        <Graph entries={entries} unitMode={unitMode} graphSettings={graphSettings} />
        <ResultsTable
          entries={entries}
          fieldUnit={fieldUnit}
          onToggleUnit={() => setFieldUnit((u) => u === "mT" ? "Oe" : "mT")}
        />
      </div>
    </div>
  );
}

export default App;
