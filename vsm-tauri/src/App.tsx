import { useState, useCallback } from "react";
import "./App.css";
import Sidebar from "./components/Sidebar";
import Graph from "./components/Graph";
import ResultsTable from "./components/ResultsTable";
import { analyzeFile, AnalysisResult, AnalysisParams } from "./api/client";

export type FileEntry = {
  file:    File;
  result:  AnalysisResult | null;
  error:   string | null;
  loading: boolean;
};

export type UnitMode = "SI" | "CGS" | "Normalized";

const DEFAULT_PARAMS: AnalysisParams = {
  thickness:        50,
  area:             100,
  demagMode:       "auto",
  offsetCorrection: false,
};

function App() {
  const [entries,   setEntries]   = useState<FileEntry[]>([]);
  const [params,    setParams]    = useState<AnalysisParams>(DEFAULT_PARAMS);
  const [unitMode,  setUnitMode]  = useState<UnitMode>("SI");
  const [fieldUnit, setFieldUnit] = useState<"mT" | "Oe">("mT");

  // ファイルを解析してエントリを更新
  const runAnalysis = useCallback(
    async (files: File[], replace: boolean) => {
      const newEntries: FileEntry[] = files.map((f) => ({
        file: f, result: null, error: null, loading: true,
      }));
      setEntries((prev) => replace ? newEntries : [...prev, ...newEntries]);

      const results = await Promise.all(
        files.map((f) =>
          analyzeFile(f, params)
            .then((r) => ({ result: r, error: null }))
            .catch((e: Error) => ({ result: null, error: e.message }))
        )
      );

      setEntries((prev) => {
        const next = replace ? [...newEntries] : [...prev];
        const offset = replace ? 0 : next.length - files.length;
        files.forEach((f, i) => {
          next[offset + i] = {
            file: f,
            result: results[i].result,
            error:  results[i].error,
            loading: false,
          };
        });
        return next;
      });
    },
    [params]
  );

  // 設定変更時に全ファイルを再解析
  const updateParams = useCallback(
    (next: Partial<AnalysisParams>) => {
      setParams((prev) => {
        const merged = { ...prev, ...next };
        // 読み込み済みファイルがあれば再解析
        setEntries((cur) => {
          if (cur.length === 0) return cur;
          const files = cur.map((e) => e.file);
          const loading = cur.map((e) => ({ ...e, loading: true }));
          Promise.all(
            files.map((f) =>
              analyzeFile(f, merged)
                .then((r) => ({ result: r, error: null }))
                .catch((e: Error) => ({ result: null, error: e.message }))
            )
          ).then((results) => {
            setEntries(files.map((f, i) => ({
              file: f, result: results[i].result,
              error: results[i].error, loading: false,
            })));
          });
          return loading;
        });
        return merged;
      });
    },
    []
  );

  return (
    <div className="flex h-screen w-screen bg-zinc-950 text-zinc-100 overflow-hidden">
      <Sidebar
        entries={entries}
        params={params}
        unitMode={unitMode}
        onLoadFiles={(f) => runAnalysis(f, true)}
        onAddFiles={(f)  => runAnalysis(f, false)}
        onClearAll={() => setEntries([])}
        onParamsChange={updateParams}
        onUnitModeChange={setUnitMode}
      />
      <div className="flex flex-col flex-1 overflow-hidden">
        <header className="h-10 flex items-center px-4 border-b border-zinc-700 bg-zinc-900 shrink-0">
          <span className="text-sm font-semibold text-zinc-200">VSM Data Analyzer</span>
          <span className="ml-3 text-xs text-zinc-500">
            {entries.length === 0 ? "ファイル未読み込み" : `${entries.length} ファイル`}
          </span>
        </header>
        <Graph entries={entries} unitMode={unitMode} />
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
