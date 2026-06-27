import { useState } from "react";
import { open as tauriOpen } from "@tauri-apps/plugin-dialog";

interface MissingEntry {
  label:       string;   // filename or path shown to user
  locatedPath: string | null;
}

interface Props {
  missing:   string[];
  onConfirm: (located: Map<string, string>) => void;
  onCancel:  () => void;
}

export default function MissingFilesDialog({ missing, onConfirm, onCancel }: Props) {
  const [entries, setEntries] = useState<MissingEntry[]>(
    missing.map((f) => ({ label: f, locatedPath: null }))
  );

  const locate = async (i: number) => {
    const selected = await tauriOpen({
      multiple: false,
      filters: [{ name: "VSM Files", extensions: ["VSM", "vsm"] }],
    });
    if (!selected || Array.isArray(selected)) return;
    setEntries((prev) =>
      prev.map((e, j) => j === i ? { ...e, locatedPath: selected } : e)
    );
  };

  const foundCount = entries.filter((e) => e.locatedPath).length;

  const handleConfirm = () => {
    const located = new Map<string, string>();
    for (const e of entries) {
      if (e.locatedPath) located.set(e.label, e.locatedPath);
    }
    onConfirm(located);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-zinc-900 border border-zinc-700 rounded-lg shadow-2xl w-[500px] max-h-[70vh] flex flex-col">
        {/* ヘッダー */}
        <div className="px-5 py-4 border-b border-zinc-800">
          <h2 className="text-sm font-semibold text-zinc-100">ファイルが見つかりません</h2>
          <p className="text-xs text-zinc-500 mt-1">
            以下のファイルが見つかりませんでした。「探す」で場所を指定するか、スキップして他のファイルだけ読み込めます。
          </p>
        </div>

        {/* ファイルリスト */}
        <div className="flex-1 overflow-y-auto px-5 py-3 space-y-2">
          {entries.map((e, i) => (
            <div key={i} className="flex items-center gap-2 rounded bg-zinc-800 px-3 py-2">
              <span className={`text-[11px] w-4 text-center font-bold ${e.locatedPath ? "text-emerald-400" : "text-red-400"}`}>
                {e.locatedPath ? "✓" : "×"}
              </span>
              <div className="flex-1 min-w-0">
                <p className="text-[11px] text-zinc-200 truncate font-mono" title={e.label}>{e.label}</p>
                {e.locatedPath && (
                  <p className="text-[10px] text-zinc-500 truncate" title={e.locatedPath}>{e.locatedPath}</p>
                )}
              </div>
              <button
                onClick={() => locate(i)}
                className="shrink-0 text-[11px] px-2.5 py-0.5 bg-indigo-700 hover:bg-indigo-600 text-indigo-100 rounded transition-colors"
              >
                探す
              </button>
            </div>
          ))}
        </div>

        {/* フッター */}
        <div className="px-5 py-3.5 border-t border-zinc-800 flex items-center justify-between gap-3">
          <span className="text-[11px] text-zinc-500">
            {foundCount}/{entries.length} ファイルが特定されました
          </span>
          <div className="flex gap-2">
            <button
              onClick={onCancel}
              className="text-sm px-3 py-1.5 bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded transition-colors"
            >
              キャンセル
            </button>
            <button
              onClick={handleConfirm}
              disabled={foundCount === 0 && entries.length > 0}
              className="text-sm px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded transition-colors"
            >
              {foundCount === entries.length
                ? "読み込む"
                : `スキップして読み込む (${entries.length - foundCount}件を除く)`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
