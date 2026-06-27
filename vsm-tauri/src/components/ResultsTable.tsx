import type { FileEntry } from "../App";

interface Props {
  entries: FileEntry[];
  fieldUnit: "mT" | "Oe";
  onToggleUnit: () => void;
}

function fmt(v: number | null | undefined, digits: number, scale = 1): string {
  if (v == null) return "N/A";
  return (v * scale).toFixed(digits);
}

const HEADERS = ["ファイル名", "Ms (kA/m)", "Mr (kA/m)", "Hc", "Hs", "角形比 (Mr/Ms)"];

export default function ResultsTable({ entries, fieldUnit, onToggleUnit }: Props) {
  const scale = fieldUnit === "mT" ? 0.1 : 1;

  const headers = HEADERS.map((h, i) =>
    i === 3 ? `Hc (${fieldUnit})` : i === 4 ? `Hs (${fieldUnit})` : h
  );

  const rows = entries.map((e) => [
    e.file.name,
    fmt(e.result?.Ms,  1),
    fmt(e.result?.Mr,  1),
    fmt(e.result?.Hc_Oe, 2, scale),
    fmt(e.result?.Hs_Oe, 2, scale),
    fmt(e.result?.squareness, 3),
  ]);

  const copyText = () => {
    const lines = [
      headers.join("\t"),
      ...rows.map((r) => r.join("\t")),
    ];
    navigator.clipboard.writeText(lines.join("\n"));
  };

  const copyTable = () => {
    const colWidths = headers.map((h, ci) =>
      Math.max(h.length, ...rows.map((r) => r[ci].length))
    );
    const pad = (s: string, w: number) => s.padEnd(w);
    const sep = colWidths.map((w) => "-".repeat(w)).join("-+-");
    const header = colWidths.map((w, i) => pad(headers[i], w)).join(" | ");
    const body = rows.map((r) =>
      colWidths.map((w, i) => pad(r[i], w)).join(" | ")
    );
    navigator.clipboard.writeText([header, sep, ...body].join("\n"));
  };

  return (
    <div className="h-44 bg-zinc-900 border-t border-zinc-700 flex flex-col shrink-0">
      {/* ヘッダー行 */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-700 shrink-0">
        <span className="text-xs font-semibold text-zinc-400 uppercase tracking-widest">解析結果</span>
        <div className="ml-auto flex gap-2">
          <button onClick={copyText}
            className="text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-200 px-3 py-1 rounded transition-colors">
            テキストコピー
          </button>
          <button onClick={copyTable}
            className="text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-200 px-3 py-1 rounded transition-colors">
            表コピー
          </button>
          <button onClick={onToggleUnit}
            className="text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-200 px-3 py-1 rounded transition-colors">
            Hc/Hs単位: {fieldUnit} → {fieldUnit === "mT" ? "Oe" : "mT"}
          </button>
        </div>
      </div>

      {/* テーブル */}
      <div className="overflow-auto flex-1">
        <table className="w-full text-sm text-zinc-300 border-collapse">
          <thead>
            <tr className="bg-zinc-800 sticky top-0">
              {headers.map((h) => (
                <th key={h} className={`px-4 py-2 text-xs font-medium text-zinc-400 whitespace-nowrap ${
                  h === "ファイル名" ? "text-left" : "text-right"
                }`}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-4 text-center text-xs text-zinc-600">
                  ファイルを読み込むと解析結果が表示されます
                </td>
              </tr>
            ) : entries.map((e, i) => (
              <tr key={i} className={i % 2 === 0 ? "bg-zinc-900" : "bg-zinc-800/50"}>
                <td className="px-4 py-1.5 text-left whitespace-nowrap">
                  {e.loading ? <span className="text-zinc-500">計算中...</span> : e.file.name}
                </td>
                <td className="px-4 py-1.5 text-right">{fmt(e.result?.Ms,  1)}</td>
                <td className="px-4 py-1.5 text-right">{fmt(e.result?.Mr,  1)}</td>
                <td className="px-4 py-1.5 text-right">{fmt(e.result?.Hc_Oe, 2, scale)}</td>
                <td className="px-4 py-1.5 text-right">{fmt(e.result?.Hs_Oe, 2, scale)}</td>
                <td className="px-4 py-1.5 text-right">{fmt(e.result?.squareness, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
