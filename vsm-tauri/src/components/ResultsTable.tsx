import { useState } from "react";
import type { FileEntry } from "../App";

interface Props {
  entries:      FileEntry[];
  fieldUnit:    "mT" | "Oe";
  onToggleUnit: () => void;
  maxHeight?:   number;
}

function fmt(v: number | null | undefined, digits: number, scale = 1): string {
  if (v == null) return "—";
  return (v * scale).toFixed(digits);
}

type SortKey = "name" | "Ms" | "Mr" | "Hc" | "Hs" | "sq";
type SortDir = "asc" | "desc";

function getSortVal(e: FileEntry, key: SortKey): string | number | null {
  switch (key) {
    case "name": return (e.legendName || e.file.name).toLowerCase();
    case "Ms":   return e.result?.Ms        ?? null;
    case "Mr":   return e.result?.Mr        ?? null;
    case "Hc":   return e.result?.Hc_Oe    ?? null;
    case "Hs":   return e.result?.Hs_Oe    ?? null;
    case "sq":   return e.result?.squareness ?? null;
  }
}

export default function ResultsTable({ entries, fieldUnit, onToggleUnit, maxHeight }: Props) {
  const [sortKey,   setSortKey]   = useState<SortKey | null>(null);
  const [sortDir,   setSortDir]   = useState<SortDir>("asc");
  const [collapsed, setCollapsed] = useState(false);

  const scale = fieldUnit === "mT" ? 0.1 : 1;

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => d === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("asc"); }
  };

  const sortedEntries = sortKey
    ? [...entries].sort((a, b) => {
        const av = getSortVal(a, sortKey);
        const bv = getSortVal(b, sortKey);
        if (av === null && bv === null) return 0;
        if (av === null) return 1;
        if (bv === null) return -1;
        const cmp = av < bv ? -1 : av > bv ? 1 : 0;
        return sortDir === "asc" ? cmp : -cmp;
      })
    : entries;

  const SortMark = ({ col }: { col: SortKey }) => (
    <span className={`ml-0.5 text-[9px] ${sortKey === col ? "text-indigo-400" : "text-zinc-700"}`}>
      {sortKey === col ? (sortDir === "asc" ? "↑" : "↓") : "⇅"}
    </span>
  );

  const hdrBase = "px-2 py-1 text-[10px] font-medium whitespace-nowrap select-none cursor-pointer transition-colors";

  const copyText = () => {
    const rows = sortedEntries.map((e) => [
      e.legendName || e.file.name,
      fmt(e.result?.Ms,  1),
      fmt(e.result?.Mr,  1),
      fmt(e.result?.Hc_Oe, 2, scale),
      fmt(e.result?.Hs_Oe, 2, scale),
      fmt(e.result?.squareness, 3),
    ]);
    const hdrs = ["ファイル名", "Ms (kA/m)", "Mr (kA/m)", `Hc (${fieldUnit})`, `Hs (${fieldUnit})`, "Mr/Ms"];
    navigator.clipboard.writeText([hdrs.join("\t"), ...rows.map((r) => r.join("\t"))].join("\n"));
  };

  const copyTable = () => {
    const hdrs = ["ファイル名", "Ms (kA/m)", "Mr (kA/m)", `Hc (${fieldUnit})`, `Hs (${fieldUnit})`, "Mr/Ms"];
    const rows = sortedEntries.map((e) => [
      e.legendName || e.file.name,
      fmt(e.result?.Ms,  1),
      fmt(e.result?.Mr,  1),
      fmt(e.result?.Hc_Oe, 2, scale),
      fmt(e.result?.Hs_Oe, 2, scale),
      fmt(e.result?.squareness, 3),
    ]);
    const colWidths = hdrs.map((h, ci) => Math.max(h.length, ...rows.map((r) => r[ci].length)));
    const pad  = (s: string, w: number) => s.padEnd(w);
    const sep  = colWidths.map((w) => "-".repeat(w)).join("-+-");
    const head = colWidths.map((w, i) => pad(hdrs[i], w)).join(" | ");
    const body = rows.map((r) => colWidths.map((w, i) => pad(r[i], w)).join(" | "));
    navigator.clipboard.writeText([head, sep, ...body].join("\n"));
  };

  const copyHTML = async () => {
    const hdrs = ["ファイル名", "Ms (kA/m)", "Mr (kA/m)", `Hc (${fieldUnit})`, `Hs (${fieldUnit})`, "Mr/Ms"];
    const rows = sortedEntries.map((e) => [
      e.legendName || e.file.name,
      fmt(e.result?.Ms,  1),
      fmt(e.result?.Mr,  1),
      fmt(e.result?.Hc_Oe, 2, scale),
      fmt(e.result?.Hs_Oe, 2, scale),
      fmt(e.result?.squareness, 3),
    ]);
    const thBase = "border:1px solid #bbb;padding:6px 12px;background:#f0f0f0;font-weight:600;white-space:nowrap;";
    const tdBase = "border:1px solid #bbb;padding:4px 12px;white-space:nowrap;";
    const thead = `<tr>${hdrs.map((h, i) => `<th style="${thBase}text-align:${i === 0 ? "left" : "right"};">${h}</th>`).join("")}</tr>`;
    const tbody = rows.map((r, ri) =>
      `<tr style="background:${ri % 2 === 1 ? "#f9f9f9" : "#fff"};">${r.map((c, ci) => `<td style="${tdBase}text-align:${ci === 0 ? "left" : "right"};">${c}</td>`).join("")}</tr>`
    ).join("");
    const html = `<table style="border-collapse:collapse;font-family:sans-serif;font-size:13px;"><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
    try {
      await navigator.clipboard.write([new ClipboardItem({ "text/html": new Blob([html], { type: "text/html" }) })]);
    } catch {
      navigator.clipboard.writeText(html);
    }
  };

  return (
    <div className="bg-zinc-900 border-t border-zinc-800 flex flex-col shrink-0"
      style={collapsed ? {} : { maxHeight: maxHeight ?? 140 }}>
      {/* ヘッダー行 */}
      <div className="flex items-center gap-1.5 px-3 py-1 border-b border-zinc-800 shrink-0">
        {/* 最小化トグル */}
        <button
          onClick={() => setCollapsed((v) => !v)}
          className="text-zinc-500 hover:text-zinc-200 transition-colors text-[10px] flex items-center gap-1"
          title={collapsed ? "展開" : "最小化"}
        >
          <span className={`transition-transform duration-200 ${collapsed ? "rotate-180" : ""}`}>▼</span>
        </button>
        <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest">解析結果</span>
        {!collapsed && (
          <div className="ml-auto flex gap-1">
            <button onClick={copyText}
              className="text-[10px] bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 px-2 py-0.5 rounded transition-colors"
              title="タブ区切りテキストでコピー（Excel貼り付け用）">
              テキスト
            </button>
            <button onClick={copyTable}
              className="text-[10px] bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 px-2 py-0.5 rounded transition-colors"
              title="Markdownテーブル形式でコピー">
              Markdown
            </button>
            <button onClick={copyHTML}
              className="text-[10px] bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 px-2 py-0.5 rounded transition-colors"
              title="HTMLテーブルでコピー（Word/PowerPoint貼り付け用）">
              HTML
            </button>
            <button onClick={onToggleUnit}
              className="text-[10px] bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 px-2 py-0.5 rounded transition-colors font-mono">
              {fieldUnit} → {fieldUnit === "mT" ? "Oe" : "mT"}
            </button>
          </div>
        )}
      </div>

      {/* テーブル: collapsed 時は非表示、table-layout fixed で数値列を固定幅にしてファイル名が余白を全部使う */}
      <div className={`overflow-auto flex-1 min-h-0 ${collapsed ? "hidden" : ""}`}>
        <table className="w-full border-collapse" style={{ tableLayout: "fixed" }}>
          <colgroup>
            <col />                           {/* ファイル名: 残り全部 */}
            <col style={{ width: 88 }} />     {/* Ms */}
            <col style={{ width: 88 }} />     {/* Mr */}
            <col style={{ width: 76 }} />     {/* Hc */}
            <col style={{ width: 88 }} />     {/* Hs */}
            <col style={{ width: 64 }} />     {/* Mr/Ms */}
          </colgroup>
          <thead>
            <tr className="bg-zinc-800/80 sticky top-0">
              <th onClick={() => handleSort("name")}
                className={`${hdrBase} text-left ${sortKey === "name" ? "text-indigo-400" : "text-zinc-500 hover:text-zinc-300"}`}>
                ファイル名<SortMark col="name" />
              </th>
              {(["Ms", "Mr", "Hc", "Hs", "sq"] as SortKey[]).map((k, i) => {
                const labels = ["Ms (kA/m)", "Mr (kA/m)", `Hc (${fieldUnit})`, `Hs (${fieldUnit})`, "Mr/Ms"];
                return (
                  <th key={k} onClick={() => handleSort(k)}
                    className={`${hdrBase} text-right ${sortKey === k ? "text-indigo-400" : "text-zinc-500 hover:text-zinc-300"}`}>
                    {labels[i]}<SortMark col={k} />
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sortedEntries.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-3 py-3 text-center text-[10px] text-zinc-600">
                  ファイルを読み込むと解析結果が表示されます
                </td>
              </tr>
            ) : sortedEntries.map((e, i) => {
              const displayName = e.legendName || e.file.name;
              return (
                <tr key={i} className={`${i % 2 === 0 ? "bg-zinc-900" : "bg-zinc-800/30"} hover:bg-zinc-800/60 transition-colors`}>
                  {/* ファイル名: 余白を全部使い、あふれたら省略 */}
                  <td className="px-2 py-0.5 overflow-hidden">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: e.color }} />
                      {e.loading
                        ? <span className="text-[10px] text-zinc-500">計算中...</span>
                        : <span className="text-[10px] text-zinc-300 font-mono truncate" title={displayName}>
                            {displayName}
                          </span>
                      }
                    </div>
                  </td>
                  <td className="px-2 py-0.5 text-right text-[11px] font-mono text-zinc-300 tabular-nums">{fmt(e.result?.Ms,  1)}</td>
                  <td className="px-2 py-0.5 text-right text-[11px] font-mono text-zinc-300 tabular-nums">{fmt(e.result?.Mr,  1)}</td>
                  <td className="px-2 py-0.5 text-right text-[11px] font-mono text-zinc-300 tabular-nums">{fmt(e.result?.Hc_Oe, 2, scale)}</td>
                  <td className="px-2 py-0.5 text-right text-[11px] font-mono text-zinc-300 tabular-nums">{fmt(e.result?.Hs_Oe, 2, scale)}</td>
                  <td className="px-2 py-0.5 text-right text-[11px] font-mono text-zinc-300 tabular-nums">{fmt(e.result?.squareness, 3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
