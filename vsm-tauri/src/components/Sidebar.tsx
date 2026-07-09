import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { FileEntry, UnitMode, GraphSettings, PaperColorScheme } from "../App";
import type { AnalysisParams, FileCalcSettings, FileWithPath } from "../api/client";
import { openVSMFiles } from "../api/client";
import { searchSuggestions, getCurrentToken } from "../utils/legendSuggestions";
import type { Suggestion } from "../utils/legendSuggestions";
import ExportDialog from "./ExportDialog";

// ── Props ──────────────────────────
interface Props {
  style?:                React.CSSProperties;
  entries:               FileEntry[];
  params:                AnalysisParams;
  unitMode:              UnitMode;
  graphSettings:         GraphSettings;
  onLoadFiles:           (files: FileWithPath[]) => void;
  onAddFiles:            (files: FileWithPath[]) => void;
  onClearAll:            () => void;
  onParamsChange:        (next: Partial<AnalysisParams>) => void;
  onUnitModeChange:      (mode: UnitMode) => void;
  onGraphSettingsChange: (next: Partial<GraphSettings>) => void;
  onEntryDisplayChange:  (index: number, patch: Partial<Pick<FileEntry, "legendName" | "color" | "markerSymbol" | "showAnnot">>) => void;
  onEntryCalcChange:     (index: number, patch: Partial<FileCalcSettings>) => void;
  onEntryRemove:         (index: number) => void;
  onEntryMove:           (from: number, to: number) => void;
  onApplyFirstToAll:     () => void;
  onSaveSession:         () => Promise<boolean>;
  onLoadSession:         () => void;
}

type Tab = "analysis" | "graph" | "save" | "log";

// ── 凡例名オートコンプリート入力 ────────────────
function LegendInput({ value, placeholder, onChange }: {
  value: string; placeholder: string; onChange: (v: string) => void;
}) {
  const inputRef  = useRef<HTMLInputElement>(null);
  const [sugs,     setSugs]     = useState<Suggestion[]>([]);
  const [selIdx,   setSelIdx]   = useState(-1);
  const [tokStart, setTokStart] = useState(0);
  const [dropPos,  setDropPos]  = useState<{ top: number; left: number; width: number } | null>(null);

  const showDropdown = (matches: Suggestion[]) => {
    if (matches.length === 0) { setSugs([]); setDropPos(null); return; }
    const rect = inputRef.current?.getBoundingClientRect();
    if (rect) setDropPos({ top: rect.bottom + 2, left: rect.left, width: rect.width });
    setSugs(matches);
  };

  const updateSugs = (val: string, cursor: number) => {
    const { token, start } = getCurrentToken(val, cursor);
    setTokStart(start);
    setSelIdx(-1);
    showDropdown(searchSuggestions(token));
  };

  const accept = (s: Suggestion) => {
    const cursor = inputRef.current?.selectionStart ?? value.length;
    const newVal = value.slice(0, tokStart) + s.insert + value.slice(cursor);
    onChange(newVal);
    setSugs([]);
    setDropPos(null);
    setTimeout(() => {
      const pos = tokStart + s.insert.length;
      inputRef.current?.setSelectionRange(pos, pos);
      inputRef.current?.focus();
    }, 0);
  };

  const dropdown = sugs.length > 0 && dropPos && createPortal(
    <div
      style={{ position: "fixed", top: dropPos.top, left: dropPos.left, width: dropPos.width, zIndex: 9999 }}
      className="bg-zinc-800 border border-zinc-600 rounded shadow-2xl overflow-hidden"
    >
      {sugs.map((s, i) => (
        <button
          key={i}
          onMouseDown={(e) => { e.preventDefault(); accept(s); }}
          className={`w-full text-left flex items-center gap-2 px-2.5 py-1.5 text-xs transition-colors ${
            i === selIdx ? "bg-indigo-700/60 text-white" : "text-zinc-300 hover:bg-zinc-700/60"
          }`}
        >
          <span className="font-mono font-semibold text-zinc-100 shrink-0">{s.insert}</span>
          <span className="text-zinc-500 text-[10px] truncate">{s.label}</span>
        </button>
      ))}
    </div>,
    document.body,
  );

  return (
    <div className="flex-1 min-w-0">
      <input
        ref={inputRef}
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => {
          onChange(e.target.value);
          updateSugs(e.target.value, e.target.selectionStart ?? e.target.value.length);
        }}
        onKeyDown={(e) => {
          if (sugs.length === 0) return;
          if (e.key === "ArrowDown")  { e.preventDefault(); setSelIdx((i) => Math.min(i + 1, sugs.length - 1)); }
          else if (e.key === "ArrowUp")   { e.preventDefault(); setSelIdx((i) => Math.max(i - 1, -1)); }
          else if ((e.key === "Enter" || e.key === "Tab") && selIdx >= 0) { e.preventDefault(); accept(sugs[selIdx]); }
          else if (e.key === "Escape") { setSugs([]); setDropPos(null); }
        }}
        onBlur={() => setTimeout(() => { setSugs([]); setDropPos(null); }, 150)}
        className="w-full bg-zinc-700/50 border border-zinc-600/50 text-zinc-200 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500 placeholder:text-zinc-600"
        title="凡例名（空欄=ファイル名）。TeX: $\alpha$ $\mu_0$ など"
      />
      {dropdown}
    </div>
  );
}

// ── 共通コンポーネント ──────────────────────────
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="p-4 border-b border-zinc-800">
      <h2 className="text-xs font-semibold text-zinc-500 uppercase tracking-widest mb-3">{title}</h2>
      {children}
    </section>
  );
}

function NumberInput({ label, value, step = 1, min, onChange }: {
  label: string; value: number; step?: number; min?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-3">
      <label className="text-xs text-zinc-400 block mb-1">{label}</label>
      <input type="number" value={value} step={step} min={min}
        onChange={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(v); }}
        className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500"
      />
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer mb-2">
      <div onClick={() => onChange(!checked)}
        className={`w-8 h-4 rounded-full relative transition-colors cursor-pointer ${checked ? "bg-indigo-500" : "bg-zinc-600"}`}>
        <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full shadow transition-transform ${checked ? "translate-x-4" : "translate-x-0.5"}`} />
      </div>
      {label}
    </label>
  );
}

function Select({ label, value, options, onChange }: {
  label: string; value: string; options: { value: string; label: string }[]; onChange: (v: string) => void;
}) {
  return (
    <div className="mb-3">
      <label className="text-xs text-zinc-400 block mb-1">{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500">
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

// 3択ボタングループ
function SegmentedControl({ value, options, onChange }: {
  value: string; options: { value: string; label: string }[]; onChange: (v: string) => void;
}) {
  return (
    <div className="flex rounded-md overflow-hidden border border-zinc-700 mb-3">
      {options.map((o, i) => (
        <button key={o.value} onClick={() => onChange(o.value)}
          className={`flex-1 text-xs py-1 transition-colors ${
            value === o.value ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200"
          } ${i > 0 ? "border-l border-zinc-700" : ""}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

// 範囲入力 (min ～ max)
function RangeInput({ label, min, max, onMinChange, onMaxChange, unit = "T" }: {
  label: string; min: string; max: string; unit?: string;
  onMinChange: (v: string) => void; onMaxChange: (v: string) => void;
}) {
  return (
    <div className="mb-2">
      <label className="text-xs text-zinc-500 block mb-1">{label} ({unit})</label>
      <div className="flex items-center gap-1">
        <input type="number" value={min} step={0.1}
          onChange={(e) => onMinChange(e.target.value)}
          className="w-full bg-zinc-700/50 border border-zinc-600 text-zinc-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500"
          placeholder="下限"
        />
        <span className="text-zinc-600 text-xs shrink-0">～</span>
        <input type="number" value={max} step={0.1}
          onChange={(e) => onMaxChange(e.target.value)}
          className="w-full bg-zinc-700/50 border border-zinc-600 text-zinc-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500"
          placeholder="上限"
        />
      </div>
    </div>
  );
}

// ── ファイルエントリ1件 ──────────────────────────
const MARKER_OPTIONS = [
  { value: "",           label: "グローバル設定" },
  { value: "circle",     label: "○ 丸" },
  { value: "square",     label: "□ 四角" },
  { value: "diamond",    label: "◇ ひし形" },
  { value: "triangle-up", label: "△ 三角" },
  { value: "cross",      label: "+ クロス" },
  { value: "x",          label: "× バツ" },
];

function FileEntryItem({ entry, index, total, params, onDisplayChange, onCalcChange, onRemove, onMove }: {
  entry:          FileEntry;
  index:          number;
  total:          number;
  params:         AnalysisParams;
  onDisplayChange: (patch: Partial<Pick<FileEntry, "legendName" | "color" | "markerSymbol" | "showAnnot">>) => void;
  onCalcChange:   (patch: Partial<FileCalcSettings>) => void;
  onRemove:       () => void;
  onMove:         (dir: -1 | 1) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const s = entry.calcSettings ?? {};

  const demagMode = s.perDemagMode ?? "";   // "" = グローバル設定
  const demagLink = s.demagLinkRanges ?? true;
  const msManual  = s.msManual ?? false;
  const msLink    = s.msLinkRanges ?? true;

  const stemName = entry.file.name.replace(/\.[^.]+$/, "");
  const displayName = entry.legendName ?? "";

  // 反磁性補正範囲連動: 正側が変わったら負側を自動ミラー
  const setDemagPosMin = (v: string) => {
    const n = parseFloat(v) || 0.5;
    onCalcChange({ demagPosMin: n, ...(demagLink ? { demagNegMax: -n } : {}) });
  };
  const setDemagPosMax = (v: string) => {
    const n = parseFloat(v) || 2.0;
    onCalcChange({ demagPosMax: n, ...(demagLink ? { demagNegMin: -n } : {}) });
  };

  // Ms範囲連動: 正側が変わったら負側も反転して更新
  const setMsPosMin = (v: string) => {
    onCalcChange({ msPosMin: parseFloat(v) || 0 });
    if (msLink) onCalcChange({ msNegMin: -(parseFloat(v) || 0) });
  };
  const setMsPosMax = (v: string) => {
    const n = parseFloat(v) || 0;
    onCalcChange({ msPosMax: n });
    if (msLink) onCalcChange({ msNegMax: -Math.abs(s.msPosMin ?? 0.5), msNegMin: -n });
  };

  return (
    <li className="rounded-lg bg-zinc-800/40 border border-zinc-700/50 overflow-hidden">
      {/* ── メイン行 ── */}
      <div className="flex items-center gap-1.5 px-2 py-2">
        {/* カラーピッカー */}
        <input type="color" value={entry.color}
          onChange={(e) => onDisplayChange({ color: e.target.value })}
          className="w-5 h-5 rounded cursor-pointer border-0 bg-transparent shrink-0"
          title="色を変更"
        />

        {/* 並び替えボタン */}
        <div className="flex flex-col gap-0.5 shrink-0">
          <button onClick={() => onMove(-1)} disabled={index === 0}
            className="w-4 h-3 flex items-center justify-center text-zinc-600 hover:text-zinc-300 disabled:opacity-20 disabled:cursor-not-allowed transition-colors"
            title="上へ">
            <svg viewBox="0 0 8 5" className="w-3 h-2 fill-current"><path d="M4 0L8 5H0z"/></svg>
          </button>
          <button onClick={() => onMove(1)} disabled={index === total - 1}
            className="w-4 h-3 flex items-center justify-center text-zinc-600 hover:text-zinc-300 disabled:opacity-20 disabled:cursor-not-allowed transition-colors"
            title="下へ">
            <svg viewBox="0 0 8 5" className="w-3 h-2 fill-current"><path d="M4 5L0 0H8z"/></svg>
          </button>
        </div>

        {/* 凡例名入力（オートコンプリート付き） */}
        <LegendInput
          value={displayName}
          placeholder={stemName}
          onChange={(v) => onDisplayChange({ legendName: v })}
        />

        {/* ステータスドット */}
        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
          entry.loading ? "bg-yellow-500 animate-pulse" :
          entry.error   ? "bg-red-500" : "bg-emerald-500"
        }`} title={entry.error ?? (entry.loading ? "解析中" : "完了")} />

        {/* 解析注釈トグル: このファイルの物性値(Ms/Hc/Mr/Hs/Heb)をグラフに注釈表示 */}
        <button onClick={() => onDisplayChange({ showAnnot: !entry.showAnnot })}
          className={`shrink-0 w-6 h-6 flex items-center justify-center rounded transition-colors ${
            entry.showAnnot
              ? "text-indigo-300 bg-indigo-700/50 ring-1 ring-inset ring-indigo-600/50"
              : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700"
          }`} title="このファイルの物性値(Ms/Hc/Mr/Hs)をグラフに注釈表示">
          <svg viewBox="0 0 16 16" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="8" cy="8" r="3.5" />
            <path d="M8 1v2M8 13v2M1 8h2M13 8h2" strokeLinecap="round" />
          </svg>
        </button>

        {/* 設定展開ボタン */}
        <button onClick={() => setExpanded((v) => !v)}
          className={`text-xs shrink-0 w-6 h-6 flex items-center justify-center rounded transition-colors ${
            expanded ? "text-indigo-400 bg-indigo-900/40" : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700"
          }`} title="ファイル別設定">
          <svg viewBox="0 0 16 16" className="w-3.5 h-3.5 fill-current">
            <path d="M8 10.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zm5.72-1.45a5.5 5.5 0 0 0 .05-.55 5.5 5.5 0 0 0-.05-.55l1.19-.93a.3.3 0 0 0 .07-.38l-1.13-1.96a.3.3 0 0 0-.36-.13l-1.4.56a5.37 5.37 0 0 0-.95-.55l-.21-1.49A.3.3 0 0 0 10.5 3h-2.25a.3.3 0 0 0-.3.25l-.21 1.5a5.37 5.37 0 0 0-.95.55l-1.4-.57a.3.3 0 0 0-.36.13L3.9 6.82a.29.29 0 0 0 .07.38l1.19.93a5.56 5.56 0 0 0-.05.55 5.56 5.56 0 0 0 .05.55L3.97 10.16a.29.29 0 0 0-.07.38l1.13 1.96c.07.13.23.18.36.13l1.4-.56c.3.2.62.38.95.55l.21 1.49a.3.3 0 0 0 .3.25h2.25a.3.3 0 0 0 .3-.25l.21-1.5c.33-.17.65-.34.95-.55l1.4.57c.13.05.29 0 .36-.13l1.13-1.96a.29.29 0 0 0-.07-.38l-1.19-.93z"/>
          </svg>
        </button>

        {/* 削除ボタン */}
        <button onClick={onRemove}
          className="text-zinc-600 hover:text-red-400 text-xs shrink-0 w-5 h-5 flex items-center justify-center rounded hover:bg-red-900/20 transition-colors"
          title="削除">✕
        </button>
      </div>

      {/* ── 展開パネル ── */}
      {expanded && (
        <div className="border-t border-zinc-700/50 px-3 pt-3 pb-3 space-y-4 bg-zinc-900/50">

          {/* スタイル */}
          <div>
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">スタイル</p>
            <label className="text-xs text-zinc-500 block mb-1">マーカー形状</label>
            <select value={entry.markerSymbol ?? ""}
              onChange={(e) => onDisplayChange({ markerSymbol: e.target.value || undefined })}
              className="w-full bg-zinc-700/50 border border-zinc-600 text-zinc-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500">
              {MARKER_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          {/* サンプル設定 */}
          <div>
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">サンプル</p>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-zinc-500 block mb-1">膜厚 (nm)</label>
                <input type="number" min={0.1} step={1}
                  placeholder={String(params.thickness)}
                  value={s.thickness ?? ""}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    onCalcChange({ thickness: isNaN(v) ? undefined : v });
                  }}
                  className="w-full bg-zinc-700/50 border border-zinc-600 text-zinc-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500 placeholder:text-zinc-600"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-500 block mb-1">面積 (mm²)</label>
                <input type="number" min={0.1} step={1}
                  placeholder={String(params.area)}
                  value={s.area ?? ""}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    onCalcChange({ area: isNaN(v) ? undefined : v });
                  }}
                  className="w-full bg-zinc-700/50 border border-zinc-600 text-zinc-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500 placeholder:text-zinc-600"
                />
              </div>
            </div>
          </div>

          {/* 除外点（グラフ上の点クリックで除外/復帰） */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-500"
              title="グラフ上のデータ点をクリックすると解析から除外できます（もう一度クリックで復帰）">
              除外点: {s.excludedIndices?.length ?? 0} 点
            </span>
            {(s.excludedIndices?.length ?? 0) > 0 && (
              <button onClick={() => onCalcChange({ excludedIndices: [] })}
                className="text-[10px] text-zinc-400 hover:text-zinc-100 bg-zinc-700/50 hover:bg-zinc-700 px-2 py-0.5 rounded transition-colors"
                title="このファイルの除外点をすべて解除">
                すべて復帰
              </button>
            )}
          </div>

          {/* 反磁性補正 */}
          <div>
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">反磁性補正</p>
            <SegmentedControl value={demagMode}
              options={[
                { value: "",       label: `GBL` },
                { value: "auto",   label: "自動" },
                { value: "manual", label: "手動" },
                { value: "none",   label: "なし" },
              ]}
              onChange={(v) => onCalcChange({ perDemagMode: v as FileCalcSettings["perDemagMode"] })}
            />
            {demagMode === "manual" && (
              <div className="space-y-1 pl-2 border-l-2 border-indigo-800/60">
                <RangeInput label="正側" min={String(s.demagPosMin ?? 0.5)} max={String(s.demagPosMax ?? 2.0)}
                  onMinChange={setDemagPosMin}
                  onMaxChange={setDemagPosMax}
                />
                {/* 連動トグル */}
                <button
                  onClick={() => onCalcChange({ demagLinkRanges: !demagLink })}
                  className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded transition-colors ${
                    demagLink
                      ? "text-indigo-400 bg-indigo-900/40 hover:bg-indigo-900/60"
                      : "text-zinc-500 bg-zinc-700/40 hover:bg-zinc-700/70"
                  }`}
                  title="正側の範囲を変えると負側を自動ミラー反転"
                >
                  <span>{demagLink ? "🔗" : "🔓"}</span>
                  <span>{demagLink ? "負側を連動" : "独立設定"}</span>
                </button>
                <RangeInput label="負側" min={String(s.demagNegMin ?? -2.0)} max={String(s.demagNegMax ?? -0.5)}
                  onMinChange={(v) => onCalcChange({ demagNegMin: parseFloat(v) || -2.0 })}
                  onMaxChange={(v) => onCalcChange({ demagNegMax: parseFloat(v) || -0.5 })}
                />
              </div>
            )}
          </div>

          {/* Ms計算範囲 */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Ms 計算範囲</p>
              <label className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer">
                <div onClick={() => onCalcChange({ msManual: !msManual })}
                  className={`w-7 h-3.5 rounded-full relative transition-colors cursor-pointer ${msManual ? "bg-indigo-500" : "bg-zinc-600"}`}>
                  <div className={`absolute top-0.5 w-2.5 h-2.5 bg-white rounded-full shadow transition-transform ${msManual ? "translate-x-3.5" : "translate-x-0.5"}`} />
                </div>
                手動
              </label>
            </div>
            {msManual ? (
              <div className="space-y-1 pl-2 border-l-2 border-emerald-800/60">
                <RangeInput label="正側" min={String(s.msPosMin ?? 0.5)} max={String(s.msPosMax ?? 2.0)}
                  onMinChange={setMsPosMin}
                  onMaxChange={setMsPosMax}
                />
                <RangeInput label="負側" min={String(s.msNegMin ?? -2.0)} max={String(s.msNegMax ?? -0.5)}
                  onMinChange={(v) => onCalcChange({ msNegMin: parseFloat(v) || -2.0 })}
                  onMaxChange={(v) => onCalcChange({ msNegMax: parseFloat(v) || -0.5 })}
                />
                <label className="flex items-center gap-1.5 text-xs text-zinc-500 cursor-pointer mt-1">
                  <input type="checkbox" checked={msLink}
                    onChange={(e) => onCalcChange({ msLinkRanges: e.target.checked })}
                    className="accent-indigo-500"
                  />
                  正負連動
                </label>
              </div>
            ) : null}
          </div>

        </div>
      )}
    </li>
  );
}

// ── 解析タブ ──────────────────────────
function AnalysisTab({ entries, params, unitMode, onLoadFiles, onAddFiles, onClearAll,
  onParamsChange, onUnitModeChange, onEntryDisplayChange, onEntryCalcChange, onEntryRemove, onEntryMove,
  onApplyFirstToAll }: {
  entries: FileEntry[]; params: AnalysisParams; unitMode: UnitMode;
  onLoadFiles: (f: FileWithPath[]) => void; onAddFiles: (f: FileWithPath[]) => void; onClearAll: () => void;
  onParamsChange: (n: Partial<AnalysisParams>) => void; onUnitModeChange: (m: UnitMode) => void;
  onEntryDisplayChange: (i: number, p: Partial<Pick<FileEntry, "legendName" | "color" | "markerSymbol" | "showAnnot">>) => void;
  onEntryCalcChange: (i: number, p: Partial<FileCalcSettings>) => void;
  onEntryRemove: (i: number) => void;
  onEntryMove: (from: number, to: number) => void;
  onApplyFirstToAll: () => void;
}) {
  const pickLoad = async () => { const f = await openVSMFiles(true); if (f.length) onLoadFiles(f); };
  const pickAdd  = async () => { const f = await openVSMFiles(true); if (f.length) onAddFiles(f); };

  return (
    <div className="flex-1 overflow-y-auto">
      <Section title="ファイル">
        <button onClick={pickLoad}
          className="w-full bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium py-2 px-3 rounded mb-2 transition-colors">
          ファイルを選択 (新規)
        </button>
        <button onClick={pickAdd}
          className="w-full bg-zinc-700 hover:bg-zinc-600 text-zinc-100 text-sm py-2 px-3 rounded mb-2 transition-colors">
          ファイルを追加...
        </button>
        <button onClick={onClearAll}
          className="w-full bg-zinc-800 hover:bg-red-900/50 text-zinc-400 hover:text-red-300 text-sm py-2 px-3 rounded transition-colors">
          全て削除
        </button>
      </Section>

      {entries.length > 0 && (
        <Section title={`読み込み済み (${entries.length} ファイル)`}>
          <ul className="space-y-2">
            {entries.map((e, i) => (
              <FileEntryItem key={i} entry={e} index={i} total={entries.length} params={params}
                onDisplayChange={(p) => onEntryDisplayChange(i, p)}
                onCalcChange={(p) => onEntryCalcChange(i, p)}
                onRemove={() => onEntryRemove(i)}
                onMove={(dir) => onEntryMove(i, i + dir)}
              />
            ))}
          </ul>
          {entries.length >= 2 && (
            <button onClick={onApplyFirstToAll}
              className="mt-3 w-full flex items-center justify-center gap-1.5 bg-zinc-800 hover:bg-amber-900/40 text-zinc-400 hover:text-amber-300 border border-zinc-700 hover:border-amber-700/50 text-xs py-2 px-3 rounded transition-colors">
              <svg viewBox="0 0 16 16" className="w-3.5 h-3.5 fill-current">
                <path d="M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2zm0 1.5a4.5 4.5 0 1 1 0 9 4.5 4.5 0 0 1 0-9zM7.25 5v3.25H4.5v1.5h2.75V13h1.5V9.75H11.5v-1.5H8.75V5h-1.5z"/>
              </svg>
              1番目の設定を全ファイルに適用
            </button>
          )}
        </Section>
      )}

      <Section title="サンプル情報（デフォルト）">
        <NumberInput label="膜厚 (nm)" value={params.thickness} step={1} min={0.1}
          onChange={(v) => onParamsChange({ thickness: v })} />
        <NumberInput label="面積 (mm²)" value={params.area} step={1} min={0.1}
          onChange={(v) => onParamsChange({ area: v })} />
      </Section>

      <Section title="解析設定">
        <Select label="表示単位系" value={unitMode}
          options={[
            { value: "SI",         label: "SI (T, kA/m)" },
            { value: "CGS",        label: "CGS (Oe, emu/cm³)" },
            { value: "Normalized", label: "Normalized (T, M/Ms)" },
          ]}
          onChange={(v) => onUnitModeChange(v as UnitMode)}
        />
        <Select label="反磁性補正（デフォルト）" value={params.demagMode}
          options={[
            { value: "auto", label: "自動検出" },
            { value: "none", label: "なし" },
          ]}
          onChange={(v) => onParamsChange({ demagMode: v as "auto" | "none" })}
        />
        <Toggle label="磁化オフセット補正" checked={params.offsetCorrection}
          onChange={(v) => onParamsChange({ offsetCorrection: v })} />
      </Section>

      <Section title="飽和磁場 (Hs)">
        <NumberInput label="許容範囲 (%)" value={params.hsTolerance} step={0.5} min={0.1}
          onChange={(v) => onParamsChange({ hsTolerance: v })} />
        <NumberInput label="連続点数 (最小)" value={params.hsMinConsecutive} step={1} min={1}
          onChange={(v) => onParamsChange({ hsMinConsecutive: Math.round(v) })} />
      </Section>
    </div>
  );
}

// ── グラフタブ用ヘルパー ──────────────────────────────────────

function GLabel({ children }: { children: React.ReactNode }) {
  return <p className="text-[9px] font-bold text-zinc-600 tracking-widest uppercase mb-1.5">{children}</p>;
}

function Pill({ active, onClick, children, wide, title }: {
  active: boolean; onClick: () => void; children: React.ReactNode; wide?: boolean; title?: string;
}) {
  return (
    <button onClick={onClick} title={title}
      className={`${wide ? "flex-1" : ""} px-2 py-1.5 text-[11px] font-medium rounded transition-colors ${
        active
          ? "bg-indigo-700/50 text-indigo-200 ring-1 ring-inset ring-indigo-600/50"
          : "bg-zinc-800 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-300"
      }`}>
      {children}
    </button>
  );
}

function CInput({ value, onChange, placeholder, type = "text", step }: {
  value: string | number; onChange: (v: string) => void;
  placeholder?: string; type?: string; step?: string;
}) {
  return (
    <input type={type} value={value} placeholder={placeholder} step={step}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-zinc-800/80 border border-zinc-700/60 text-zinc-100 text-xs rounded px-2 py-1.5 text-center focus:outline-none focus:border-indigo-500/70 placeholder:text-zinc-700"
    />
  );
}

// ── 凡例位置ポップオーバーピッカー ────────────────────────────
const LEGEND_POSITIONS = [
  "top-left",    "top-center",    "top-right",
  "mid-left",    "center",        "mid-right",
  "bottom-left", "bottom-center", "bottom-right",
] as const;

const POS_LABEL: Record<string, string> = {
  "top-left": "左上", "top-center": "上中", "top-right": "右上",
  "mid-left": "左中", "center":     "中央", "mid-right": "右中",
  "bottom-left": "左下", "bottom-center": "下中", "bottom-right": "右下",
};

function LegendPositionPicker({ value, onChange }: {
  value: GraphSettings["legendPosition"];
  onChange: (pos: GraphSettings["legendPosition"]) => void;
}) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const [popPos, setPopPos] = useState<{ top: number; left: number } | null>(null);

  const openPicker = () => {
    const rect = btnRef.current?.getBoundingClientRect();
    if (rect) setPopPos({ top: rect.bottom + 4, left: rect.left });
    setOpen((o) => !o);
  };

  const select = (pos: GraphSettings["legendPosition"]) => {
    onChange(pos);
    setOpen(false);
  };

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (!btnRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const currentIdx = LEGEND_POSITIONS.indexOf(value as typeof LEGEND_POSITIONS[number]);

  const popup = open && popPos && createPortal(
    <div
      style={{ position: "fixed", top: popPos.top, left: popPos.left, zIndex: 9999 }}
      className="bg-zinc-800 border border-zinc-600/80 rounded-lg shadow-2xl p-1.5"
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div className="grid grid-cols-3 gap-1">
        {LEGEND_POSITIONS.map((pos) => {
          const active = value === pos;
          return (
            <button key={pos} onClick={() => select(pos as GraphSettings["legendPosition"])}
              title={POS_LABEL[pos]}
              className={`w-9 h-7 rounded flex items-center justify-center transition-colors ${
                active
                  ? "bg-indigo-600 border border-indigo-400/60"
                  : "bg-zinc-700/70 border border-zinc-600/40 hover:bg-zinc-600/60"
              }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${active ? "bg-white" : "bg-zinc-500"}`} />
            </button>
          );
        })}
      </div>
    </div>,
    document.body,
  );

  return (
    <>
      <button ref={btnRef} onClick={openPicker}
        className="flex items-center gap-2 px-2 py-1.5 rounded border border-zinc-700/60 bg-zinc-800/60 hover:bg-zinc-700/60 hover:border-zinc-600 transition-colors w-full mb-2"
        title="凡例の表示位置を選択">
        <div className="grid grid-cols-3 gap-px shrink-0" style={{ width: 21, height: 15 }}>
          {LEGEND_POSITIONS.map((pos, i) => (
            <div key={pos} style={{ width: 5, height: 4 }}
              className={`rounded-sm ${currentIdx === i ? "bg-indigo-400" : "bg-zinc-600"}`} />
          ))}
        </div>
        <span className="text-[11px] text-zinc-300 flex-1 text-left">{POS_LABEL[value] ?? value}</span>
        <svg viewBox="0 0 10 6" className={`w-2.5 h-1.5 fill-current text-zinc-500 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}>
          <path d="M5 6L0 0h10z"/>
        </svg>
      </button>
      {popup}
    </>
  );
}

// ── 基本パネル ────────────────────────────
function BasicPanel({ g, ch }: { g: GraphSettings; ch: (p: Partial<GraphSettings>) => void }) {
  return (
    <div className="space-y-4">

      {/* 論文モード */}
      <div>
        <GLabel>出力モード</GLabel>
        <button onClick={() => ch({ paperMode: !g.paperMode })}
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg border transition-all ${
            g.paperMode
              ? "bg-indigo-950/50 border-indigo-700/60 text-indigo-200"
              : "bg-zinc-800/60 border-zinc-700/50 text-zinc-400 hover:border-zinc-600"
          }`}>
          <span className="text-xs font-semibold">論文モード</span>
          <span className={`relative w-9 h-5 rounded-full transition-colors shrink-0 ${g.paperMode ? "bg-indigo-500" : "bg-zinc-600"}`}>
            <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${g.paperMode ? "left-4" : "left-0.5"}`} />
          </span>
        </button>
        {g.paperMode && (
          <div className="flex gap-1 mt-1.5">
            {([
              { value: "current",   label: "現在色" },
              { value: "journal",   label: "Journal" },
              { value: "grayscale", label: "Grayscale" },
            ] as { value: PaperColorScheme; label: string }[]).map((opt) => (
              <Pill key={opt.value} wide active={g.paperColorScheme === opt.value}
                onClick={() => ch({ paperColorScheme: opt.value })}>
                {opt.label}
              </Pill>
            ))}
          </div>
        )}
      </div>

      {/* 表示トグル（4項目・2×2）。解析注釈はファイル一覧の各行で個別に切り替える */}
      <div>
        <GLabel>表示</GLabel>
        <div className="grid grid-cols-2 gap-1">
          <Pill wide active={g.showLegend}    onClick={() => ch({ showLegend:    !g.showLegend    })}>凡例</Pill>
          <Pill wide active={g.showGrid}      onClick={() => ch({ showGrid:      !g.showGrid      })}>グリッド</Pill>
          <Pill wide active={g.showZeroLines} onClick={() => ch({ showZeroLines: !g.showZeroLines })}>原点線</Pill>
          <Pill wide active={g.showExcluded}  onClick={() => ch({ showExcluded:  !g.showExcluded  })}
            title="OFFで除外点（灰色×）を画面・画像出力の両方から非表示にします">除外点</Pill>
        </div>
      </div>

      {/* 描画範囲 */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <GLabel>描画範囲</GLabel>
          <button onClick={() => ch({ xMin: "", xMax: "", yMin: "", yMax: "" })}
            className="text-[10px] text-zinc-600 hover:text-indigo-400 transition-colors">
            リセット
          </button>
        </div>
        <div className="space-y-1">
          {([
            { axis: "X", minKey: "xMin", maxKey: "xMax" },
            { axis: "Y", minKey: "yMin", maxKey: "yMax" },
          ] as { axis: string; minKey: "xMin" | "xMax"; maxKey: "xMin" | "xMax" }[]).map(({ axis, minKey, maxKey }) => (
            <div key={axis} className="flex items-center gap-1.5">
              <span className="text-[10px] text-zinc-600 w-3 text-center shrink-0">{axis}</span>
              <CInput type="number" value={g[minKey]} placeholder="min" onChange={(v) => ch({ [minKey]: v })} />
              <span className="text-zinc-700 text-xs shrink-0">—</span>
              <CInput type="number" value={g[maxKey]} placeholder="max" onChange={(v) => ch({ [maxKey]: v })} />
            </div>
          ))}
        </div>
      </div>

      {/* 凡例位置・列数 */}
      {g.showLegend && (
        <div>
          <GLabel>凡例位置</GLabel>
          <LegendPositionPicker value={g.legendPosition} onChange={(pos) => ch({ legendPosition: pos })} />
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-zinc-600 shrink-0">列数</span>
            <div className="flex gap-1 flex-1">
              {[1, 2, 3].map((n) => (
                <Pill key={n} wide active={g.legendColumns === n} onClick={() => ch({ legendColumns: n })}>
                  {n}列
                </Pill>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* プロット */}
      <div>
        <GLabel>プロット</GLabel>
        <div className="space-y-2.5">
          {/* 線幅スライダー */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-500 w-10 shrink-0">線幅</span>
            <input type="range" min={0.5} max={5} step={0.5} value={g.lineWidth}
              onChange={(e) => ch({ lineWidth: parseFloat(e.target.value) })}
              className="flex-1 accent-indigo-500 cursor-pointer" />
            <span className="text-[11px] text-zinc-300 font-mono w-5 text-right shrink-0">{g.lineWidth}</span>
          </div>
          {/* マーカーサイズスライダー */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-500 w-10 shrink-0">点サイズ</span>
            <input type="range" min={0} max={12} step={1} value={g.markerSize}
              onChange={(e) => ch({ markerSize: parseInt(e.target.value) })}
              className="flex-1 accent-indigo-500 cursor-pointer" />
            <span className="text-[11px] text-zinc-300 font-mono w-5 text-right shrink-0">
              {g.markerSize === 0 ? "off" : g.markerSize}
            </span>
          </div>
          {/* マーカー形状ピル */}
          {g.markerSize > 0 && (
            <div className="flex gap-1">
              {([
                ["circle",      "●"],
                ["square",      "■"],
                ["diamond",     "◆"],
                ["triangle-up", "▲"],
                ["cross",       "+"],
                ["x",           "×"],
              ] as [string, string][]).map(([sym, icon]) => (
                <Pill key={sym} wide active={g.markerSymbol === sym} onClick={() => ch({ markerSymbol: sym })}>
                  {icon}
                </Pill>
              ))}
            </div>
          )}
        </div>
      </div>

    </div>
  );
}

// ── 軸・目盛りパネル ─────────────────────────
function AxisPanel({ g, ch }: { g: GraphSettings; ch: (p: Partial<GraphSettings>) => void }) {
  return (
    <div className="space-y-4">

      {/* 軸ラベル上書き */}
      <div>
        <GLabel>軸ラベル</GLabel>
        <div className="space-y-1.5">
          {([
            { axis: "X", key: "xLabelOverride" as const, ph: "例: μ₀H (T)" },
            { axis: "Y", key: "yLabelOverride" as const, ph: "例: M (kA/m)" },
          ]).map(({ axis, key, ph }) => (
            <div key={axis} className="flex items-center gap-2">
              <span className="text-[10px] text-zinc-600 w-3 shrink-0 text-center">{axis}</span>
              <input type="text" value={g[key]} placeholder={ph}
                onChange={(e) => ch({ [key]: e.target.value })}
                title="TeX: $\alpha$ $\mu_0$ など"
                className="flex-1 bg-zinc-800/80 border border-zinc-700/60 text-zinc-100 text-xs rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500/70 placeholder:text-zinc-700"
              />
            </div>
          ))}
        </div>
      </div>

      {/* フォントサイズ */}
      <div>
        <GLabel>フォントサイズ</GLabel>
        <div className="grid grid-cols-3 gap-1.5">
          {([
            { label: "軸ラベル", key: "axisLabelSize"  as const },
            { label: "目盛り",   key: "tickLabelSize"  as const },
            { label: "凡例",     key: "legendFontSize" as const },
          ]).map(({ label, key }) => (
            <div key={key} className="text-center">
              <p className="text-[9px] text-zinc-600 mb-1">{label}</p>
              <input type="number" value={g[key]} min={6} step={1}
                onChange={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v) && v >= 6) ch({ [key]: v }); }}
                className="w-full bg-zinc-800/80 border border-zinc-700/60 text-zinc-100 text-xs rounded px-1 py-1.5 text-center focus:outline-none focus:border-indigo-500/70"
              />
            </div>
          ))}
        </div>
      </div>

      {/* 軸ラベル余白 */}
      <div>
        <GLabel>軸ラベル余白 (px)</GLabel>
        <div className="grid grid-cols-2 gap-1.5">
          {([
            { label: "下 (X軸)", key: "marginB" as const },
            { label: "左 (Y軸)", key: "marginL" as const },
          ]).map(({ label, key }) => (
            <div key={key} className="text-center">
              <p className="text-[9px] text-zinc-600 mb-1">{label}</p>
              <input type="number" value={g[key] ?? (key === "marginB" ? 70 : 90)} min={20} max={200} step={5}
                onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v) && v >= 20) ch({ [key]: v }); }}
                className="w-full bg-zinc-800/80 border border-zinc-700/60 text-zinc-100 text-xs rounded px-1 py-1.5 text-center focus:outline-none focus:border-indigo-500/70"
              />
            </div>
          ))}
        </div>
      </div>

      {/* 目盛り間隔 */}
      <div>
        <GLabel>目盛り間隔</GLabel>
        <div className="flex gap-1.5 mb-2">
          {([
            { axis: "X", key: "xDtick" as const },
            { axis: "Y", key: "yDtick" as const },
          ]).map(({ axis, key }) => (
            <div key={axis} className="flex items-center gap-1 flex-1">
              <span className="text-[10px] text-zinc-600 shrink-0">{axis}</span>
              <CInput type="number" value={g[key] ?? ""} placeholder="auto" step="any"
                onChange={(v) => ch({ [key]: v })} />
            </div>
          ))}
        </div>
        {/* 補助目盛り */}
        <div className="flex items-center gap-1.5">
          <Pill active={g.showMinorTicks ?? true}
            onClick={() => ch({ showMinorTicks: !(g.showMinorTicks ?? true) })}>
            補助目盛り
          </Pill>
          {(g.showMinorTicks ?? true) && (
            <select value={String(g.minorDivisions ?? 5)}
              onChange={(e) => ch({ minorDivisions: Number(e.target.value) })}
              className="flex-1 bg-zinc-800/80 border border-zinc-700/60 text-zinc-300 text-[11px] rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500/70"
            >
              <option value="2">÷2（間1本）</option>
              <option value="4">÷4（間3本）</option>
              <option value="5">÷5（間4本）</option>
              <option value="10">÷10（間9本）</option>
            </select>
          )}
        </div>
      </div>

      {/* 目盛りラベル書式 */}
      <div>
        <GLabel>目盛りラベル書式</GLabel>
        <div className="grid grid-cols-2 gap-1.5">
          {([
            { label: "X", key: "xTickFormat" as const },
            { label: "Y", key: "yTickFormat" as const },
          ]).map(({ label, key }) => (
            <div key={key}>
              <p className="text-[9px] text-zinc-600 mb-1">{label}</p>
              <select value={g[key]} onChange={(e) => ch({ [key]: e.target.value })}
                className="w-full bg-zinc-800/80 border border-zinc-700/60 text-zinc-300 text-[11px] rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500/70"
              >
                <option value="">自動</option>
                <option value=".0f">整数</option>
                <option value=".1f">小数1桁</option>
                <option value=".2f">小数2桁</option>
                <option value=".3f">小数3桁</option>
                <option value=".2g">有効数字2桁</option>
                <option value=".3g">有効数字3桁</option>
                <option value=".2e">指数表記</option>
              </select>
            </div>
          ))}
        </div>
      </div>

      {/* 線スタイル */}
      <div>
        <GLabel>線スタイル</GLabel>
        {([
          { label: "原点線", styleKey: "zeroLineStyle" as const, colorKey: "zeroLineColor" as const },
          { label: "グリッド", styleKey: "gridStyle"    as const, colorKey: "gridColor"    as const },
        ]).map(({ label, styleKey, colorKey }) => (
          <div key={styleKey} className="flex items-center gap-1.5 mb-1.5">
            <span className="text-[10px] text-zinc-500 w-12 shrink-0">{label}</span>
            <select value={g[styleKey]} onChange={(e) => ch({ [styleKey]: e.target.value })}
              className="flex-1 bg-zinc-800/80 border border-zinc-700/60 text-zinc-300 text-[11px] rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500/70"
            >
              <option value="solid">実線 ──</option>
              <option value="dash">破線 --</option>
              <option value="dot">点線 ···</option>
            </select>
            <input type="color" value={g[colorKey]}
              onChange={(e) => ch({ [colorKey]: e.target.value })}
              className="w-8 h-7 rounded cursor-pointer border border-zinc-700 bg-transparent shrink-0"
            />
          </div>
        ))}
      </div>

    </div>
  );
}

// ── グラフ設定タブ ──────────────────────────
function GraphTab({ graphSettings, onChange }: { graphSettings: GraphSettings; onChange: (next: Partial<GraphSettings>) => void }) {
  const [sub, setSub] = useState<"basic" | "axis">("basic");

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* サブタブ */}
      <div className="flex shrink-0 gap-1 px-3 pt-3">
        {(["basic", "axis"] as const).map((id) => (
          <button key={id} onClick={() => setSub(id)}
            className={`flex-1 py-1.5 text-xs font-medium rounded transition-colors ${
              sub === id
                ? "bg-indigo-700/40 text-indigo-200 ring-1 ring-inset ring-indigo-600/40"
                : "text-zinc-500 hover:text-zinc-300 bg-zinc-800/40 hover:bg-zinc-800"
            }`}>
            {id === "basic" ? "基本" : "軸・目盛り"}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {sub === "basic"
          ? <BasicPanel g={graphSettings} ch={onChange} />
          : <AxisPanel  g={graphSettings} ch={onChange} />
        }
      </div>
    </div>
  );
}

// ── 保存タブ ──────────────────────────
function SaveTab({ onSave, onLoad, hasEntries }: {
  onSave: () => Promise<boolean>; onLoad: () => void; hasEntries: boolean;
}) {
  const [showExport, setShowExport] = useState(false);

  return (
    <div className="flex-1 overflow-y-auto">
      {showExport && <ExportDialog onClose={() => setShowExport(false)} />}

      {/* ── グラフ画像 ── */}
      <Section title="グラフ画像出力">
        <button onClick={() => setShowExport(true)}
          className="w-full flex items-center justify-center gap-2 bg-indigo-700 hover:bg-indigo-600 text-white text-sm font-medium py-2 px-3 rounded transition-colors">
          🖼 画像を出力...
        </button>
      </Section>

      {/* ── セッション ── */}
      <Section title="セッション">
        <button onClick={async () => { await onSave(); }}
          disabled={!hasEntries}
          className="w-full bg-zinc-700 hover:bg-zinc-600 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed text-zinc-100 text-sm py-2 px-3 rounded mb-2 transition-colors">
          セッションを保存...
        </button>
        <button onClick={onLoad}
          className="w-full bg-zinc-700 hover:bg-zinc-600 text-zinc-100 text-sm py-2 px-3 rounded transition-colors">
          セッションを読み込み...
        </button>
      </Section>
    </div>
  );
}

// ── ログタブ用定数 ────────────────────────────────────────────
const META_DISPLAY: { key: string; label: string }[] = [
  { key: "date",                       label: "測定日" },
  { key: "sample name",                label: "サンプル名" },
  { key: "measuring points",           label: "測定点数" },
  { key: "max magnetic field",         label: "最大磁場 (Oe)" },
  { key: "lock-in amp. sensitivity",   label: "LIA 感度" },
  { key: "lock-in amp. time constant", label: "LIA 時定数 (ms)" },
  { key: "lock-in amp. phase",         label: "LIA 位相 (°)" },
  { key: "calibration value",          label: "校正定数" },
  { key: "measuring time",             label: "測定時間 (min)" },
  { key: "pole piece gap",             label: "ポール間隔 (mm)" },
  { key: "magnet angle",               label: "磁場角度 (°)" },
];

const CORRECTION_KEYS = [
  "correction(demagnetization field)",
  "correction(diamagnetism)",
  "correction(subtraction)",
  "correction(addition)",
  "correction(spline)",
  "correction(smoothing)",
  "correction(image effect)",
] as const;

const CORRECTION_LABELS: Record<string, string> = {
  "correction(demagnetization field)": "反磁場補正",
  "correction(diamagnetism)":          "反磁性補正",
  "correction(subtraction)":           "差し引き補正",
  "correction(addition)":              "加算補正",
  "correction(spline)":                "スプライン補正",
  "correction(smoothing)":             "スムージング",
  "correction(image effect)":          "イメージ効果補正",
};

// ── ログタブ ──────────────────────────
function LogTab({ entries }: { entries: FileEntry[] }) {
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [logFilter,     setLogFilter]     = useState("");

  if (entries.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-xs text-zinc-600">ファイルを読み込むとログが表示されます</p>
      </div>
    );
  }

  const idx    = Math.min(selectedIndex, Math.max(0, entries.length - 1));
  const target = entries[idx] ?? entries[0];
  const result = target?.result;
  const logs   = result?.logs ?? [];


  // ── ログ行の色分け ──
  const logLineClass = (line: string): string => {
    if (/^---/.test(line))                                        return "text-zinc-200 font-semibold";
    if (/警告|warn/i.test(line))                                  return "text-amber-400";
    if (/エラー|error/i.test(line))                               return "text-red-400";
    if (/R²/.test(line) || /[Mm]s=|[Mm]r=|[Hh]c=|[Hh]s=/.test(line)) return "text-cyan-300";
    if (/補正|correction/i.test(line))                            return "text-emerald-300";
    return "text-zinc-400";
  };

  // ── ログからR²抽出 / slopeはAPIフィールドを直接使用 ──
  const logText = logs.join("\n");
  const r2Match = logText.match(/R²=\[正\s*([\d.]+),\s*負\s*([\d.]+)\]/);
  const r2Pos   = r2Match ? parseFloat(r2Match[1]) : null;
  const r2Neg   = r2Match ? parseFloat(r2Match[2]) : null;
  const slope   = result?.demag_slope ?? null;

  const r2Color = (v: number | null) =>
    v === null ? "text-zinc-500" : v >= 0.9995 ? "text-emerald-400" : v >= 0.999 ? "text-yellow-400" : "text-red-400";

  // ── フィルター済みリスト ──
  const filteredLogs = logFilter
    ? logs.filter((l) => l.toLowerCase().includes(logFilter.toLowerCase()))
    : logs;
  const meta = result?.metadata ?? {};

  // 表示すべき重要フィールド（0・空・デフォルト値を除外）
  const displayMeta = META_DISPLAY
    .map(({ key, label }) => ({ label, value: meta[key] }))
    .filter(({ value }) => value != null && value !== "0" && value !== "");

  // 補正フラグのサマリ
  const activeCorrections = CORRECTION_KEYS
    .filter((k) => meta[k]?.toUpperCase() === "YES")
    .map((k) => CORRECTION_LABELS[k]);
  const correctionPresent = CORRECTION_KEYS.some((k) => k in meta);

  const copyLog  = () => navigator.clipboard.writeText(logs.join("\n"));
  const copyMeta = () =>
    navigator.clipboard.writeText(displayMeta.map(({ label, value }) => `${label}: ${value}`).join("\n"));

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* ファイル選択 */}
      <div className="px-3 pt-3 pb-2 border-b border-zinc-800 shrink-0">
        <select value={idx} onChange={(e) => setSelectedIndex(Number(e.target.value))}
          className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500">
          {entries.map((e, i) => (
            <option key={i} value={i}>{e.legendName || e.file.name}</option>
          ))}
        </select>
        {target.error && (
          <p className="mt-1.5 text-[10px] text-red-400 bg-red-900/20 rounded px-2 py-1">{target.error}</p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">

        {/* ── 解析値カード ── */}
        {result && (
          <div className="px-3 py-2.5 border-b border-zinc-800">
            <p className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest mb-2">解析値</p>
            <div className="grid grid-cols-2 gap-1.5 mb-2">
              {([
                { label: "Ms", value: result.Ms?.toFixed(1),                                   unit: "kA/m" },
                { label: "Mr", value: result.Mr?.toFixed(1),                                   unit: "kA/m" },
                { label: "Hc", value: result.Hc_Oe != null ? (result.Hc_Oe * 0.1).toFixed(2) : null, unit: "mT" },
                { label: "Hs", value: result.Hs_Oe != null ? (result.Hs_Oe * 0.1).toFixed(2) : null, unit: "mT" },
              ]).map(({ label, value, unit }) => (
                <div key={label} className="bg-zinc-800/60 rounded px-2.5 py-2 border border-zinc-700/40">
                  <p className="text-[9px] text-zinc-500 font-mono mb-0.5">{label}</p>
                  <p className="text-[13px] font-mono font-semibold text-zinc-100 leading-none">
                    {value ?? "—"}
                    <span className="text-[9px] text-zinc-500 ml-1 font-normal">{unit}</span>
                  </p>
                </div>
              ))}
            </div>
            {/* 保形性バー */}
            {result.squareness != null && (
              <div className="bg-zinc-800/60 rounded px-2.5 py-2 border border-zinc-700/40">
                <div className="flex items-center justify-between mb-1.5">
                  <p className="text-[9px] text-zinc-500 font-mono">保形性 Mr/Ms</p>
                  <p className="text-[11px] font-mono font-semibold text-zinc-200">{result.squareness.toFixed(3)}</p>
                </div>
                <div className="h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-indigo-500 transition-all"
                    style={{ width: `${Math.min(100, result.squareness * 100)}%` }} />
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── 反磁性補正品質 ── */}
        {(r2Pos !== null || slope !== null) && (
          <div className="px-3 py-2.5 border-b border-zinc-800">
            <p className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest mb-2">反磁性補正品質</p>
            <div className="space-y-1.5">
              {slope !== null && (
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-zinc-500">補正傾き</span>
                  <span className="text-[10px] font-mono text-zinc-300">{slope.toExponential(3)}</span>
                </div>
              )}
              {([
                { label: "R² (正側)", val: r2Pos },
                { label: "R² (負側)", val: r2Neg },
              ]).filter(({ val }) => val !== null).map(({ label, val }) => (
                <div key={label} className="flex items-center justify-between">
                  <span className="text-[10px] text-zinc-500">{label}</span>
                  <span className={`text-[10px] font-mono font-semibold ${r2Color(val)}`}>
                    {val!.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── 測定情報 ── */}
        {(displayMeta.length > 0 || correctionPresent) && (
          <div className="px-3 py-2.5 border-b border-zinc-800">
            <div className="flex items-center justify-between mb-1.5">
              <p className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest">測定情報</p>
              {displayMeta.length > 0 && (
                <button onClick={copyMeta}
                  className="text-[9px] text-zinc-600 hover:text-indigo-400 transition-colors px-1.5 py-0.5 rounded hover:bg-indigo-900/20">
                  コピー
                </button>
              )}
            </div>
            <dl className="space-y-0">
              {displayMeta.map(({ label, value }) => (
                <div key={label} className="flex gap-2 text-[10px] py-0.5 rounded px-1 -mx-1 hover:bg-zinc-800/50 group">
                  <dt className="text-zinc-500 shrink-0" style={{ width: "8.5rem" }}>{label}</dt>
                  <dd className="text-zinc-300 break-all min-w-0 group-hover:text-zinc-100 transition-colors">{value}</dd>
                </div>
              ))}
              {/* 装置補正サマリ */}
              {correctionPresent && (
                <div className="flex gap-2 text-[10px] py-0.5 rounded px-1 -mx-1 mt-0.5 border-t border-zinc-800/60 pt-1">
                  <dt className="text-zinc-500 shrink-0" style={{ width: "8.5rem" }}>VSM装置補正</dt>
                  <dd className={activeCorrections.length > 0 ? "text-amber-300 break-all" : "text-zinc-600"}>
                    {activeCorrections.length > 0
                      ? activeCorrections.join(", ")
                      : "なし（このアプリで処理）"}
                  </dd>
                </div>
              )}
            </dl>
          </div>
        )}

        {/* ── 解析ログ ── */}
        <div className="px-3 py-2.5">
          <div className="flex items-center justify-between mb-2">
            <p className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest">
              解析ログ <span className="text-zinc-700 font-normal">({logs.length}行)</span>
            </p>
            {logs.length > 0 && (
              <button onClick={copyLog}
                className="text-[9px] text-zinc-600 hover:text-indigo-400 transition-colors px-1.5 py-0.5 rounded hover:bg-indigo-900/20">
                コピー
              </button>
            )}
          </div>
          {logs.length > 0 && (
            <input type="text" value={logFilter} onChange={(e) => setLogFilter(e.target.value)}
              placeholder="ログを絞り込み..."
              className="w-full bg-zinc-800/60 border border-zinc-700/40 text-zinc-300 text-[10px] rounded px-2 py-1 mb-2 focus:outline-none focus:border-indigo-500/60 placeholder:text-zinc-700"
            />
          )}
          {filteredLogs.length > 0 ? (
            <div className="space-y-0.5 font-mono text-[10px] leading-relaxed">
              {filteredLogs.map((line, i) => (
                <p key={i} className={`${logLineClass(line)} whitespace-pre-wrap break-all`}>{line}</p>
              ))}
              {logFilter && filteredLogs.length === 0 && (
                <p className="text-zinc-600">「{logFilter}」に一致しません</p>
              )}
            </div>
          ) : !logFilter && target.error ? (
            <p className="text-[10px] text-red-400">{target.error}</p>
          ) : !logFilter ? (
            <p className="text-[10px] text-zinc-600">ログなし</p>
          ) : null}
        </div>

      </div>
    </div>
  );
}

// ── メイン Sidebar ──────────────────────────
export default function Sidebar({
  style,
  entries, params, unitMode, graphSettings,
  onLoadFiles, onAddFiles, onClearAll,
  onParamsChange, onUnitModeChange, onGraphSettingsChange,
  onEntryDisplayChange, onEntryCalcChange, onEntryRemove, onEntryMove,
  onApplyFirstToAll, onSaveSession, onLoadSession,
}: Props) {
  const [activeTab, setActiveTab] = useState<Tab>("analysis");

  const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
    {
      id: "analysis", label: "解析",
      icon: (
        <svg viewBox="0 0 14 10" width="13" height="10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round">
          <polyline points="0,5 2,5 3,1 4.5,9 6,3 7.5,7 9,5 14,5" />
        </svg>
      ),
    },
    {
      id: "graph", label: "グラフ",
      icon: (
        <svg viewBox="0 0 14 12" width="13" height="11" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round">
          <polyline points="1,10 5,5 9,7 13,2" />
          <line x1="1" y1="11.5" x2="13" y2="11.5" strokeOpacity="0.5" />
        </svg>
      ),
    },
    {
      id: "save", label: "保存",
      icon: (
        <svg viewBox="0 0 14 14" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round">
          <path d="M7 1v8M4 6l3 3 3-3" />
          <path d="M1 11v2h12v-2" />
        </svg>
      ),
    },
    {
      id: "log", label: "ログ",
      icon: (
        <svg viewBox="0 0 14 14" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round">
          <rect x="2" y="1" width="10" height="12" rx="1" />
          <line x1="5" y1="5" x2="9" y2="5" />
          <line x1="5" y1="8" x2="9" y2="8" />
        </svg>
      ),
    },
  ];

  return (
    <aside className="shrink-0 bg-zinc-900 flex flex-col overflow-hidden" style={style}>

      <div className="flex border-b border-zinc-800 shrink-0">
        {TABS.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-2 text-[10px] font-medium transition-colors flex flex-col items-center gap-0.5 ${
              activeTab === tab.id
                ? "text-indigo-400 border-b-2 border-indigo-500"
                : "text-zinc-500 hover:text-zinc-300 border-b-2 border-transparent"
            }`}>
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "analysis" && (
        <AnalysisTab
          entries={entries} params={params} unitMode={unitMode}
          onLoadFiles={onLoadFiles} onAddFiles={onAddFiles} onClearAll={onClearAll}
          onParamsChange={onParamsChange} onUnitModeChange={onUnitModeChange}
          onEntryDisplayChange={onEntryDisplayChange}
          onEntryCalcChange={onEntryCalcChange}
          onEntryRemove={onEntryRemove}
          onEntryMove={onEntryMove}
          onApplyFirstToAll={onApplyFirstToAll}
        />
      )}
      {activeTab === "graph" && (
        <GraphTab graphSettings={graphSettings} onChange={onGraphSettingsChange} />
      )}
      {activeTab === "save" && (
        <SaveTab onSave={onSaveSession} onLoad={onLoadSession} hasEntries={entries.length > 0} />
      )}
      {activeTab === "log"  && <LogTab entries={entries} />}
    </aside>
  );
}
