import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { FileEntry, UnitMode, GraphSettings, PaperColorScheme } from "../App";
import type { AnalysisParams, FileCalcSettings, FileWithPath } from "../api/client";
import { openVSMFiles } from "../api/client";
import type { ExportOptions } from "../utils/graphExport";
import { searchSuggestions, getCurrentToken } from "../utils/legendSuggestions";
import type { Suggestion } from "../utils/legendSuggestions";

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
  onEntryDisplayChange:  (index: number, patch: Partial<Pick<FileEntry, "legendName" | "color" | "markerSymbol">>) => void;
  onEntryCalcChange:     (index: number, patch: Partial<FileCalcSettings>) => void;
  onEntryRemove:         (index: number) => void;
  onEntryMove:           (from: number, to: number) => void;
  onApplyFirstToAll:     () => void;
  onSaveSession:         () => Promise<boolean>;
  onLoadSession:         () => void;
  onSaveGraph:           (opts: ExportOptions) => Promise<boolean>;
  onCopyGraph:           (scale: number) => Promise<void>;
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
  onDisplayChange: (patch: Partial<Pick<FileEntry, "legendName" | "color" | "markerSymbol">>) => void;
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
            <p className="text-xs text-zinc-600 mb-2">
              {demagMode === "" ? `GBL = グローバル設定 (${params.demagMode === "auto" ? "自動" : "なし"}) を使用` :
               demagMode === "auto" ? "高磁場領域の傾きを自動検出" :
               demagMode === "manual" ? "磁場範囲を手動で指定" : "補正を行わない"}
            </p>
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
            ) : (
              <p className="text-xs text-zinc-600">全磁場域の飽和値から自動計算</p>
            )}
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
  onEntryDisplayChange: (i: number, p: Partial<Pick<FileEntry, "legendName" | "color" | "markerSymbol">>) => void;
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
        <p className="text-xs text-zinc-600 mb-3">各ファイルの ⚙ で個別上書き可能</p>
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

// ── グラフ設定タブ ──────────────────────────
function GraphTab({ graphSettings, onChange }: { graphSettings: GraphSettings; onChange: (next: Partial<GraphSettings>) => void }) {
  return (
    <div className="flex-1 overflow-y-auto">

      {/* ── 論文モード ── */}
      <Section title="論文モード">
        <Toggle
          label="論文モード（白背景・黒軸・四辺枠）"
          checked={graphSettings.paperMode}
          onChange={(v) => onChange({ paperMode: v })}
        />
        {graphSettings.paperMode && (
          <div className="mt-3 space-y-2">
            <label className="text-xs text-zinc-400 block mb-1.5">配色スキーム</label>
            {(
              [
                { value: "current",  label: "現在の色そのまま" },
                { value: "journal",  label: "ジャーナル標準色" },
                { value: "grayscale", label: "グレースケール（モノクロ印刷対応）" },
              ] as { value: PaperColorScheme; label: string }[]
            ).map((opt) => (
              <label key={opt.value} className="flex items-center gap-2 cursor-pointer group">
                <input
                  type="radio"
                  name="paperColorScheme"
                  value={opt.value}
                  checked={graphSettings.paperColorScheme === opt.value}
                  onChange={() => onChange({ paperColorScheme: opt.value })}
                  className="accent-indigo-500"
                />
                <span className="text-xs text-zinc-300 group-hover:text-zinc-100 transition-colors">
                  {opt.label}
                </span>
              </label>
            ))}
          </div>
        )}
      </Section>

      <Section title="表示">
        <Toggle label="凡例を表示"      checked={graphSettings.showLegend}    onChange={(v) => onChange({ showLegend: v })} />
        <Toggle label="グリッド線を表示" checked={graphSettings.showGrid}      onChange={(v) => onChange({ showGrid: v })} />
        <Toggle label="原点線を表示"    checked={graphSettings.showZeroLines} onChange={(v) => onChange({ showZeroLines: v })} />
      </Section>

      <Section title="凡例">
        <Select label="位置" value={graphSettings.legendPosition}
          options={[
            { value: "top-right",    label: "右上" },
            { value: "top-left",     label: "左上" },
            { value: "bottom-right", label: "右下" },
            { value: "bottom-left",  label: "左下" },
          ]}
          onChange={(v) => onChange({ legendPosition: v as GraphSettings["legendPosition"] })}
        />
        <NumberInput label="フォントサイズ" value={graphSettings.legendFontSize} step={1} min={6}
          onChange={(v) => onChange({ legendFontSize: v })} />
      </Section>

      <Section title="軸ラベル">
        <div className="mb-3">
          <label className="text-xs text-zinc-400 block mb-1">X 軸ラベル（空欄 = 自動）</label>
          <input type="text" value={graphSettings.xLabelOverride}
            placeholder="例: μ₀H (T)"
            onChange={(e) => onChange({ xLabelOverride: e.target.value })}
            className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500"
          />
        </div>
        <div className="mb-3">
          <label className="text-xs text-zinc-400 block mb-1">Y 軸ラベル（空欄 = 自動）</label>
          <input type="text" value={graphSettings.yLabelOverride}
            placeholder="例: M (kA/m)"
            onChange={(e) => onChange({ yLabelOverride: e.target.value })}
            className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500"
          />
        </div>
        <NumberInput label="軸ラベルフォントサイズ" value={graphSettings.axisLabelSize} step={1} min={6}
          onChange={(v) => onChange({ axisLabelSize: v })} />
        <NumberInput label="目盛りフォントサイズ" value={graphSettings.tickLabelSize} step={1} min={6}
          onChange={(v) => onChange({ tickLabelSize: v })} />
      </Section>

      <Section title="描画範囲（空欄 = 自動）">
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div>
            <label className="text-xs text-zinc-400 block mb-1">X 最小</label>
            <input type="number" value={graphSettings.xMin} placeholder="auto"
              onChange={(e) => onChange({ xMin: e.target.value })}
              className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500"
            />
          </div>
          <div>
            <label className="text-xs text-zinc-400 block mb-1">X 最大</label>
            <input type="number" value={graphSettings.xMax} placeholder="auto"
              onChange={(e) => onChange({ xMax: e.target.value })}
              className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500"
            />
          </div>
          <div>
            <label className="text-xs text-zinc-400 block mb-1">Y 最小</label>
            <input type="number" value={graphSettings.yMin} placeholder="auto"
              onChange={(e) => onChange({ yMin: e.target.value })}
              className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500"
            />
          </div>
          <div>
            <label className="text-xs text-zinc-400 block mb-1">Y 最大</label>
            <input type="number" value={graphSettings.yMax} placeholder="auto"
              onChange={(e) => onChange({ yMax: e.target.value })}
              className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500"
            />
          </div>
        </div>
        <button onClick={() => onChange({ xMin: "", xMax: "", yMin: "", yMax: "" })}
          className="w-full bg-zinc-700 hover:bg-zinc-600 text-zinc-300 text-xs py-1.5 px-3 rounded transition-colors">
          範囲をリセット (自動)
        </button>
      </Section>

      <Section title="プロット">
        <NumberInput label="線幅" value={graphSettings.lineWidth} step={0.5} min={0.5}
          onChange={(v) => onChange({ lineWidth: v })} />
        <NumberInput label="マーカーサイズ（0 = なし）" value={graphSettings.markerSize} step={1} min={0}
          onChange={(v) => onChange({ markerSize: v })} />
        <Select label="マーカー形状" value={graphSettings.markerSymbol}
          options={[
            { value: "circle",      label: "○ 丸" },
            { value: "square",      label: "□ 四角" },
            { value: "diamond",     label: "◇ ひし形" },
            { value: "triangle-up", label: "△ 三角" },
            { value: "cross",       label: "+ クロス" },
            { value: "x",           label: "× バツ" },
          ]}
          onChange={(v) => onChange({ markerSymbol: v })}
        />
      </Section>

      <Section title="目盛り書式">
        {/* 目盛り間隔 */}
        <div className="mb-3">
          <p className="text-xs text-zinc-500 mb-2">目盛り間隔（空欄 = 自動）</p>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-zinc-400 block mb-1">X 間隔</label>
              <input type="number" value={graphSettings.xDtick ?? ""} step="any" min={0}
                placeholder="auto"
                onChange={(e) => onChange({ xDtick: e.target.value })}
                className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500 placeholder:text-zinc-600"
              />
            </div>
            <div>
              <label className="text-xs text-zinc-400 block mb-1">Y 間隔</label>
              <input type="number" value={graphSettings.yDtick ?? ""} step="any" min={0}
                placeholder="auto"
                onChange={(e) => onChange({ yDtick: e.target.value })}
                className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-indigo-500 placeholder:text-zinc-600"
              />
            </div>
          </div>
        </div>
        {/* 目盛りラベル書式 */}
        <Select label="X ラベル書式" value={graphSettings.xTickFormat}
          options={[
            { value: "",     label: "自動" },
            { value: ".0f",  label: "整数 (例: 1)" },
            { value: ".1f",  label: "小数1桁 (例: 1.0)" },
            { value: ".2f",  label: "小数2桁 (例: 1.00)" },
            { value: ".3f",  label: "小数3桁 (例: 1.000)" },
            { value: ".2g",  label: "有効数字2桁" },
            { value: ".3g",  label: "有効数字3桁" },
            { value: ".2e",  label: "指数表記 (例: 1.00e+0)" },
          ]}
          onChange={(v) => onChange({ xTickFormat: v })}
        />
        <Select label="Y ラベル書式" value={graphSettings.yTickFormat}
          options={[
            { value: "",     label: "自動" },
            { value: ".0f",  label: "整数 (例: 100)" },
            { value: ".1f",  label: "小数1桁 (例: 100.0)" },
            { value: ".2f",  label: "小数2桁 (例: 100.00)" },
            { value: ".2g",  label: "有効数字2桁" },
            { value: ".3g",  label: "有効数字3桁" },
            { value: ".2e",  label: "指数表記 (例: 1.00e+2)" },
          ]}
          onChange={(v) => onChange({ yTickFormat: v })}
        />
      </Section>

      <Section title="線スタイル">
        <Select label="原点線のスタイル" value={graphSettings.zeroLineStyle}
          options={[
            { value: "solid", label: "実線 ──" },
            { value: "dash",  label: "破線 --" },
            { value: "dot",   label: "点線 ···" },
          ]}
          onChange={(v) => onChange({ zeroLineStyle: v })}
        />
        <div className="mb-3">
          <label className="text-xs text-zinc-400 block mb-1">原点線の色</label>
          <div className="flex items-center gap-2">
            <input type="color" value={graphSettings.zeroLineColor}
              onChange={(e) => onChange({ zeroLineColor: e.target.value })}
              className="w-8 h-8 rounded cursor-pointer border border-zinc-600 bg-transparent"
            />
            <span className="text-xs text-zinc-400">{graphSettings.zeroLineColor}</span>
          </div>
        </div>
        <Select label="グリッドのスタイル" value={graphSettings.gridStyle}
          options={[
            { value: "solid", label: "実線 ──" },
            { value: "dash",  label: "破線 --" },
            { value: "dot",   label: "点線 ···" },
          ]}
          onChange={(v) => onChange({ gridStyle: v })}
        />
        <div className="mb-3">
          <label className="text-xs text-zinc-400 block mb-1">グリッドの色</label>
          <div className="flex items-center gap-2">
            <input type="color" value={graphSettings.gridColor}
              onChange={(e) => onChange({ gridColor: e.target.value })}
              className="w-8 h-8 rounded cursor-pointer border border-zinc-600 bg-transparent"
            />
            <span className="text-xs text-zinc-400">{graphSettings.gridColor}</span>
          </div>
        </div>
      </Section>
    </div>
  );
}

// ── 保存タブ ──────────────────────────
function SaveTab({ onSave, onLoad, hasEntries, onSaveGraph, onCopyGraph }: {
  onSave: () => Promise<boolean>; onLoad: () => void; hasEntries: boolean;
  onSaveGraph: (opts: ExportOptions) => Promise<boolean>;
  onCopyGraph: (scale: number) => Promise<void>;
}) {

  const [format,        setFormat]        = useState<ExportOptions["format"]>("png");
  const [scale,         setScale]         = useState(2);
  const [useCustomSize, setUseCustomSize] = useState(false);
  const [width,         setWidth]         = useState(1200);
  const [height,        setHeight]        = useState(900);
  const [saving,        setSaving]        = useState(false);
  const [copying,       setCopying]       = useState(false);
  const [msg,           setMsg]           = useState<{ text: string; ok: boolean } | null>(null);

  const flash = (text: string, ok: boolean) => {
    setMsg({ text, ok });
    setTimeout(() => setMsg(null), 2500);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const saved = await onSaveGraph({ format, scale, useCustomSize, width, height });
      if (saved) flash("保存しました", true);
    } catch (e: unknown) {
      flash((e as Error).message ?? "保存に失敗しました", false);
    } finally { setSaving(false); }
  };

  const handleCopy = async () => {
    setCopying(true);
    try {
      await onCopyGraph(scale);
      flash("クリップボードにコピーしました", true);
    } catch (e: unknown) {
      flash((e as Error).message ?? "コピーに失敗しました", false);
    } finally { setCopying(false); }
  };

  const SCALES = [1, 2, 3, 4];
  const FORMATS: { value: ExportOptions["format"]; label: string }[] = [
    { value: "png",  label: "PNG" },
    { value: "svg",  label: "SVG" },
    { value: "jpeg", label: "JPEG" },
  ];

  return (
    <div className="flex-1 overflow-y-auto">

      {/* ── グラフ画像 ── */}
      <Section title="グラフ画像出力">

        {/* 形式 */}
        <div className="mb-3">
          <label className="text-xs text-zinc-400 block mb-1.5">ファイル形式</label>
          <div className="flex rounded-md overflow-hidden border border-zinc-700">
            {FORMATS.map((f, i) => (
              <button key={f.value} onClick={() => setFormat(f.value)}
                className={`flex-1 text-xs py-1.5 font-medium transition-colors ${
                  format === f.value ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                } ${i > 0 ? "border-l border-zinc-700" : ""}`}>
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* 解像度倍率 */}
        <div className="mb-3">
          <label className="text-xs text-zinc-400 block mb-1.5">解像度倍率</label>
          <div className="flex rounded-md overflow-hidden border border-zinc-700">
            {SCALES.map((s, i) => (
              <button key={s} onClick={() => setScale(s)}
                className={`flex-1 text-xs py-1.5 font-mono transition-colors ${
                  scale === s ? "bg-indigo-600 text-white font-bold" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                } ${i > 0 ? "border-l border-zinc-700" : ""}`}>
                ×{s}
              </button>
            ))}
          </div>
          <p className="text-[10px] text-zinc-600 mt-1">×2 = 画面サイズの2倍 (推奨)</p>
        </div>

        {/* カスタムサイズ */}
        <div className="mb-4">
          <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer mb-2">
            <input type="checkbox" checked={useCustomSize} onChange={(e) => setUseCustomSize(e.target.checked)}
              className="accent-indigo-500" />
            カスタムサイズ (px)
          </label>
          {useCustomSize && (
            <div className="grid grid-cols-2 gap-2 pl-4 border-l-2 border-indigo-800/40">
              <div>
                <label className="text-[10px] text-zinc-500 block mb-1">幅</label>
                <input type="number" value={width} min={100} step={100}
                  onChange={(e) => setWidth(Number(e.target.value))}
                  className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="text-[10px] text-zinc-500 block mb-1">高さ</label>
                <input type="number" value={height} min={100} step={100}
                  onChange={(e) => setHeight(Number(e.target.value))}
                  className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>
          )}
        </div>

        {/* フィードバックメッセージ */}
        {msg && (
          <div className={`text-xs px-3 py-1.5 rounded mb-3 ${
            msg.ok ? "bg-emerald-900/40 text-emerald-300 border border-emerald-800/50"
                   : "bg-red-900/40 text-red-300 border border-red-800/50"
          }`}>
            {msg.text}
          </div>
        )}

        {/* ボタン */}
        <button onClick={handleSave} disabled={saving}
          className="w-full flex items-center justify-center gap-2 bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 text-white text-sm font-medium py-2 px-3 rounded mb-2 transition-colors">
          <span>{saving ? "保存中..." : "💾 ダウンロード"}</span>
        </button>
        {format !== "svg" && (
          <button onClick={handleCopy} disabled={copying}
            className="w-full flex items-center justify-center gap-2 bg-zinc-700 hover:bg-zinc-600 disabled:opacity-50 text-zinc-100 text-sm py-2 px-3 rounded transition-colors">
            <span>{copying ? "コピー中..." : "📋 クリップボードにコピー"}</span>
          </button>
        )}
      </Section>

      {/* ── セッション ── */}
      <Section title="セッション">
        <p className="text-xs text-zinc-500 mb-3">
          ファイルデータと全設定を <code className="text-indigo-400">.vsm_session</code> に保存し次回復元できます
        </p>
        <button onClick={async () => { const saved = await onSave(); if (saved) flash("セッションを保存しました", true); }}
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

// ── ログタブ ──────────────────────────
function LogTab({ entries }: { entries: FileEntry[] }) {
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const loaded = entries.filter((e) => e.result);

  if (loaded.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-xs text-zinc-600">ファイルを読み込むとログが表示されます</p>
      </div>
    );
  }

  const idx    = Math.min(selectedIndex, entries.length - 1);
  const target = entries[idx] ?? entries[0];

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="p-3 border-b border-zinc-800 shrink-0">
        <select value={idx}
          onChange={(e) => setSelectedIndex(Number(e.target.value))}
          className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1.5">
          {entries.map((e, i) => (
            <option key={i} value={i}>{e.legendName || e.file.name}</option>
          ))}
        </select>
      </div>

      {target.result?.metadata && Object.keys(target.result.metadata).length > 0 && (
        <div className="p-3 border-b border-zinc-800 shrink-0">
          <p className="text-xs font-semibold text-zinc-500 uppercase tracking-widest mb-2">測定情報</p>
          <dl className="space-y-0.5">
            {Object.entries(target.result.metadata).slice(0, 12).map(([k, v]) => (
              <div key={k} className="flex gap-2 text-xs">
                <dt className="text-zinc-500 shrink-0 w-24 truncate" title={k}>{k}</dt>
                <dd className="text-zinc-300 truncate" title={v}>{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-3">
        <p className="text-xs font-semibold text-zinc-500 uppercase tracking-widest mb-2">解析ログ</p>
        {target.result?.logs && target.result.logs.length > 0 ? (
          <pre className="text-xs text-zinc-400 font-mono leading-relaxed whitespace-pre-wrap">
            {target.result.logs.join("\n")}
          </pre>
        ) : target.error ? (
          <p className="text-xs text-red-400">{target.error}</p>
        ) : (
          <p className="text-xs text-zinc-600">ログなし</p>
        )}
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
  onApplyFirstToAll, onSaveSession, onLoadSession, onSaveGraph, onCopyGraph,
}: Props) {
  const [activeTab, setActiveTab] = useState<Tab>("analysis");

  const TABS: { id: Tab; label: string }[] = [
    { id: "analysis", label: "解析" },
    { id: "graph",    label: "グラフ" },
    { id: "save",     label: "保存" },
    { id: "log",      label: "ログ" },
  ];

  return (
    <aside className="shrink-0 bg-zinc-900 flex flex-col overflow-hidden" style={style}>

      <div className="flex border-b border-zinc-800 shrink-0">
        {TABS.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-2.5 text-xs font-medium transition-colors ${
              activeTab === tab.id
                ? "text-indigo-400 border-b-2 border-indigo-500"
                : "text-zinc-500 hover:text-zinc-300 border-b-2 border-transparent"
            }`}>
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
        <SaveTab
          onSave={onSaveSession} onLoad={onLoadSession} hasEntries={entries.length > 0}
          onSaveGraph={onSaveGraph} onCopyGraph={onCopyGraph}
        />
      )}
      {activeTab === "log"  && <LogTab entries={entries} />}
    </aside>
  );
}
