import { useRef, useState } from "react";
import type { FileEntry, UnitMode, GraphSettings } from "../App";
import { FILE_COLORS } from "../App";
import type { AnalysisParams } from "../api/client";

interface Props {
  entries:               FileEntry[];
  params:                AnalysisParams;
  unitMode:              UnitMode;
  graphSettings:         GraphSettings;
  onLoadFiles:           (files: File[]) => void;
  onAddFiles:            (files: File[]) => void;
  onClearAll:            () => void;
  onParamsChange:        (next: Partial<AnalysisParams>) => void;
  onUnitModeChange:      (mode: UnitMode) => void;
  onGraphSettingsChange: (next: Partial<GraphSettings>) => void;
  onEntryColorChange:    (index: number, color: string) => void;
  onEntryRemove:         (index: number) => void;
  onEntrySettingsChange: (index: number, patch: { thickness?: number; area?: number }) => void;
}

type Tab = "analysis" | "graph" | "save" | "log";

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

// ── 解析タブ ──────────────────────────
function AnalysisTab({ entries, params, unitMode, onLoadFiles, onAddFiles, onClearAll,
  onParamsChange, onUnitModeChange, onEntryColorChange, onEntryRemove, onEntrySettingsChange,
  newRef, addRef, pickFiles }: {
  entries: FileEntry[]; params: AnalysisParams; unitMode: UnitMode;
  onLoadFiles: (f: File[]) => void; onAddFiles: (f: File[]) => void; onClearAll: () => void;
  onParamsChange: (n: Partial<AnalysisParams>) => void; onUnitModeChange: (m: UnitMode) => void;
  onEntryColorChange: (i: number, c: string) => void; onEntryRemove: (i: number) => void;
  onEntrySettingsChange: (i: number, p: { thickness?: number; area?: number }) => void;
  newRef: React.RefObject<HTMLInputElement | null>; addRef: React.RefObject<HTMLInputElement | null>;
  pickFiles: (ref: React.RefObject<HTMLInputElement | null>, handler: (f: File[]) => void) => void;
}) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  return (
    <div className="flex-1 overflow-y-auto">
      <Section title="ファイル">
        <button onClick={() => pickFiles(newRef, onLoadFiles)}
          className="w-full bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium py-2 px-3 rounded mb-2 transition-colors">
          ファイルを選択 (新規)
        </button>
        <button onClick={() => pickFiles(addRef, onAddFiles)}
          className="w-full bg-zinc-700 hover:bg-zinc-600 text-zinc-100 text-sm py-2 px-3 rounded mb-2 transition-colors">
          ファイルを追加...
        </button>
        <button onClick={onClearAll}
          className="w-full bg-zinc-800 hover:bg-red-900/50 text-zinc-400 hover:text-red-300 text-sm py-2 px-3 rounded transition-colors">
          全て削除
        </button>
      </Section>

      {/* ファイルリスト（色・個別設定） */}
      {entries.length > 0 && (
        <Section title="読み込み済み">
          <ul className="space-y-1">
            {entries.map((e, i) => (
              <li key={i} className="rounded bg-zinc-800/50">
                {/* メイン行 */}
                <div className="flex items-center gap-2 px-2 py-1.5">
                  <input type="color" value={e.color}
                    onChange={(ev) => onEntryColorChange(i, ev.target.value)}
                    className="w-5 h-5 rounded cursor-pointer border-0 bg-transparent shrink-0"
                    title="色を変更"
                  />
                  <span className={`text-xs truncate flex-1 ${e.error ? "text-red-400" : e.loading ? "text-zinc-500" : "text-zinc-300"}`}
                    title={e.file.name}>
                    {e.loading ? "⏳" : e.error ? "✗" : "✓"} {e.file.name}
                  </span>
                  {/* 個別設定展開ボタン */}
                  <button
                    onClick={() => setExpandedIndex(expandedIndex === i ? null : i)}
                    className={`text-xs px-1 shrink-0 transition-colors ${expandedIndex === i ? "text-indigo-400" : "text-zinc-600 hover:text-zinc-300"}`}
                    title="ファイル別設定">⚙
                  </button>
                  <button onClick={() => onEntryRemove(i)}
                    className="text-zinc-600 hover:text-red-400 text-xs shrink-0 transition-colors"
                    title="削除">✕
                  </button>
                </div>

                {/* 個別設定パネル */}
                {expandedIndex === i && (
                  <div className="px-3 pb-3 border-t border-zinc-700 mt-1 pt-2 space-y-2">
                    <p className="text-xs text-zinc-500 mb-2">
                      空欄 = グローバル値 ({params.thickness} nm / {params.area} mm²) を使用
                    </p>
                    <div>
                      <label className="text-xs text-zinc-400 block mb-1">膜厚 (nm)</label>
                      <input type="number" min={0.1} step={1}
                        placeholder={String(params.thickness)}
                        value={e.thickness ?? ""}
                        onChange={(ev) => {
                          const v = parseFloat(ev.target.value);
                          onEntrySettingsChange(i, { thickness: isNaN(v) ? undefined : v });
                        }}
                        className="w-full bg-zinc-700 border border-zinc-600 text-zinc-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-zinc-400 block mb-1">面積 (mm²)</label>
                      <input type="number" min={0.1} step={1}
                        placeholder={String(params.area)}
                        value={e.area ?? ""}
                        onChange={(ev) => {
                          const v = parseFloat(ev.target.value);
                          onEntrySettingsChange(i, { area: isNaN(v) ? undefined : v });
                        }}
                        className="w-full bg-zinc-700 border border-zinc-600 text-zinc-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {/* グローバル設定 */}
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
        <Select label="反磁性補正" value={params.demagMode}
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

      {/* 表示ON/OFF */}
      <Section title="表示">
        <Toggle label="凡例を表示"      checked={graphSettings.showLegend}    onChange={(v) => onChange({ showLegend: v })} />
        <Toggle label="グリッド線を表示" checked={graphSettings.showGrid}      onChange={(v) => onChange({ showGrid: v })} />
        <Toggle label="原点線を表示"    checked={graphSettings.showZeroLines} onChange={(v) => onChange({ showZeroLines: v })} />
      </Section>

      {/* 凡例 */}
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

      {/* 軸ラベル */}
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

      {/* 描画範囲 */}
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

      {/* プロット */}
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

      {/* 目盛り書式 */}
      <Section title="目盛り書式（有効数字）">
        <Select label="X 軸" value={graphSettings.xTickFormat}
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
        <Select label="Y 軸" value={graphSettings.yTickFormat}
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

      {/* 原点線・グリッドスタイル */}
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
function SaveTab() {
  return (
    <div className="flex-1 overflow-y-auto">
      <Section title="グラフ保存">
        <p className="text-xs text-zinc-500 mb-3">グラフ右上の 📷 アイコンから PNG / SVG で保存できます</p>
      </Section>
      <Section title="セッション">
        <button disabled className="w-full bg-zinc-700 text-zinc-500 text-sm py-2 px-3 rounded mb-2 cursor-not-allowed">
          セッションを保存... (未実装)
        </button>
        <button disabled className="w-full bg-zinc-700 text-zinc-500 text-sm py-2 px-3 rounded cursor-not-allowed">
          セッションを読み込み... (未実装)
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

  const target = entries[selectedIndex] ?? entries[0];

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* ファイル選択 */}
      <div className="p-3 border-b border-zinc-800 shrink-0">
        <select
          value={selectedIndex}
          onChange={(e) => setSelectedIndex(Number(e.target.value))}
          className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1.5"
        >
          {entries.map((e, i) => (
            <option key={i} value={i}>{e.file.name}</option>
          ))}
        </select>
      </div>

      {/* メタデータ */}
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

      {/* ログ出力 */}
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
  entries, params, unitMode, graphSettings,
  onLoadFiles, onAddFiles, onClearAll,
  onParamsChange, onUnitModeChange, onGraphSettingsChange,
  onEntryColorChange, onEntryRemove, onEntrySettingsChange,
}: Props) {
  const [activeTab, setActiveTab] = useState<Tab>("analysis");
  const newRef = useRef<HTMLInputElement>(null);
  const addRef = useRef<HTMLInputElement>(null);

  const pickFiles = (ref: React.RefObject<HTMLInputElement | null>, handler: (f: File[]) => void) => {
    ref.current?.click();
    const onChange = () => {
      const files = Array.from(ref.current?.files ?? []);
      if (files.length) handler(files);
      ref.current!.value = "";
      ref.current!.removeEventListener("change", onChange);
    };
    ref.current?.addEventListener("change", onChange);
  };

  const TABS: { id: Tab; label: string }[] = [
    { id: "analysis", label: "解析" },
    { id: "graph",    label: "グラフ" },
    { id: "save",     label: "保存" },
    { id: "log",      label: "ログ" },
  ];

  return (
    <aside className="w-72 min-w-72 bg-zinc-900 border-r border-zinc-800 flex flex-col overflow-hidden">
      <input ref={newRef} type="file" accept=".VSM,.vsm,.dat" multiple className="hidden" />
      <input ref={addRef} type="file" accept=".VSM,.vsm,.dat" multiple className="hidden" />

      {/* タブヘッダー */}
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
          onEntryColorChange={onEntryColorChange} onEntryRemove={onEntryRemove}
          onEntrySettingsChange={onEntrySettingsChange}
          newRef={newRef} addRef={addRef} pickFiles={pickFiles}
        />
      )}
      {activeTab === "graph" && (
        <GraphTab graphSettings={graphSettings} onChange={onGraphSettingsChange} />
      )}
      {activeTab === "save" && <SaveTab />}
      {activeTab === "log"  && <LogTab entries={entries} />}
    </aside>
  );
}
