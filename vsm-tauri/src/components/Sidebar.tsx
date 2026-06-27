import { useRef } from "react";
import type { FileEntry, UnitMode } from "../App";
import type { AnalysisParams } from "../api/client";

interface Props {
  entries:          FileEntry[];
  params:           AnalysisParams;
  unitMode:         UnitMode;
  onLoadFiles:      (files: File[]) => void;
  onAddFiles:       (files: File[]) => void;
  onClearAll:       () => void;
  onParamsChange:   (next: Partial<AnalysisParams>) => void;
  onUnitModeChange: (mode: UnitMode) => void;
}

function NumberInput({
  label, value, step = 1, min,
  onChange,
}: {
  label: string; value: number; step?: number; min?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-3">
      <label className="text-xs text-zinc-400 block mb-1">{label}</label>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!isNaN(v) && (min === undefined || v >= min)) onChange(v);
        }}
        className="w-full bg-zinc-800 border border-zinc-600 text-zinc-100 text-sm rounded-md px-2 py-1.5"
      />
    </div>
  );
}

export default function Sidebar({
  entries, params, unitMode,
  onLoadFiles, onAddFiles, onClearAll,
  onParamsChange, onUnitModeChange,
}: Props) {
  const newRef = useRef<HTMLInputElement>(null);
  const addRef = useRef<HTMLInputElement>(null);

  const pickFiles = (
    ref: React.RefObject<HTMLInputElement | null>,
    handler: (f: File[]) => void
  ) => {
    ref.current?.click();
    const onChange = () => {
      const files = Array.from(ref.current?.files ?? []);
      if (files.length) handler(files);
      ref.current!.value = "";
      ref.current!.removeEventListener("change", onChange);
    };
    ref.current?.addEventListener("change", onChange);
  };

  return (
    <aside className="w-72 min-w-72 bg-zinc-900 border-r border-zinc-700 flex flex-col overflow-y-auto">
      <input ref={newRef} type="file" accept=".VSM,.vsm,.dat" multiple className="hidden" />
      <input ref={addRef} type="file" accept=".VSM,.vsm,.dat" multiple className="hidden" />

      {/* ファイル操作 */}
      <section className="p-4 border-b border-zinc-700">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-3">ファイル</h2>
        <button onClick={() => pickFiles(newRef, onLoadFiles)}
          className="w-full bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium py-2 px-3 rounded-md mb-2 transition-colors">
          ファイルを選択 (新規)
        </button>
        <button onClick={() => pickFiles(addRef, onAddFiles)}
          className="w-full bg-zinc-700 hover:bg-zinc-600 text-zinc-100 text-sm py-2 px-3 rounded-md mb-2 transition-colors">
          ファイルを追加...
        </button>
        <button onClick={onClearAll}
          className="w-full bg-zinc-800 hover:bg-zinc-700 text-zinc-400 text-sm py-2 px-3 rounded-md transition-colors">
          全て削除
        </button>
      </section>

      {/* ファイルリスト */}
      {entries.length > 0 && (
        <section className="p-4 border-b border-zinc-700">
          <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">読み込み済み</h2>
          <ul className="space-y-1">
            {entries.map((e, i) => (
              <li key={i} className="text-xs text-zinc-300 truncate flex items-center gap-2">
                {e.loading
                  ? <span className="text-zinc-500">⏳</span>
                  : e.error
                    ? <span className="text-red-400" title={e.error}>✗</span>
                    : <span className="text-green-400">✓</span>}
                {e.file.name}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* サンプル情報 */}
      <section className="p-4 border-b border-zinc-700">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-3">サンプル情報</h2>
        <NumberInput
          label="膜厚 (nm)"
          value={params.thickness}
          step={1}
          min={0.1}
          onChange={(v) => onParamsChange({ thickness: v })}
        />
        <NumberInput
          label="面積 (mm²)"
          value={params.area}
          step={1}
          min={0.1}
          onChange={(v) => onParamsChange({ area: v })}
        />
      </section>

      {/* 解析設定 */}
      <section className="p-4 border-b border-zinc-700">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-3">解析設定</h2>

        {/* 表示単位系 */}
        <div className="mb-3">
          <label className="text-xs text-zinc-400 block mb-1">表示単位系</label>
          <select
            value={unitMode}
            onChange={(e) => onUnitModeChange(e.target.value as UnitMode)}
            className="w-full bg-zinc-800 border border-zinc-600 text-zinc-100 text-sm rounded-md px-2 py-1.5"
          >
            <option value="SI">SI (T, kA/m)</option>
            <option value="CGS">CGS (Oe, emu/cm³)</option>
            <option value="Normalized">Normalized (T, M/Ms)</option>
          </select>
        </div>

        {/* 反磁性補正 */}
        <div className="mb-3">
          <label className="text-xs text-zinc-400 block mb-1">反磁性補正</label>
          <select
            value={params.demagMode}
            onChange={(e) => onParamsChange({ demagMode: e.target.value as "auto" | "none" })}
            className="w-full bg-zinc-800 border border-zinc-600 text-zinc-100 text-sm rounded-md px-2 py-1.5"
          >
            <option value="auto">自動検出</option>
            <option value="none">なし</option>
          </select>
        </div>

        {/* 磁化オフセット補正 */}
        <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
          <input
            type="checkbox"
            checked={params.offsetCorrection}
            onChange={(e) => onParamsChange({ offsetCorrection: e.target.checked })}
            className="accent-indigo-500"
          />
          磁化オフセット補正
        </label>
      </section>

      {/* 飽和磁場 */}
      <section className="p-4">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-3">飽和磁場 (Hs)</h2>
        <NumberInput
          label="許容範囲 (%)"
          value={params.hsTolerance}
          step={0.5}
          min={0.1}
          onChange={(v) => onParamsChange({ hsTolerance: v })}
        />
        <NumberInput
          label="連続点数 (最小)"
          value={params.hsMinConsecutive}
          step={1}
          min={1}
          onChange={(v) => onParamsChange({ hsMinConsecutive: Math.round(v) })}
        />
      </section>
    </aside>
  );
}
