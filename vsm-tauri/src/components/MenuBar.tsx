import { useEffect, useRef, useState } from "react";
import type { UnitMode, GraphSettings } from "../App";
import type { ExportOptions } from "../utils/graphExport";

interface Props {
  hasEntries:      boolean;
  unitMode:        UnitMode;
  graphSettings:   GraphSettings;
  onOpenFiles:     (files: File[]) => void;
  onAddFiles:      (files: File[]) => void;
  onClearAll:      () => void;
  onSaveSession:   () => Promise<boolean>;
  onLoadSession:   (file: File) => void;
  onUnitMode:      (m: UnitMode) => void;
  onGraphSettings: (p: Partial<GraphSettings>) => void;
  onSaveGraph:     (opts: ExportOptions) => Promise<boolean>;
  onCopyGraph:     (scale: number) => Promise<void>;
}

// ── ドロップダウンアイテム型 ────────────────────────────────
type MenuSep  = { kind: "sep" };
type MenuAct  = { kind: "act"; label: string; disabled?: boolean; onClick: () => void; shortcut?: string };
type MenuCheck = { kind: "check"; label: string; checked: boolean; onClick: () => void };
type MenuRadio = { kind: "radio"; label: string; checked: boolean; onClick: () => void };
type MenuHead  = { kind: "head"; label: string };
type MenuItem = MenuSep | MenuAct | MenuCheck | MenuRadio | MenuHead;

// ── ドロップダウン ──────────────────────────────────────────
function Dropdown({ items, onClose }: { items: MenuItem[]; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  return (
    <div ref={ref}
      className="absolute top-full left-0 mt-0.5 min-w-55 bg-zinc-800 border border-zinc-700 rounded shadow-2xl z-50 py-1 select-none">
      {items.map((item, i) => {
        if (item.kind === "sep")  return <div key={i} className="my-1 border-t border-zinc-700" />;
        if (item.kind === "head") return (
          <div key={i} className="px-3 py-1 text-[10px] text-zinc-600 uppercase tracking-wider font-semibold">{item.label}</div>
        );
        const baseClass = "w-full text-left flex items-center gap-2 px-3 py-1.5 text-sm transition-colors";
        if (item.kind === "act") return (
          <button key={i} disabled={item.disabled}
            className={`${baseClass} ${item.disabled ? "text-zinc-600 cursor-not-allowed" : "text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100"}`}
            onClick={() => { if (!item.disabled) { item.onClick(); onClose(); } }}>
            <span className="flex-1">{item.label}</span>
            {item.shortcut && <span className="text-zinc-600 text-xs">{item.shortcut}</span>}
          </button>
        );
        if (item.kind === "check") return (
          <button key={i} className={`${baseClass} text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100`}
            onClick={() => { item.onClick(); onClose(); }}>
            <span className="w-4 text-center text-indigo-400">{item.checked ? "✓" : ""}</span>
            {item.label}
          </button>
        );
        if (item.kind === "radio") return (
          <button key={i} className={`${baseClass} text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100`}
            onClick={() => { item.onClick(); onClose(); }}>
            <span className="w-4 text-center text-indigo-400">{item.checked ? "●" : "○"}</span>
            {item.label}
          </button>
        );
        return null;
      })}
    </div>
  );
}

// ── メニューボタン ──────────────────────────────────────────
function MenuButton({ label, items }: { label: string; items: MenuItem[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        className={`px-3 py-1 text-sm rounded transition-colors ${
          open ? "bg-zinc-700 text-zinc-100" : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
        }`}
        onClick={() => setOpen((v) => !v)}
      >
        {label}
      </button>
      {open && <Dropdown items={items} onClose={() => setOpen(false)} />}
    </div>
  );
}

// ── メインメニューバー ──────────────────────────────────────
export default function MenuBar({
  hasEntries, unitMode, graphSettings,
  onOpenFiles, onAddFiles, onClearAll,
  onSaveSession, onLoadSession,
  onUnitMode, onGraphSettings,
  onSaveGraph, onCopyGraph,
}: Props) {
  const openRef   = useRef<HTMLInputElement>(null);
  const addRef    = useRef<HTMLInputElement>(null);
  const sessionRef = useRef<HTMLInputElement>(null);

  const pick = (ref: React.RefObject<HTMLInputElement | null>, cb: (f: File[]) => void) => {
    ref.current?.click();
    const handler = () => {
      const files = Array.from(ref.current?.files ?? []);
      if (files.length) cb(files);
      ref.current!.value = "";
      ref.current!.removeEventListener("change", handler);
    };
    ref.current?.addEventListener("change", handler);
  };

  const fileMenu: MenuItem[] = [
    { kind: "act",  label: "開く...",            onClick: () => pick(openRef, onOpenFiles) },
    { kind: "act",  label: "ファイルを追加...",   onClick: () => pick(addRef, onAddFiles) },
    { kind: "sep" },
    { kind: "act",  label: "セッションを保存...", disabled: !hasEntries, onClick: onSaveSession },
    { kind: "act",  label: "セッションを読み込み...", onClick: () => pick(sessionRef, (f) => onLoadSession(f[0])) },
    { kind: "sep" },
    { kind: "act",  label: "すべてクリア",        disabled: !hasEntries, onClick: onClearAll },
  ];

  const viewMenu: MenuItem[] = [
    { kind: "head",  label: "単位" },
    { kind: "radio", label: "SI（μ₀H / kA/m）",      checked: unitMode === "SI",         onClick: () => onUnitMode("SI") },
    { kind: "radio", label: "CGS（Oe / emu/cm³）",    checked: unitMode === "CGS",        onClick: () => onUnitMode("CGS") },
    { kind: "radio", label: "規格化（μ₀H / M/Ms）",   checked: unitMode === "Normalized", onClick: () => onUnitMode("Normalized") },
    { kind: "sep" },
    { kind: "check", label: "論文モード",     checked: graphSettings.paperMode,     onClick: () => onGraphSettings({ paperMode: !graphSettings.paperMode }) },
    { kind: "check", label: "凡例を表示",     checked: graphSettings.showLegend,    onClick: () => onGraphSettings({ showLegend: !graphSettings.showLegend }) },
    { kind: "check", label: "グリッド線",     checked: graphSettings.showGrid,      onClick: () => onGraphSettings({ showGrid: !graphSettings.showGrid }) },
    { kind: "check", label: "原点線",         checked: graphSettings.showZeroLines, onClick: () => onGraphSettings({ showZeroLines: !graphSettings.showZeroLines }) },
  ];

  const graphMenu: MenuItem[] = [
    { kind: "act", label: "PNG でダウンロード（×2）",
      onClick: () => onSaveGraph({ format: "png", scale: 2, useCustomSize: false, width: 0, height: 0 }) },
    { kind: "act", label: "SVG でダウンロード",
      onClick: () => onSaveGraph({ format: "svg", scale: 1, useCustomSize: false, width: 0, height: 0 }) },
    { kind: "sep" },
    { kind: "act", label: "クリップボードにコピー（×2）", onClick: () => onCopyGraph(2) },
  ];

  return (
    <div className="h-8 shrink-0 bg-zinc-900 border-b border-zinc-800 flex items-center px-2 gap-0.5 select-none">
      {/* 隠し input */}
      <input ref={openRef}   type="file" accept=".VSM,.vsm,.dat" multiple className="hidden" />
      <input ref={addRef}    type="file" accept=".VSM,.vsm,.dat" multiple className="hidden" />
      <input ref={sessionRef} type="file" accept=".vsm_session"  className="hidden" />

      <MenuButton label="ファイル" items={fileMenu} />
      <MenuButton label="表示"     items={viewMenu} />
      <MenuButton label="グラフ"   items={graphMenu} />
    </div>
  );
}
