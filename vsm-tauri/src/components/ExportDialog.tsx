import { useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { downloadGraphImage, copyGraphToClipboard, getPreviewUrl } from "../utils/graphExport";
import type { ExportOptions } from "../utils/graphExport";

type Format = ExportOptions["format"];

interface Preset { label: string; w: number; h: number }

const PRESETS: Preset[] = [
  { label: "正方形 S  800 × 800",   w: 800,  h: 800  },
  { label: "正方形 M  1000 × 1000", w: 1000, h: 1000 },
  { label: "正方形 L  1200 × 1200", w: 1200, h: 1200 },
  { label: "横長 4:3  1067 × 800",  w: 1067, h: 800  },
  { label: "横長 3:2  1200 × 800",  w: 1200, h: 800  },
  { label: "縦長 2:3  800 × 1200",  w: 800,  h: 1200 },
  { label: "カスタム",              w: 0,    h: 0    },
];

// 300 dpi 基準の mm 換算
function mmAt300(px: number) {
  return (px / 300 * 25.4).toFixed(1);
}

// クリップボードコピー不可の形式
const NO_COPY_FORMATS: Format[] = ["svg", "pdf"];


export default function ExportDialog({ onClose }: { onClose: () => void }) {
  const [format,     setFormat]     = useState<Format>("pdf");
  const [presetIdx,  setPresetIdx]  = useState(1);
  const [customW,    setCustomW]    = useState(1000);
  const [customH,    setCustomH]    = useState(1000);
  const [lockAR,     setLockAR]     = useState(true);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [saving,     setSaving]     = useState(false);
  const [copying,    setCopying]    = useState(false);
  const [msg,        setMsg]        = useState<{ text: string; ok: boolean } | null>(null);

  const prevUrlRef = useRef<string | null>(null);
  const arRatioRef = useRef(1.0);

  const isCustom = PRESETS[presetIdx].w === 0;
  const outW = isCustom ? customW : PRESETS[presetIdx].w;
  const outH = isCustom ? customH : PRESETS[presetIdx].h;

  // プリセット選択
  const pickPreset = (idx: number) => {
    setPresetIdx(idx);
    if (PRESETS[idx].w > 0) {
      setCustomW(PRESETS[idx].w);
      setCustomH(PRESETS[idx].h);
      arRatioRef.current = PRESETS[idx].w / PRESETS[idx].h;
    }
  };

  // カスタムサイズ変更（アスペクト比ロック対応）
  const changeW = (v: number) => {
    const w = Math.max(100, v);
    setCustomW(w);
    if (lockAR) setCustomH(Math.max(100, Math.round(w / arRatioRef.current)));
    else arRatioRef.current = w / customH;
  };
  const changeH = (v: number) => {
    const h = Math.max(100, v);
    setCustomH(h);
    if (lockAR) setCustomW(Math.max(100, Math.round(h * arRatioRef.current)));
    else arRatioRef.current = customW / h;
  };

  // プレビュー生成（200ms デバウンス）— 全 .main-svg 合成 SVG
  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        const url = await getPreviewUrl(outW, outH);
        if (cancelled) { URL.revokeObjectURL(url); return; }
        if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
        prevUrlRef.current = url;
        setPreviewUrl(url);
      } catch { /* グラフ未描画時はプレビューなし */ }
    }, 200);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [outW, outH]);

  useEffect(() => () => { if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current); }, []);

  const flash = (text: string, ok: boolean) => {
    setMsg({ text, ok });
    setTimeout(() => setMsg(null), ok ? 3000 : 6000);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const saved = await downloadGraphImage({ format, scale: 1, useCustomSize: true, width: outW, height: outH });
      if (saved) flash("保存しました", true);
    } catch (e) {
      flash((e as Error).message ?? "保存に失敗しました", false);
    } finally { setSaving(false); }
  };

  const handleCopy = async () => {
    setCopying(true);
    try {
      await copyGraphToClipboard(1, outW, outH);
      flash("クリップボードにコピーしました", true);
    } catch (e) {
      flash((e as Error).message ?? "コピーに失敗しました", false);
    } finally { setCopying(false); }
  };

  const canCopy = !NO_COPY_FORMATS.includes(format);

  return createPortal(
    <div className="fixed inset-0 z-9999 flex items-center justify-center bg-black/75"
         onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl flex overflow-hidden"
           style={{ width: 840, height: 560 }}>

        {/* ── 左: 設定パネル ── */}
        <div className="w-64 shrink-0 border-r border-zinc-800 flex flex-col bg-zinc-950/60">
          {/* ヘッダー */}
          <div className="px-4 py-3 border-b border-zinc-800 flex items-center justify-between">
            <h2 className="text-sm font-bold text-zinc-100">グラフ画像の出力</h2>
            <button onClick={onClose} className="text-zinc-500 hover:text-zinc-200 text-xl leading-none">&times;</button>
          </div>

          {/* 設定スクロールエリア */}
          <div className="flex-1 overflow-y-auto p-4 space-y-5">

            {/* ファイル形式 */}
            <div>
              <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2">ファイル形式</p>
              <div className="grid grid-cols-2 gap-1">
                {(["pdf", "svg", "png", "jpeg"] as Format[]).map((f) => (
                  <button key={f} onClick={() => setFormat(f)}
                    className={`text-xs py-2 font-medium rounded transition-colors border ${
                      format === f
                        ? "bg-indigo-600 border-indigo-500 text-white"
                        : "bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700"
                    }`}>
                    {f.toUpperCase()}
                    {f === "pdf" && <span className="ml-1 text-[9px] opacity-70">推奨</span>}
                  </button>
                ))}
              </div>
            </div>

            {/* 出力サイズ */}
            <div>
              <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2">出力サイズ</p>
              <div className="space-y-1">
                {PRESETS.map((p, i) => (
                  <button key={i} onClick={() => pickPreset(i)}
                    className={`w-full text-left flex items-center gap-2 px-2.5 py-2 rounded text-xs transition-colors ${
                      presetIdx === i
                        ? "bg-indigo-700/40 text-indigo-200 ring-1 ring-indigo-600/50"
                        : "text-zinc-400 hover:bg-zinc-800/60"
                    }`}>
                    <span className={`w-3.5 h-3.5 rounded-full border-2 shrink-0 flex items-center justify-center ${
                      presetIdx === i ? "border-indigo-400" : "border-zinc-600"
                    }`}>
                      {presetIdx === i && <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 block" />}
                    </span>
                    <span className="flex-1 font-medium">{p.label}</span>
                    {p.w > 0 && (
                      <span className="text-[10px] text-zinc-600 font-mono shrink-0">
                        {mmAt300(p.w)}mm
                      </span>
                    )}
                  </button>
                ))}
              </div>

              {/* カスタムサイズ入力 */}
              {isCustom && (
                <div className="mt-3 space-y-2">
                  <div className="flex items-end gap-1.5">
                    <div className="flex-1">
                      <label className="text-[10px] text-zinc-500 block mb-1">幅 (px)</label>
                      <input type="number" value={customW} min={100} max={8000} step={10}
                        onChange={(e) => changeW(Number(e.target.value))}
                        className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-500" />
                    </div>
                    <button
                      onClick={() => {
                        if (!lockAR) arRatioRef.current = customW / customH;
                        setLockAR(!lockAR);
                      }}
                      title={lockAR ? "比率固定中（クリックで解除）" : "比率フリー（クリックで固定）"}
                      className={`mb-0.5 px-2 py-1.5 rounded text-sm transition-colors ${
                        lockAR ? "text-indigo-400 bg-indigo-900/30" : "text-zinc-500 hover:bg-zinc-700"
                      }`}>
                      {lockAR ? "🔒" : "🔓"}
                    </button>
                    <div className="flex-1">
                      <label className="text-[10px] text-zinc-500 block mb-1">高さ (px)</label>
                      <input type="number" value={customH} min={100} max={8000} step={10}
                        onChange={(e) => changeH(Number(e.target.value))}
                        className="w-full bg-zinc-800 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-500" />
                    </div>
                  </div>
                  <p className="text-[10px] text-zinc-600">≈ {mmAt300(customW)} × {mmAt300(customH)} mm</p>
                </div>
              )}
            </div>
          </div>

          {/* 保存ボタン */}
          <div className="p-4 border-t border-zinc-800 space-y-2">
            {msg && (
              <div className={`text-xs px-3 py-1.5 rounded text-center ${
                msg.ok ? "bg-emerald-900/40 text-emerald-300" : "bg-red-900/40 text-red-300"
              }`}>{msg.text}</div>
            )}
            <button onClick={handleSave} disabled={saving}
              className="w-full bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 text-white text-sm font-medium py-2 rounded transition-colors">
              {saving ? "変換中..." : "💾 ダウンロード"}
            </button>
            {canCopy && (
              <button onClick={handleCopy} disabled={copying}
                className="w-full bg-zinc-700 hover:bg-zinc-600 disabled:opacity-50 text-zinc-100 text-sm py-2 rounded transition-colors">
                {copying ? "コピー中..." : "📋 クリップボードにコピー"}
              </button>
            )}
          </div>
        </div>

        {/* ── 右: プレビューパネル ── */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="px-5 py-3 border-b border-zinc-800 flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold text-zinc-300">プレビュー</span>
              <span className="text-[10px] text-zinc-600 ml-3 font-mono">
                {outW} × {outH} px &nbsp;≈ {mmAt300(outW)} × {mmAt300(outH)} mm
              </span>
            </div>
          </div>

          {/* チェッカーボード背景でプレビュー境界を可視化 */}
          <div
            className="flex-1 flex items-center justify-center p-6 overflow-hidden"
            style={{
              background: "repeating-conic-gradient(#d4d4d4 0% 25%, #e8e8e8 0% 50%) 0 0 / 16px 16px",
            }}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="出力プレビュー"
                className="shadow-xl"
                style={{ maxWidth: "100%", maxHeight: "100%", display: "block" }}
              />
            ) : null}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
