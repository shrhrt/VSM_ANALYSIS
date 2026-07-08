import { memo, useCallback, useEffect, useRef, useState } from "react";
import type { FileEntry } from "../App";
import { pickText, rollRarity } from "../data/trivia";

interface Props {
  entries:       FileEntry[];
  backendStatus: "starting" | "ready" | "error";
  diagLog?:      string[];
}

// ── ファイルカラードット ──────────────────────────────────────
const MAX_DOTS = 8;

function FileColorDots({ entries }: { entries: FileEntry[] }) {
  const visible = entries.slice(0, MAX_DOTS);
  const overflow = entries.length - MAX_DOTS;

  return (
    <span className="shrink-0 flex items-center gap-1.5" style={{ width: 96 }}>
      {visible.length === 0
        ? Array(MAX_DOTS).fill(0).map((_, i) => (
            <span key={i} className="w-2 h-2 rounded-sm bg-zinc-800 border border-zinc-700/40 shrink-0" />
          ))
        : <>
            {visible.map((e, i) => (
              <span
                key={i}
                className="shrink-0 w-2.5 h-2.5 rounded-sm"
                title={e.file?.name ?? ""}
                style={{
                  background:  e.error   ? "#52525b" : e.color,
                  opacity:     e.loading ? 0.35       : e.error ? 0.4 : 1,
                  boxShadow:   !e.error && !e.loading && e.result
                    ? `0 0 5px ${e.color}99` : undefined,
                }}
              />
            ))}
            {overflow > 0 && (
              <span className="text-[9px] text-zinc-600 font-mono leading-none shrink-0">
                +{overflow}
              </span>
            )}
          </>
      }
    </span>
  );
}

function Pipe() {
  return <span className="text-indigo-900/80 mx-2 text-xs">│</span>;
}

// ── 超超激レア演出オーバーレイ ─────────────────────────────
function SssrOverlay({ onDone }: { onDone: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDone, 8000);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const rings = [
    { color: "200,80,255",  delay: "0.1s" },
    { color: "80,200,255",  delay: "0.45s" },
    { color: "255,200,60",  delay: "0.8s" },
    { color: "255,80,140",  delay: "1.15s" },
    { color: "80,255,180",  delay: "1.5s" },
    { color: "255,140,40",  delay: "1.85s" },
  ];
  const streaks = [
    { top: "12%", rotate: "-13deg", delay: "0.25s", color: "200,80,255" },
    { top: "34%", rotate:  "9deg",  delay: "0.55s", color: "80,200,255" },
    { top: "58%", rotate: "-6deg",  delay: "0.85s", color: "255,200,60" },
    { top: "78%", rotate:  "14deg", delay: "1.1s",  color: "255,80,140" },
  ];

  return (
    <div className="fixed inset-0 pointer-events-none z-50 vsm-sssr-shake">
      {/* 白フラッシュ */}
      <div className="vsm-sssr-initial-flash absolute inset-0 bg-white" />

      {/* 暗転背景 */}
      <div className="vsm-sssr-blackout absolute inset-0 bg-black"
        style={{ backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(255,255,255,0.012) 3px, rgba(255,255,255,0.012) 4px)" }} />

      {/* 稲妻ストリーク */}
      {streaks.map((s, i) => (
        <div key={i} style={{
          position: "absolute", top: s.top, left: 0, right: 0,
          height: 2,
          background: `linear-gradient(90deg, transparent 0%, rgba(${s.color},0.9) 20%, white 50%, rgba(${s.color},0.9) 80%, transparent 100%)`,
          boxShadow: `0 0 16px 4px rgba(${s.color},0.7), 0 0 40px rgba(${s.color},0.3)`,
          transform: `rotate(${s.rotate})`,
          animation: `vsm-streak 2s ${s.delay} ease-out forwards`,
        }} />
      ))}

      {/* コニックビーム（大・中の2枚重ね） */}
      {[{ size: 600, speed: "8s" }, { size: 400, speed: "6s" }].map(({ size, speed }, i) => (
        <div key={i} style={{
          position: "absolute", top: "50%", left: "50%",
          width: size, height: size, marginLeft: -size / 2, marginTop: -size / 2,
          borderRadius: "50%",
          background: "conic-gradient(rgba(200,80,255,0.4) 0deg, transparent 20deg, rgba(80,200,255,0.4) 60deg, transparent 80deg, rgba(255,200,60,0.4) 120deg, transparent 140deg, rgba(255,80,140,0.4) 180deg, transparent 200deg, rgba(80,255,180,0.4) 240deg, transparent 260deg, rgba(255,140,40,0.4) 300deg, transparent 320deg)",
          animation: `vsm-sssr-beams ${speed} ease-out forwards`,
          animationDelay: `${i * 0.3}s`,
        }} />
      ))}

      {/* 拡張リング */}
      {rings.map((r, i) => (
        <div key={i} style={{
          position: "absolute", top: "50%", left: "50%",
          width: 80, height: 80, marginLeft: -40, marginTop: -40,
          borderRadius: "50%",
          border: `2px solid rgba(${r.color},1)`,
          boxShadow: `0 0 24px rgba(${r.color},0.7), 0 0 60px rgba(${r.color},0.3)`,
          animation: `vsm-sssr-ring-expand 2.6s ${r.delay} ease-out forwards`,
        }} />
      ))}

      {/* 中央✦スター */}
      <div style={{
        position: "absolute", top: "50%", left: "50%",
        fontSize: 88, lineHeight: 1,
        animation: "vsm-sssr-star 8s ease-in-out forwards",
      }} className="vsm-rainbow-text">✦</div>

      {/* 超超激レア！ バッジ */}
      <div style={{
        position: "absolute", top: "calc(50% + 72px)", left: "50%",
        fontSize: 22, fontWeight: 900, letterSpacing: "0.2em",
        fontFamily: "system-ui, sans-serif",
        animation: "vsm-sssr-badge 8s ease-out forwards",
        whiteSpace: "nowrap",
      }} className="vsm-rainbow-text">超超激レア！！！</div>

      {/* 四隅グロー */}
      {[
        { top: 0,    left: 0,    transform: "" },
        { top: 0,    right: 0,   transform: "" },
        { bottom: 0, left: 0,    transform: "" },
        { bottom: 0, right: 0,   transform: "" },
      ].map((pos, i) => (
        <div key={i} style={{
          position: "absolute", ...pos,
          width: 200, height: 200,
          background: `radial-gradient(ellipse at ${i < 2 ? "top" : "bottom"} ${i % 2 === 0 ? "left" : "right"}, rgba(200,80,255,0.35) 0%, transparent 65%)`,
          animation: "vsm-sssr-corner-glow 8s ease-out forwards",
          animationDelay: `${i * 0.15}s`,
        }} />
      ))}
    </div>
  );
}

// ── ガチャ演出の有効/無効切り替え ─────────────────────────
// false にすると豆知識だけが表示され、レア演出は一切発生しない。
// (演出ロジック自体は下に残したままにしてあるので、true に戻せば復活する)
const GACHA_ENABLED = false;

// ── 豆知識表示（props なし、親の再レンダリングから完全隔離）────────
const TriviaDisplay = memo(function TriviaDisplay() {
  const [text,         setText]      = useState(() => pickText());
  const [visible,      setVisible]   = useState(true);
  const [showSssr,     setShowSssr]  = useState(false);
  const [textClass,    setTextClass] = useState("text-zinc-500");

  const textRef    = useRef(text);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const changeFnRef = useRef<() => void>(null!);

  useEffect(() => { textRef.current = text; }, [text]);

  const changeFact = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);

    setVisible(false);
    setTimeout(() => {
      const nextText   = pickText(textRef.current);
      const nextRarity = GACHA_ENABLED ? rollRarity() : "normal";
      setText(nextText);
      setTextClass(nextRarity === "sssr" ? "vsm-rainbow-text" : "text-zinc-500");
      setVisible(true);
      if (nextRarity === "sssr") setShowSssr(true);
      intervalRef.current = setInterval(() => changeFnRef.current(), 10_000);
    }, 350);
  }, []);

  changeFnRef.current = changeFact;

  useEffect(() => {
    intervalRef.current = setInterval(() => changeFnRef.current(), 10_000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, []);

  return (
    <>
      {showSssr && <SssrOverlay onDone={() => setShowSssr(false)} />}
      <span className={`flex items-center flex-1 min-w-0 transition-opacity duration-300 ${visible ? "opacity-100" : "opacity-0"}`}>
        <span className={`text-[10px] font-bold truncate ${textClass}`}>{text}</span>
      </span>
    </>
  );
});

// ── メインコンポーネント ──────────────────────────────────────
export default function StatusBar({ entries, backendStatus, diagLog = [] }: Props) {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString("ja-JP", { hour12: false }));
  const [showDiag, setShowDiag] = useState(false);

  useEffect(() => {
    const t = setInterval(() => setTime(new Date().toLocaleTimeString("ja-JP", { hour12: false })), 1000);
    return () => clearInterval(t);
  }, []);

  // エラー時は自動で診断ログを開く
  useEffect(() => {
    if (backendStatus === "error") setShowDiag(true);
  }, [backendStatus]);

  const total   = entries.length;
  const errored = entries.filter(e => e.error && !e.loading).length;

  const statusColor =
    backendStatus === "ready" ? "bg-indigo-500" :
    backendStatus === "error" ? "bg-red-600"    : "bg-amber-500 animate-pulse";

  return (
    <>
      {/* 診断ログパネル (エラー時に自動展開、クリックでトグル) */}
      {showDiag && (
        <div className="bg-zinc-900 border-t border-red-900/60 px-3 py-2 text-[10px] font-mono max-h-40 overflow-y-auto">
          <div className="flex justify-between items-center mb-1">
            <span className="text-red-400 font-bold tracking-wider">— 診断ログ —</span>
            <button
              className="text-zinc-500 hover:text-zinc-300 text-[9px]"
              onClick={() => setShowDiag(false)}
            >✕ 閉じる</button>
          </div>
          {diagLog.length === 0
            ? <span className="text-zinc-600">（ログなし）</span>
            : diagLog.map((line, i) => (
                <div key={i} className={
                  line.includes("✓") ? "text-emerald-400" :
                  line.includes("✗") ? "text-red-400" : "text-zinc-400"
                }>{line}</div>
              ))
          }
          {errored > 0 && entries.filter(e => e.error).map((e, i) => (
            <div key={`err-${i}`} className="text-red-400">[ファイル] {e.file?.name}: {e.error}</div>
          ))}
        </div>
      )}

    <div className="h-7 shrink-0 bg-zinc-950 border-t border-indigo-950/80 flex items-center relative overflow-hidden select-none">
      {/* スキャンライン */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.025]"
        style={{ backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(120,120,255,0.3) 3px, rgba(120,120,255,0.3) 4px)" }} />

      {/* 左端アクセントバー */}
      <div className={`w-0.75 h-full shrink-0 ${statusColor}`} />

      {/* システム状態 (クリックで診断ログ表示) */}
      <button
        className={`font-mono text-[9px] tracking-[0.15em] font-bold px-3 shrink-0 cursor-pointer hover:opacity-70 ${
          backendStatus === "ready" ? "text-indigo-400" :
          backendStatus === "error" ? "text-red-400"    : "text-amber-400"
        }`}
        onClick={() => setShowDiag(v => !v)}
        title="クリックで診断ログを表示"
      >
        {backendStatus === "ready" ? "SYS:ONLINE" :
         backendStatus === "error" ? "SYS:OFFLINE [診断]" : "SYS:INIT"}
      </button>

      <Pipe />

      {/* ファイル数 */}
      <span className="flex items-center gap-2 shrink-0">
        <span className="text-indigo-400/70 text-[10px]">◈</span>
        <span className={`font-mono text-[11px] font-bold ${total > 0 ? "text-cyan-300" : "text-zinc-600"}`}>{total}</span>
        <span className="text-zinc-600 text-[9px] tracking-widest uppercase">files</span>
        {errored > 0 && (
          <span className="font-mono text-[9px] tracking-wider font-semibold text-red-400">{errored} ERR</span>
        )}
      </span>

      <Pipe />

      {/* ファイルカラードット */}
      <FileColorDots entries={entries} />

      <Pipe />

      {/* 豆知識 */}
      <div className="flex-1 min-w-0 flex items-center overflow-hidden">
        <TriviaDisplay />
      </div>

      {/* 時刻 */}
      <span className="flex items-center gap-1.5 px-3 shrink-0">
        <span className="text-indigo-900 text-[9px]">⊙</span>
        <span className="font-mono text-[10px] text-zinc-600 tracking-wider tabular-nums">{time}</span>
      </span>

      {/* 右端アクセントバー */}
      <div className="w-0.75 h-full shrink-0 bg-indigo-900/60" />
    </div>
    </>
  );
}
