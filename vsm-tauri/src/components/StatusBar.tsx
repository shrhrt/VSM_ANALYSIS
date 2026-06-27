import { memo, useCallback, useEffect, useRef, useState } from "react";
import type { FileEntry } from "../App";
import { pickText, rollRarity, type Rarity } from "../data/trivia";

interface Props {
  entries:       FileEntry[];
  backendStatus: "starting" | "ready" | "error";
}

// ── バーチャートビジュアライザー ──────────────────────────────
const BLOCKS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'] as const;
const VIZ_COLS = 16;

function WaveVisualizer() {
  const [bars, setBars] = useState<number[]>(() => Array(VIZ_COLS).fill(0));

  useEffect(() => {
    const t = setInterval(() => {
      setBars(prev => prev.map(h => {
        // 一定確率でランダムな高さにポップアップ
        if (Math.random() < 0.09) return 3 + Math.floor(Math.random() * 5);
        // それ以外は1ずつ減衰
        return Math.max(0, h - 1);
      }));
    }, 75);
    return () => clearInterval(t);
  }, []);

  return (
    <span
      className="shrink-0 inline-block font-mono text-[11px] leading-none select-none"
      style={{
        color: "#34d399",
        textShadow: "0 0 5px rgba(52,211,153,0.5)",
        width: 96,
        letterSpacing: "0.02em",
        overflow: "hidden",
      }}
    >
      {bars.map((b, i) => <span key={i}>{BLOCKS[b]}</span>)}
    </span>
  );
}

function Pipe() {
  return <span className="text-indigo-900/80 mx-2 text-xs">│</span>;
}

// ── レアリティ設定 ────────────────────────────────────────────
const RARITY_CONFIG: Record<Rarity, {
  label:      string;
  textClass:  string;
  badgeClass: string;
  iconClass:  string;
}> = {
  normal: { label: "",         textClass: "text-zinc-500",   badgeClass: "", iconClass: "text-indigo-600" },
  rare:   { label: "RARE",     textClass: "text-amber-300",  badgeClass: "text-amber-400 border-amber-600/50 bg-amber-950/60", iconClass: "text-amber-500" },
  sr:     { label: "激レア",   textClass: "text-violet-300", badgeClass: "text-violet-300 border-violet-500/50 bg-violet-950/60", iconClass: "text-violet-400" },
  ssr:    { label: "超激レア", textClass: "text-orange-300", badgeClass: "text-orange-300 border-orange-500/50 bg-orange-950/60", iconClass: "text-orange-400" },
  sssr:   { label: "超超激レア", textClass: "vsm-rainbow-text", badgeClass: "border-yellow-400/50 bg-black", iconClass: "text-yellow-300" },
};

// ── エフェクトオーバーレイ（テキストなし・ビジュアルのみ）──────
function RarityOverlay({ rarity, onDone }: { rarity: Rarity | null; onDone: () => void }) {
  if (!rarity || rarity === "normal") return null;

  const DURATIONS: Record<Exclude<Rarity, "normal">, number> = {
    rare: 1500, sr: 2500, ssr: 4000, sssr: 7000,
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    const t = setTimeout(onDone, DURATIONS[rarity as keyof typeof DURATIONS]);
    return () => clearTimeout(t);
  }, [rarity]);

  if (rarity === "rare") {
    return (
      <div className="fixed inset-0 pointer-events-none z-40">
        <div className="vsm-rare-flash absolute bottom-0 left-0 right-0 h-40"
          style={{ background: "radial-gradient(ellipse at 50% 100%, rgba(251,191,36,0.55) 0%, transparent 65%)" }} />
        {[8, 22, 37, 53, 68, 82].map((left, i) => (
          <span key={i} style={{
            position: "absolute", bottom: 28, left: `${left}%`,
            color: `rgba(251,${170 + i * 10},36,0.95)`,
            fontSize: 11 + (i % 3),
            animation: `vsm-particle-rise 1s ${i * 0.09}s ease-out forwards`,
          }}>✦</span>
        ))}
      </div>
    );
  }

  if (rarity === "sr") {
    return (
      <div className="fixed inset-0 pointer-events-none z-40">
        <div className="vsm-sr-dim absolute inset-0 bg-indigo-950/65" />
        <div className="vsm-sr-flash absolute inset-0"
          style={{ background: "radial-gradient(ellipse at 50% 100%, rgba(167,139,250,0.6) 0%, transparent 55%)" }} />
        <div className="vsm-sr-scanbeam" />
        <div className="vsm-sr-dim absolute inset-0" style={{
          background: "radial-gradient(ellipse 90% 80% at 50% 50%, transparent 40%, rgba(88,28,135,0.5) 100%)",
          animationDuration: "2.5s",
        }} />
      </div>
    );
  }

  if (rarity === "ssr") {
    const streaks = [
      { top: "18%", rotate: "-14deg", delay: "0.2s", r: "255", g: "150", b: "50" },
      { top: "55%", rotate:  "10deg", delay: "0.5s", r: "255", g: "100", b: "30" },
      { top: "35%", rotate:  "-4deg", delay: "0.8s", r: "255", g: "200", b: "80" },
    ];
    return (
      <div className="fixed inset-0 pointer-events-none z-40 vsm-shake">
        <div className="vsm-ssr-dim absolute inset-0 bg-zinc-950/88"
          style={{ backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,100,0,0.03) 2px, rgba(255,100,0,0.03) 4px)" }} />
        <div className="vsm-ssr-lightning absolute inset-0"
          style={{ background: "radial-gradient(ellipse at 50% 50%, rgba(255,150,50,0.75) 0%, transparent 50%)" }} />
        {streaks.map((s, i) => (
          <div key={i} style={{
            position: "absolute", top: s.top, left: 0, right: 0,
            height: i === 2 ? 1 : 2,
            background: `linear-gradient(90deg, transparent 0%, rgba(${s.r},${s.g},${s.b},0.9) 25%, white 50%, rgba(${s.r},${s.g},${s.b},0.9) 75%, transparent 100%)`,
            boxShadow: `0 0 12px 2px rgba(${s.r},${s.g},${s.b},0.6)`,
            transform: `rotate(${s.rotate})`,
            animation: `vsm-streak 1.6s ${s.delay} ease-out forwards`,
          }} />
        ))}
      </div>
    );
  }

  if (rarity === "sssr") {
    const rings = [
      { color: "200,100,255", delay: "0.2s" },
      { color: "100,200,255", delay: "0.55s" },
      { color: "255,200,80",  delay: "0.9s" },
    ];
    return (
      <div className="fixed inset-0 pointer-events-none z-50">
        <div className="vsm-sssr-blackout absolute inset-0 bg-black"
          style={{ backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(255,255,255,0.012) 3px, rgba(255,255,255,0.012) 4px)" }} />
        {/* コニックビーム回転 */}
        <div style={{
          position: "absolute", top: "50%", left: "50%",
          width: 400, height: 400, marginLeft: -200, marginTop: -200,
          borderRadius: "50%",
          background: "conic-gradient(rgba(200,100,255,0.35) 0deg, transparent 25deg, rgba(100,200,255,0.35) 60deg, transparent 85deg, rgba(255,200,80,0.35) 120deg, transparent 145deg, rgba(100,255,150,0.35) 180deg, transparent 205deg, rgba(255,100,100,0.35) 240deg, transparent 265deg, rgba(180,100,255,0.35) 300deg, transparent 325deg)",
          animation: "vsm-sssr-beams 7s ease-out forwards",
        }} />
        {/* 拡張リング */}
        {rings.map((r, i) => (
          <div key={i} style={{
            position: "absolute", top: "50%", left: "50%",
            width: 100, height: 100, marginLeft: -50, marginTop: -50,
            borderRadius: "50%",
            border: `${2 - i * 0.4}px solid rgba(${r.color},0.9)`,
            boxShadow: `0 0 20px rgba(${r.color},0.5)`,
            animation: `vsm-ring-expand 2.8s ${r.delay} ease-out forwards`,
          }} />
        ))}
        {/* 中央スター */}
        <div style={{
          position: "absolute", top: "50%", left: "50%",
          transform: "translate(-50%,-50%)",
          fontSize: 64, lineHeight: 1,
          animation: "vsm-sssr-star 7s ease-in-out forwards",
        }} className="vsm-rainbow-text">✦</div>
      </div>
    );
  }
  return null;
}

// ── ガチャボタン ────────────────────────────────────────────
function GachaButton({ onClick, disabled }: { onClick: () => void; disabled: boolean }) {
  const [spinning, setSpinning] = useState(false);

  const handleClick = () => {
    if (disabled || spinning) return;
    setSpinning(true);
    onClick();
    setTimeout(() => setSpinning(false), 400);
  };

  return (
    <button
      onClick={handleClick}
      disabled={spinning}
      className={`vsm-gacha-btn shrink-0 flex items-center justify-center w-5 h-5 rounded text-[11px] text-red-100 cursor-pointer select-none ${spinning ? "pointer-events-none" : ""}`}
      title="豆知識をロール"
    >
      {spinning && <div className="vsm-gacha-flash absolute inset-0 bg-red-200 pointer-events-none rounded" />}
      <span className={`leading-none ${spinning ? "vsm-gacha-rolling inline-block" : ""}`}>◈</span>
    </button>
  );
}

// ── 豆知識表示（props なし、親の再レンダリングから完全隔離）────────
const TriviaDisplay = memo(function TriviaDisplay() {
  const [text,         setText]      = useState(() => pickText());
  const [rarity,       setRarity]    = useState<Rarity>("normal");
  const [visible,      setVisible]   = useState(true);
  const [effectRarity, setEffectRarity] = useState<Rarity | null>(null);
  const [rolling,      setRolling]   = useState(false);

  const textRef    = useRef(text);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const changeFnRef = useRef<() => void>(null!);

  useEffect(() => { textRef.current = text; }, [text]);

  // ROLLしたらインターバルをリセット（連続発火を防ぐ）
  const changeFact = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);

    setRolling(true);
    setVisible(false);
    setTimeout(() => {
      const nextText   = pickText(textRef.current);
      const nextRarity = rollRarity();
      setText(nextText);
      setRarity(nextRarity);
      setVisible(true);
      setRolling(false);
      if (nextRarity !== "normal") {
        setEffectRarity(nextRarity);
      }
      // 変化後に新しい10秒タイマーをセット
      intervalRef.current = setInterval(() => changeFnRef.current(), 10_000);
    }, 350);
  }, []);

  changeFnRef.current = changeFact;

  useEffect(() => {
    intervalRef.current = setInterval(() => changeFnRef.current(), 10_000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, []);

  const cfg = RARITY_CONFIG[rarity];

  return (
    <>
      <RarityOverlay rarity={effectRarity} onDone={() => setEffectRarity(null)} />
      <GachaButton onClick={changeFact} disabled={rolling} />
      <span className={`flex items-center flex-1 min-w-0 ml-1.5 transition-opacity duration-300 ${visible ? "opacity-100" : "opacity-0"}`}>
        <span className={`text-[10px] font-bold truncate ${cfg.textClass}`}>{text}</span>
      </span>
    </>
  );
});

// ── メインコンポーネント ──────────────────────────────────────
export default function StatusBar({ entries, backendStatus }: Props) {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString("ja-JP", { hour12: false }));

  useEffect(() => {
    const t = setInterval(() => setTime(new Date().toLocaleTimeString("ja-JP", { hour12: false })), 1000);
    return () => clearInterval(t);
  }, []);

  const total   = entries.length;
  const errored = entries.filter(e => e.error && !e.loading).length;

  const statusColor =
    backendStatus === "ready" ? "bg-indigo-500" :
    backendStatus === "error" ? "bg-red-600"    : "bg-amber-500 animate-pulse";

  return (
    <div className="h-7 shrink-0 bg-zinc-950 border-t border-indigo-950/80 flex items-center relative overflow-hidden select-none">
      {/* スキャンライン */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.025]"
        style={{ backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(120,120,255,0.3) 3px, rgba(120,120,255,0.3) 4px)" }} />

      {/* 左端アクセントバー */}
      <div className={`w-0.75 h-full shrink-0 ${statusColor}`} />

      {/* システム状態 */}
      <span className={`font-mono text-[9px] tracking-[0.15em] font-bold px-3 shrink-0 ${
        backendStatus === "ready" ? "text-indigo-400" :
        backendStatus === "error" ? "text-red-400"    : "text-amber-400"
      }`}>
        {backendStatus === "ready" ? "SYS:ONLINE" :
         backendStatus === "error" ? "SYS:OFFLINE" : "SYS:INIT"}
      </span>

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

      {/* ASCII波形ビジュアライザー */}
      <WaveVisualizer />

      <Pipe />

      {/* 豆知識 — flex-1ラッパーで位置を固定 */}
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
  );
}
