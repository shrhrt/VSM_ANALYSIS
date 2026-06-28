import { useState } from "react";
import { createPortal } from "react-dom";
import { BlockMath, InlineMath } from "react-katex";

// ── LaTeX ヘルパー ───────────────────────────────────────────────
function BM({ children }: { children: string }) {
  return <BlockMath math={children} />;
}
function IM({ children }: { children: string }) {
  return <InlineMath math={children} />;
}

// ── M-H ループ SVG（補正後の正確な形状） ─────────────────────────
function MHLoopSVG() {
  const W = 340, H = 240;
  const cx = W / 2, cy = H / 2;
  const Hmax = 130, Mmax = 88;
  const Hc = 52, Mr = 66, Hs = 88;

  const downPath = [
    `M ${cx + Hmax} ${cy - Mmax}`,
    `L ${cx + Hs} ${cy - Mmax}`,
    `C ${cx + Hs * 0.42} ${cy - Mmax} ${cx + Hc * 0.48} ${cy - Mr} ${cx} ${cy - Mr}`,
    `C ${cx - Hc * 0.48} ${cy - Mr} ${cx - Hc} ${cy - Mr * 0.28} ${cx - Hc} ${cy}`,
    `C ${cx - Hc} ${cy + Mr * 0.28} ${cx - Hs * 0.42} ${cy + Mmax} ${cx - Hs} ${cy + Mmax}`,
    `L ${cx - Hmax} ${cy + Mmax}`,
  ].join(" ");

  const upPath = [
    `M ${cx - Hmax} ${cy + Mmax}`,
    `L ${cx - Hs} ${cy + Mmax}`,
    `C ${cx - Hs * 0.42} ${cy + Mmax} ${cx - Hc * 0.48} ${cy + Mr} ${cx} ${cy + Mr}`,
    `C ${cx + Hc * 0.48} ${cy + Mr} ${cx + Hc} ${cy + Mr * 0.28} ${cx + Hc} ${cy}`,
    `C ${cx + Hc} ${cy - Mr * 0.28} ${cx + Hs * 0.42} ${cy - Mmax} ${cx + Hs} ${cy - Mmax}`,
    `L ${cx + Hmax} ${cy - Mmax}`,
  ].join(" ");

  const ax = { stroke: "#52525b", strokeWidth: 1 };
  const lbl = { fontSize: 11, fill: "#a1a1aa" };
  const s = 6;
  const dx = cx - Hmax * 0.65, dy = cy + Mmax;
  const ux = cx + Hmax * 0.65, uy = cy - Mmax;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 200 }}>
      <line x1={10} y1={cy} x2={W - 10} y2={cy} {...ax} />
      <line x1={cx} y1={8}  x2={cx}     y2={H - 8} {...ax} />
      <polygon points={`${W-10},${cy} ${W-18},${cy-4} ${W-18},${cy+4}`} fill="#52525b" />
      <polygon points={`${cx},8 ${cx-4},16 ${cx+4},16`} fill="#52525b" />
      <text x={W - 8} y={cy - 6} {...lbl}>H</text>
      <text x={cx + 5} y={14}    {...lbl}>M</text>
      <line x1={10} y1={cy - Mmax} x2={W - 10} y2={cy - Mmax} stroke="#10b981" strokeWidth="1" strokeDasharray="4 3" opacity="0.6" />
      <line x1={10} y1={cy + Mmax} x2={W - 10} y2={cy + Mmax} stroke="#10b981" strokeWidth="1" strokeDasharray="4 3" opacity="0.6" />
      <text x={8} y={cy - Mmax - 4} fontSize="10" fill="#10b981" fontWeight="bold">Ms</text>
      <text x={8} y={cy + Mmax + 11} fontSize="10" fill="#10b981" fontWeight="bold">−Ms</text>
      <line x1={cx - 6} y1={cy - Mr} x2={cx + 6} y2={cy - Mr} stroke="#f59e0b" strokeWidth="1.5" />
      <line x1={cx - 6} y1={cy + Mr} x2={cx + 6} y2={cy + Mr} stroke="#f59e0b" strokeWidth="1.5" />
      <text x={cx + 8} y={cy - Mr + 4} fontSize="10" fill="#f59e0b" fontWeight="bold">Mr</text>
      <text x={cx + 8} y={cy + Mr + 5} fontSize="10" fill="#f59e0b" fontWeight="bold">−Mr</text>
      <line x1={cx + Hc} y1={cy - 6} x2={cx + Hc} y2={cy + 6} stroke="#ef4444" strokeWidth="1.5" />
      <line x1={cx - Hc} y1={cy - 6} x2={cx - Hc} y2={cy + 6} stroke="#ef4444" strokeWidth="1.5" />
      <text x={cx + Hc - 6} y={cy - 9} fontSize="10" fill="#ef4444" fontWeight="bold">Hc</text>
      <text x={cx - Hc - 16} y={cy - 9} fontSize="10" fill="#ef4444" fontWeight="bold">−Hc</text>
      <line x1={cx + Hs} y1={cy - 5} x2={cx + Hs} y2={cy + 5} stroke="#a78bfa" strokeWidth="1.5" />
      <line x1={cx - Hs} y1={cy - 5} x2={cx - Hs} y2={cy + 5} stroke="#a78bfa" strokeWidth="1.5" />
      <text x={cx + Hs - 6} y={cy + 16} fontSize="10" fill="#a78bfa" fontWeight="bold">Hs</text>
      <text x={cx - Hs - 16} y={cy + 16} fontSize="10" fill="#a78bfa" fontWeight="bold">−Hs</text>
      <path d={downPath} fill="none" stroke="#6366f1" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d={upPath}   fill="none" stroke="#6366f1" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <polygon points={`${dx+s},${dy-s*0.7} ${dx-s*0.5},${dy} ${dx+s},${dy+s*0.7}`} fill="#6366f1" opacity="0.9" />
      <polygon points={`${ux-s},${uy-s*0.7} ${ux+s*0.5},${uy} ${ux-s},${uy+s*0.7}`} fill="#6366f1" opacity="0.9" />
    </svg>
  );
}

// ── 共通スタイルコンポーネント ───────────────────────────────────
function Section({ title }: { title: string }) {
  return <h3 className="text-[11px] font-bold text-zinc-500 uppercase tracking-widest mt-5 mb-2 border-b border-zinc-800 pb-1">{title}</h3>;
}
function Note({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-2 my-3 text-xs text-amber-300/90 bg-amber-950/25 border border-amber-800/30 rounded px-3 py-2">
      <span className="shrink-0 mt-0.5">⚠</span><span className="leading-relaxed">{children}</span>
    </div>
  );
}
function Info({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-2 my-3 text-xs text-sky-300/90 bg-sky-950/25 border border-sky-800/30 rounded px-3 py-2">
      <span className="shrink-0 mt-0.5">ℹ</span><span className="leading-relaxed">{children}</span>
    </div>
  );
}

// 記号定義リスト：式の直後に各記号の意味を一覧表示する
function Sym({ items }: { items: Array<{ sym: string; desc: React.ReactNode }> }) {
  return (
    <div className="my-2.5 space-y-1 pl-3 border-l-2 border-indigo-900/50">
      {items.map(({ sym, desc }) => (
        <div key={sym} className="flex items-baseline gap-1.5 text-xs leading-relaxed">
          <span className="shrink-0 w-32 text-indigo-200/90"><IM>{sym}</IM></span>
          <span className="text-zinc-500 shrink-0 mr-0.5">…</span>
          <span className="text-zinc-400">{desc}</span>
        </div>
      ))}
    </div>
  );
}

// 処理ステップ一覧
function Steps({ items }: { items: string[] }) {
  return (
    <ol className="space-y-1.5 my-2">
      {items.map((item, idx) => (
        <li key={idx} className="flex gap-2 text-xs text-zinc-400 leading-relaxed">
          <span className="shrink-0 w-5 h-5 rounded-full bg-indigo-800/40 text-indigo-300 flex items-center justify-center text-[10px] font-bold mt-0.5">
            {idx + 1}
          </span>
          <span className="pt-0.5">{item}</span>
        </li>
      ))}
    </ol>
  );
}

// ── チャプター定義 ───────────────────────────────────────────────
const CHAPTERS = [
  {
    id: "overview",
    label: "解析の全体像",
    icon: "🔍",
    content: () => (
      <div>
        <p className="text-sm text-zinc-300 leading-relaxed mb-4">
          VSM（振動試料磁力計）・PPMS から出力された生データを読み込み、
          下記の処理を順番に適用して磁気特性パラメータを算出する。
        </p>

        <div className="space-y-1 mb-5">
          {[
            ["①", "単位換算", "Oe → T、emu → kA/m への変換と体積磁化への換算"],
            ["②", "反磁性補正", "基板・ホルダー由来の線形常磁性成分を差し引く"],
            ["③", "オフセット補正", "M 軸のゼロ点ドリフトを除去"],
            ["④", "ブランチ分割", "往路（H 減少方向）と復路（H 増加方向）に分離"],
            ["⑤", "特性量算出", "Ms・Mr・Hc・Hs・角形比 S を計算"],
          ].map(([n, title, desc]) => (
            <div key={n} className="flex gap-3 items-start py-1.5 border-b border-zinc-800/60">
              <span className="shrink-0 w-6 h-6 rounded bg-indigo-800/60 text-indigo-300 text-[11px] flex items-center justify-center font-bold">{n}</span>
              <div>
                <span className="text-sm text-zinc-200 font-semibold">{title}</span>
                <span className="text-xs text-zinc-500 ml-2">{desc}</span>
              </div>
            </div>
          ))}
        </div>

        <p className="text-xs text-zinc-500 mb-2 text-center">補正後の M-H ヒステリシスループと各特性量の位置</p>
        <MHLoopSVG />

        <div className="mt-3 grid grid-cols-4 gap-1.5 text-[10px] text-center">
          {[
            { label: "Ms", color: "text-emerald-400", desc: "飽和磁化" },
            { label: "Mr", color: "text-amber-400",   desc: "残留磁化" },
            { label: "Hc", color: "text-red-400",     desc: "保磁力" },
            { label: "Hs", color: "text-violet-400",  desc: "飽和磁場" },
          ].map((x) => (
            <div key={x.label} className="bg-zinc-800/50 rounded py-1">
              <p className={`font-mono font-bold ${x.color}`}>{x.label}</p>
              <p className="text-zinc-600 mt-0.5">{x.desc}</p>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    id: "units",
    label: "単位換算・体積磁化",
    icon: "📐",
    content: () => (
      <div>
        <p className="text-sm text-zinc-300 leading-relaxed mb-3">
          生データは CGS 単位系（Oe・emu）で出力される。サンプル寸法（面積・膜厚）を用いて体積磁化に換算し、SI 単位系で解析する。
        </p>

        <Section title="体積の計算" />
        <p className="text-xs text-zinc-400 mb-1">入力された面積と膜厚からサンプル体積を計算する。</p>
        <BM>{"V\\ [\\mathrm{cm^3}] = A\\ [\\mathrm{mm^2}] \\times t\\ [\\mathrm{nm}] \\times 10^{-7}"}</BM>
        <Sym items={[
          { sym: "V", desc: "サンプルの体積（薄膜の場合 = 面積 × 膜厚）" },
          { sym: "A", desc: "サンプルの面積。サイドバーに入力する値（単位: mm²）" },
          { sym: "t", desc: "薄膜の膜厚。サイドバーに入力する値（単位: nm）" },
          { sym: "10^{-7}", desc: "単位変換係数。mm² × nm = 10⁻⁶ mm³ = 10⁻⁷ cm³ の関係による" },
        ]} />

        <Section title="磁場の換算（CGS → SI）" />
        <p className="text-xs text-zinc-400 mb-1">測定ファイル中の磁場値 H [Oe] を SI 単位の μ₀H [T] に変換する。</p>
        <BM>{"\\mu_0 H\\ [\\mathrm{T}] = H\\ [\\mathrm{Oe}] \\times 10^{-4}"}</BM>
        <Sym items={[
          { sym: "\\mu_0", desc: "真空の透磁率（= 4π × 10⁻⁷ H/m）。Oe を T に変換する際の係数として使われる" },
          { sym: "H\\ [\\mathrm{Oe}]", desc: "VSM ファイルに記録された生の磁場値（エルステッド）" },
          { sym: "10^{-4}", desc: "換算係数。1 Oe = 10⁻⁴ T" },
        ]} />

        <Section title="磁化の換算（モーメント → 体積磁化）" />
        <p className="text-xs text-zinc-400 mb-1">VSM が測定した磁気モーメント m [emu] をサンプル体積 V で割り、単位体積あたりの磁化に変換する。</p>
        <BM>{"M\\ [\\mathrm{kA/m}] = \\frac{m\\ [\\mathrm{emu}]}{V\\ [\\mathrm{cm^3}]}"}</BM>
        <Sym items={[
          { sym: "M", desc: "体積磁化（単位体積あたりの磁気モーメント）。グラフの縦軸として表示される値" },
          { sym: "m", desc: "VSM が直接測定した磁気モーメント（単位: emu）" },
          { sym: "V", desc: "上の式で求めたサンプル体積（単位: cm³）" },
        ]} />
        <Info>1 emu/cm³ = 1 kA/m は CGS と SI の間の恒等式であり、追加の係数は不要である。</Info>

        <Section title="表示単位の切り替え" />
        <div className="grid grid-cols-3 gap-2 text-xs text-center">
          {[
            { label: "SI", h: "\\mu_0 H\\ [\\mathrm{T}]", m: "M\\ [\\mathrm{kA/m}]" },
            { label: "CGS", h: "H\\ [\\mathrm{Oe}]", m: "M\\ [\\mathrm{emu/cm^3}]" },
            { label: "規格化", h: "\\mu_0 H\\ [\\mathrm{T}]", m: "M/M_s" },
          ].map((u) => (
            <div key={u.label} className="bg-zinc-800/50 rounded p-2">
              <p className="font-semibold text-zinc-300 mb-1">{u.label}</p>
              <p className="text-zinc-400 text-[10px]"><IM>{u.h}</IM></p>
              <p className="text-zinc-400 text-[10px]"><IM>{u.m}</IM></p>
            </div>
          ))}
        </div>

        <Note>膜厚・面積の入力誤差は Ms・Mr の絶対値に直接影響する。ファイル別設定で個別に上書きできる。</Note>
      </div>
    ),
  },
  {
    id: "demag",
    label: "反磁性補正",
    icon: "📉",
    content: () => (
      <div>
        <p className="text-sm text-zinc-300 leading-relaxed mb-3">
          強磁性薄膜の M-H カーブには、サンプルホルダー・基板・測定棒に由来する
          <span className="text-orange-300 font-semibold">線形な常磁性・反磁性成分</span>が重畳している。
          磁化が飽和した後の高磁場領域でこの傾きを計測し、全データから差し引く。
        </p>

        <Section title="補正の式" />
        <BM>{"M_\\mathrm{corrected} = M_\\mathrm{raw} - \\chi \\cdot H"}</BM>
        <Sym items={[
          { sym: "M_\\mathrm{corrected}", desc: "補正後の磁化（グラフに表示される値）" },
          { sym: "M_\\mathrm{raw}", desc: "VSM が測定した生の磁化値（バックグラウンド込み）" },
          { sym: "\\chi\\ (\\text{カイ})", desc: "反磁性補正の傾き（実効的な磁化率）。フィッティングで求める" },
          { sym: "H", desc: "各データ点の磁場値。全データ点について個別に補正が適用される" },
        ]} />

        <Section title="自動モード：傾き χ の求め方" />
        <p className="text-xs text-zinc-400 mb-2">
          磁化が飽和した高磁場領域では、強磁性成分は一定（≈ Ms）になる。
          残る傾きはすべてバックグラウンドに由来するため、その傾きを χ として抽出できる。
          正側と負側の両方でフィッティングを行い、その平均を取る。
        </p>

        <p className="text-xs text-zinc-500 font-semibold mb-1 mt-3">正の高磁場側の傾き:</p>
        <BM>{"\\chi_+ = \\mathrm{slope}\\!\\left(\\left\\{H_i,\\, M_i\\right\\}_{H_i > 0.85\\,H_\\max}\\right)"}</BM>
        <Sym items={[
          { sym: "i", desc: "データ点の番号。全測定点を 1 番から N 番まで番号付けした添字" },
          { sym: "H_i", desc: "i 番目のデータ点の磁場値" },
          { sym: "M_i", desc: "i 番目のデータ点の磁化値" },
          { sym: "H_\\max", desc: "全データ中で最も大きい正の磁場値（測定範囲の上限）" },
          { sym: "H_i > 0.85\\,H_\\max", desc: "「H_i が H_max の 85% を超える」という抽出条件。この条件を満たすデータ点（正の飽和領域の上位 15% ）のみを対象とする" },
          { sym: "\\mathrm{slope}(\\ldots)", desc: "指定した点群に最小二乗法で直線 M = slope·H + b を当てはめたときの傾きを返す" },
          { sym: "\\chi_+", desc: "正の高磁場側フィッティングから得られた傾き" },
        ]} />

        <p className="text-xs text-zinc-500 font-semibold mb-1 mt-3">負の高磁場側の傾き:</p>
        <BM>{"\\chi_- = \\mathrm{slope}\\!\\left(\\left\\{H_i,\\, M_i\\right\\}_{H_i < 0.85\\,H_\\min}\\right)"}</BM>
        <Sym items={[
          { sym: "H_\\min", desc: "全データ中で最も大きい負の磁場値（絶対値が最大の負の磁場）" },
          { sym: "H_i < 0.85\\,H_\\min", desc: "H_i が H_min の 85% より小さいという条件。H_min は負値なので、これは負の飽和領域の下位 15% を指す" },
          { sym: "\\chi_-", desc: "負の高磁場側フィッティングから得られた傾き" },
        ]} />

        <p className="text-xs text-zinc-500 font-semibold mb-1 mt-3">最終的な補正傾き:</p>
        <BM>{"\\chi = \\frac{\\chi_+ + \\chi_-}{2}"}</BM>
        <p className="text-xs text-zinc-400 mb-1">正負の傾きの平均を取ることで、測定の非対称性の影響を軽減する。</p>

        <Section title="処理の手順（自動モード）" />
        <Steps items={[
          "全データ点の中から H_i > 0.85 H_max を満たすものを抽出する（正の飽和領域）",
          "抽出した点群に最小二乗直線フィッティングを適用し、傾き χ+ を得る",
          "同様に H_i < 0.85 H_min を満たす点を抽出し、傾き χ− を得る",
          "χ = (χ+ + χ−) / 2 を最終的な補正傾きとして確定する",
          "全データ点に M_corrected = M_raw − χ·H を適用する",
        ]} />

        <Note>R² &lt; 0.99 の場合は警告を表示する。飽和が不十分、または非線形成分がある可能性がある。</Note>

        <Section title="手動モード：傾きの算出" />
        <p className="text-xs text-zinc-400 mb-1">
          ユーザーが指定した磁場範囲 <IM>{"[H_a,\\, H_b]"}</IM> 内のデータ点のみを使ってフィッティングする。
          自動で選ばれる飽和領域が不適切な場合に使用する。
        </p>
        <BM>{"\\chi_+ = \\mathrm{slope}\\!\\left(\\left\\{H_i,\\, M_i\\right\\}_{H_a \\le H_i \\le H_b}\\right)"}</BM>
        <Sym items={[
          { sym: "H_a,\\, H_b", desc: "ユーザーがサイドバーで指定した正の磁場範囲の下限・上限" },
        ]} />
        <Info>データ点が 5 点未満の範囲を指定すると精度が低下する。ログで点数を確認すること。</Info>
      </div>
    ),
  },
  {
    id: "offset",
    label: "オフセット補正",
    icon: "⚖️",
    content: () => (
      <div>
        <p className="text-sm text-zinc-300 leading-relaxed mb-3">
          測定器のドリフトにより、ヒステリシスループが M 軸方向に全体的にシフトすることがある。
          高磁場端の磁化平均を用い、ループが原点を中心に上下対称になるよう補正する。
        </p>

        <Section title="オフセット値の算出" />
        <p className="text-xs text-zinc-400 mb-1">
          正の高磁場端と負の高磁場端それぞれで磁化の平均値を求め、その中点をゼロ点オフセットとする。
        </p>

        <BM>{"\\bar{M}_+ = \\left\\langle M_i \\right\\rangle_{H_i > 0.9\\,H_\\max}"}</BM>
        <Sym items={[
          { sym: "\\bar{M}_+", desc: "正の高磁場側における磁化の平均値（飽和していれば ≈ +Ms）" },
          { sym: "\\langle \\cdots \\rangle", desc: "条件を満たすデータ点の算術平均（合計 ÷ 点数）" },
          { sym: "H_i > 0.9\\,H_\\max", desc: "H_max の 90% を超えるデータ点のみを対象とする条件" },
        ]} />

        <BM>{"\\bar{M}_- = \\left\\langle M_i \\right\\rangle_{H_i < 0.9\\,H_\\min}"}</BM>
        <Sym items={[
          { sym: "\\bar{M}_-", desc: "負の高磁場側における磁化の平均値（飽和していれば ≈ −Ms）" },
          { sym: "H_i < 0.9\\,H_\\min", desc: "H_min の 90% より小さいデータ点を対象とする条件" },
        ]} />

        <BM>{"\\delta = \\frac{\\bar{M}_+ + \\bar{M}_-}{2}"}</BM>
        <Sym items={[
          { sym: "\\delta\\ (\\text{デルタ})", desc: "ゼロ点オフセット量。正負の飽和値の中点を取る。理想的なループでは 0 になる" },
        ]} />

        <Section title="補正の適用" />
        <BM>{"M_\\mathrm{final} = M_\\mathrm{corrected} - \\delta"}</BM>
        <Sym items={[
          { sym: "M_\\mathrm{final}", desc: "オフセット補正まで完了した最終的な磁化値。以降の特性量算出に使用する" },
          { sym: "M_\\mathrm{corrected}", desc: "反磁性補正済みの磁化値（前ステップの出力）" },
        ]} />

        <Section title="処理の手順" />
        <Steps items={[
          "H_i > 0.9 H_max を満たすデータ点を抽出し、それらの磁化の平均値 M̄+ を求める",
          "H_i < 0.9 H_min を満たすデータ点を抽出し、平均値 M̄− を求める",
          "δ = (M̄+ + M̄−) / 2 をゼロ点オフセットとして確定する",
          "全データ点に M_final = M_corrected − δ を適用する",
        ]} />

        <Note>
          交換バイアス系など、ループが意図的に非対称な試料ではオフセット補正を無効にすること。
          補正するとシフト量が失われ、交換バイアス磁場 H_ex が正しく評価できなくなる。
        </Note>
      </div>
    ),
  },
  {
    id: "ms",
    label: "飽和磁化 Ms",
    icon: "🟢",
    content: () => (
      <div>
        <p className="text-sm text-zinc-300 leading-relaxed mb-3">
          飽和磁化 <IM>{"M_s"}</IM> は、磁場が十分大きいときに磁化が達する最大値である。
          正側・負側それぞれの高磁場領域の平均から算出する。
        </p>

        <Section title="自動モード" />
        <p className="text-xs text-zinc-400 mb-1">最大磁場の 90% 以上の領域のデータ点を抽出し、その平均を Ms とする。</p>

        <BM>{"M_{s,+} = \\left\\langle M_i \\right\\rangle_{H_i > 0.9\\,H_\\max}"}</BM>
        <Sym items={[
          { sym: "M_{s,+}", desc: "正の高磁場側から求めた Ms の推定値" },
          { sym: "H_i > 0.9\\,H_\\max", desc: "最大磁場の 90% 以上の領域。ここでは磁化が十分飽和していると仮定する" },
        ]} />

        <BM>{"M_{s,-} = \\left\\langle |M_i| \\right\\rangle_{H_i < 0.9\\,H_\\min}"}</BM>
        <Sym items={[
          { sym: "M_{s,-}", desc: "負の高磁場側から求めた Ms の推定値" },
          { sym: "|M_i|", desc: "磁化の絶対値。負の磁場では M が負になるため絶対値を取る" },
        ]} />

        <BM>{"M_s = \\frac{M_{s,+} + M_{s,-}}{2}"}</BM>
        <p className="text-xs text-zinc-400">正負の両側の平均を取ることで、測定ノイズや微小な非対称性の影響を軽減する。</p>

        <Section title="手動モード" />
        <p className="text-xs text-zinc-400 mb-1">
          ユーザーが正側と負側それぞれに磁場範囲を指定し、その範囲内の平均を Ms とする。
          自動モードで選ばれる範囲が不適切な場合（飽和が弱い、ノイズが多いなど）に使用する。
        </p>
        <BM>{"M_{s,+} = \\left\\langle M_i \\right\\rangle_{H_a \\le H_i \\le H_b}"}</BM>
        <BM>{"M_{s,-} = \\left\\langle |M_i| \\right\\rangle_{H_c \\le H_i \\le H_d}"}</BM>
        <Sym items={[
          { sym: "H_a,\\, H_b", desc: "ユーザーが指定した正側の磁場範囲の下限・上限" },
          { sym: "H_c,\\, H_d", desc: "ユーザーが指定した負側の磁場範囲の下限・上限" },
        ]} />

        <Note>
          磁場が不足して飽和に達していない場合、自動モードでは <IM>{"M_s"}</IM> が過小評価される。
          その際は手動で高磁場端に近い範囲を指定すること。
        </Note>
        <Info>
          正側と負側のどちらか一方のデータしか存在しない場合、存在する側の値のみを <IM>{"M_s"}</IM> として使用する。
        </Info>
      </div>
    ),
  },
  {
    id: "mr",
    label: "残留磁化 Mr",
    icon: "🟡",
    content: () => (
      <div>
        <p className="text-sm text-zinc-300 leading-relaxed mb-3">
          残留磁化 <IM>{"M_r"}</IM> は、印加磁場をゼロに戻したときに残る磁化の大きさ（<IM>{"H=0"}</IM> における <IM>{"|M|"}</IM>）である。
        </p>

        <Section title="ブランチ分割" />
        <p className="text-xs text-zinc-400 mb-1">
          全データを H の変化方向で往路（H が正から負へ減少する方向 ↓）と
          復路（H が負から正へ増加する方向 ↑）に分ける。
          Mr は往路と復路の両方で求め、最終的にその平均を取る。
        </p>

        <Section title="H=0 における M の推定" />
        <p className="text-xs text-zinc-400 mb-1">
          測定点が H=0 にぴったり落ちることはまれである。そのため、H=0 を挟む 2 点の間で線形補間（2 点間の比例計算）を行い、H=0 の磁化値を推定する。
        </p>
        <BM>{"M_r^\\downarrow = M^\\downarrow(H = 0) \\quad \\text{（往路）}"}</BM>
        <BM>{"M_r^\\uparrow = M^\\uparrow(H = 0) \\quad \\text{（復路）}"}</BM>
        <Sym items={[
          { sym: "M^\\downarrow(H=0)", desc: "往路データを H の降順に並べたとき、H=0 を挟む 2 点から線形補間した磁化値" },
          { sym: "M^\\uparrow(H=0)", desc: "復路データを H の昇順に並べたとき、H=0 を挟む 2 点から線形補間した磁化値" },
        ]} />

        <Section title="平均値の算出" />
        <BM>{"M_r = \\frac{|M_r^\\downarrow| + |M_r^\\uparrow|}{2}"}</BM>
        <p className="text-xs text-zinc-400">往路と復路の Mr が僅かに異なる場合（磁化の非対称性など）でも、両者の平均を報告値とする。</p>

        <Section title="処理の手順" />
        <Steps items={[
          "全データ点を H の変化方向（増加 or 減少）で往路・復路の 2 グループに分割する",
          "往路データの中から H の符号が正から負に変わる 2 点を特定する",
          "その 2 点の間で線形補間を行い、H=0 における磁化 Mr↓ を求める",
          "復路データについて同様に処理し、H=0 における磁化 Mr↑ を求める",
          "Mr = (|Mr↓| + |Mr↑|) / 2 を最終値とする",
        ]} />

        <Note>
          測定ノイズが大きいと H=0 付近でブランチの割り当てが不安定になる場合がある。
          ログの「往路/復路データ点数」を確認すること。
        </Note>
      </div>
    ),
  },
  {
    id: "hc",
    label: "保磁力 Hc",
    icon: "🔴",
    content: () => (
      <div>
        <p className="text-sm text-zinc-300 leading-relaxed mb-3">
          保磁力 <IM>{"H_c"}</IM> は、磁化がゼロになるときの磁場の絶対値（<IM>{"M=0"}</IM> における <IM>{"|H|"}</IM>）である。
          磁石の「硬さ」を表す最重要パラメータのひとつであり、値が大きいほど消磁されにくい材料といえる。
        </p>

        <Section title="M=0 における H の推定" />
        <p className="text-xs text-zinc-400 mb-1">
          Mr と同様に往路・復路に分割した後、M=0 を挟む 2 点の間で線形補間を行い、M=0 における磁場値を推定する。
        </p>
        <BM>{"H_c^\\downarrow = H^\\downarrow(M = 0) \\quad \\text{（往路）}"}</BM>
        <BM>{"H_c^\\uparrow = H^\\uparrow(M = 0) \\quad \\text{（復路）}"}</BM>
        <Sym items={[
          { sym: "H^\\downarrow(M=0)", desc: "往路（H 減少ブランチ）において M=0 を挟む 2 点から線形補間した磁場値。負の値になる（図の −Hc 側）" },
          { sym: "H^\\uparrow(M=0)", desc: "復路（H 増加ブランチ）において M=0 を挟む 2 点から線形補間した磁場値。正の値になる（図の +Hc 側）" },
        ]} />

        <Section title="平均値の算出" />
        <BM>{"H_c = \\frac{|H_c^\\downarrow| + |H_c^\\uparrow|}{2}"}</BM>
        <p className="text-xs text-zinc-400">往路と復路の Hc の絶対値を平均する。交換バイアスがある試料では左右で値が異なる。</p>

        <Section title="単位換算" />
        <BM>{"H_c\\ [\\mathrm{Oe}] = H_c\\ [\\mathrm{T}] \\times 10^4"}</BM>
        <BM>{"\\mu_0 H_c\\ [\\mathrm{mT}] = H_c\\ [\\mathrm{T}] \\times 10^3"}</BM>

        <Section title="処理の手順" />
        <Steps items={[
          "往路データの中から M の符号が正から負に変わる 2 点を特定する",
          "その 2 点の間で線形補間を行い、M=0 における磁場 Hc↓（負の値）を求める",
          "復路データについて同様に処理し、Hc↑（正の値）を求める",
          "Hc = (|Hc↓| + |Hc↑|) / 2 を最終値とする",
          "結果を Oe および mT に換算して結果テーブルに表示する",
        ]} />

        <Info>
          保磁力は往路・復路の非対称性により左右で値が異なることがある（交換バイアス等）。
          ログには <IM>{"H_c^\\downarrow"}</IM> と <IM>{"H_c^\\uparrow"}</IM> の個別値も記録される。
        </Info>
      </div>
    ),
  },
  {
    id: "hs",
    label: "飽和磁場 Hs",
    icon: "🟠",
    content: () => (
      <div>
        <p className="text-sm text-zinc-300 leading-relaxed mb-3">
          飽和磁場 <IM>{"H_s"}</IM> は、磁化が飽和に達するのに必要な最小磁場の大きさである。
          単純に「|M| が閾値を超えた最初の点」を採用するとノイズ点 1 つで誤検出するため、
          孤立した外れ値に強い「連続点アルゴリズム」を採用している。
        </p>

        <Section title="飽和判定の閾値" />
        <BM>{"M_\\mathrm{th} = M_s \\left(1 - \\frac{\\varepsilon}{100}\\right)"}</BM>
        <Sym items={[
          { sym: "M_\\mathrm{th}", desc: "飽和の判定に使う閾値。この値を超えた点を「飽和している」とみなす" },
          { sym: "M_s", desc: "前のステップで求めた飽和磁化の値" },
          { sym: "\\varepsilon\\ (\\text{イプシロン})", desc: "Ms からの許容偏差（%）。デフォルト 2%。サイドバーの解析設定から変更できる" },
          { sym: "1 - \\varepsilon/100", desc: "例: ε=2 なら閾値は 0.98 Ms（Ms の 98%）。この値より大きければ飽和とみなす" },
        ]} />

        <Section title="アルゴリズムの詳細" />
        <p className="text-xs text-zinc-400 mb-2">往路・復路ブランチそれぞれについて、以下の処理を独立に実行する。</p>
        <Steps items={[
          "ブランチのデータ点を |H|（磁場の絶対値）の昇順、すなわち磁場が弱い点から強い点の順に並び替える",
          "各データ点を順番に確認し、|M_i| < M_th（閾値未満）の点を「未飽和」と記録する",
          "「未飽和」の点が n_c 点以上連続して並ぶまとまり（デフォルト n_c = 3）を「未飽和ラン」と定義し、それが最後に現れる場所を特定する",
          "その最後の未飽和ランの直後にある最初の「飽和」点の |H| を Hs として採用する",
        ]} />
        <p className="text-xs text-zinc-400 mt-2">
          n_c 点以上の連続を要件とするため、Ms 近傍のノイズ点が 1〜2 点あっても Hs が誤検出されにくい。
        </p>

        <Section title="正側・負側の算出と平均" />
        <p className="text-xs text-zinc-400 mb-1">往路の正の磁場側と復路の負の磁場側でそれぞれ Hs を求め、平均を取る。</p>
        <BM>{"H_s = \\frac{H_{s,+} + H_{s,-}}{2}"}</BM>
        <Sym items={[
          { sym: "H_{s,+}", desc: "正の磁場側（往路の高磁場端）で検出された飽和磁場" },
          { sym: "H_{s,-}", desc: "負の磁場側（復路の高磁場端）で検出された飽和磁場" },
        ]} />

        <Note>
          ノイズが大きいデータでは n_c（連続判定点数）を増やすと安定する。
          測定範囲内で飽和に到達しない場合は Hs が算出されない。
        </Note>
      </div>
    ),
  },
  {
    id: "squareness",
    label: "角形比 Mr/Ms",
    icon: "⬛",
    content: () => (
      <div>
        <p className="text-sm text-zinc-300 leading-relaxed mb-3">
          角形比 <IM>{"S"}</IM>（スクエアネス）は、ヒステリシスループの「四角さ」を表す無次元指標である。
          磁気記録媒体や永久磁石の性能評価に広く用いられる。
        </p>

        <Section title="定義" />
        <BM>{"S = \\frac{M_r}{M_s} \\quad (0 \\le S \\le 1)"}</BM>
        <Sym items={[
          { sym: "S", desc: "角形比（スクエアネス）。0 以上 1 以下の無次元数" },
          { sym: "M_r", desc: "残留磁化（H=0 での磁化の大きさ）。前のステップで求めた値" },
          { sym: "M_s", desc: "飽和磁化（十分な磁場をかけたときの磁化の最大値）。前のステップで求めた値" },
        ]} />

        <Section title="値の解釈" />
        <div className="space-y-2 mt-2">
          {[
            { range: "S \\approx 1", color: "text-emerald-400 bg-emerald-900/20 border-emerald-800/30", title: "理想的な角形ループ", desc: "一軸磁気異方性が強く、スイッチングが急峻である。残留磁化が飽和磁化にほぼ等しい。垂直磁気記録媒体・永久磁石に有利な特性である。" },
            { range: "S \\approx 0.5", color: "text-amber-400 bg-amber-900/20 border-amber-800/30", title: "等方性（面内多結晶）", desc: "結晶方位がランダムに分布した多結晶体や面内等方性薄膜で見られる典型的な値である。" },
            { range: "S \\approx 0", color: "text-red-400 bg-red-900/20 border-red-800/30", title: "難磁化軸方向", desc: "印加磁場が難磁化軸に平行な場合、またはソフト磁性材料で磁場印加方向と易軸が揃っていない場合に現れる。" },
          ].map((x) => (
            <div key={x.range} className={`border rounded p-2.5 ${x.color}`}>
              <div className="flex items-baseline gap-2 mb-0.5">
                <span className="font-mono text-sm font-bold"><IM>{x.range}</IM></span>
                <span className="text-xs font-semibold">{x.title}</span>
              </div>
              <p className="text-[11px] opacity-80 leading-relaxed">{x.desc}</p>
            </div>
          ))}
        </div>

        <Note>S &gt; 1 は <IM>{"M_s"}</IM> の過小評価（飽和不足・範囲設定ミス）を示している可能性がある。</Note>
      </div>
    ),
  },
] as const;

// ── メインダイアログ ──────────────────────────────────────────────
export default function HelpDialog({ onClose }: { onClose: () => void }) {
  const [active, setActive] = useState<string>("overview");
  const chapter = CHAPTERS.find((c) => c.id === active) ?? CHAPTERS[0];
  const Content = chapter.content;

  return createPortal(
    <div className="fixed inset-0 z-9998 flex items-center justify-center bg-black/70">
      <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl flex overflow-hidden"
        style={{ width: 740, height: 580 }}>

        {/* 左: チャプターナビ */}
        <nav className="w-44 shrink-0 bg-zinc-950/70 border-r border-zinc-800 flex flex-col">
          <div className="px-3 py-3 border-b border-zinc-800">
            <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">計算ロジック解説</p>
          </div>
          <div className="flex-1 overflow-y-auto py-1">
            {CHAPTERS.map((c) => (
              <button key={c.id} onClick={() => setActive(c.id)}
                className={`w-full text-left flex items-center gap-2 px-3 py-2 text-xs transition-colors ${
                  active === c.id
                    ? "bg-indigo-700/40 text-indigo-200 font-semibold border-l-2 border-indigo-500"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 border-l-2 border-transparent"
                }`}>
                <span className="text-base leading-none">{c.icon}</span>
                <span className="leading-tight">{c.label}</span>
              </button>
            ))}
          </div>
          <div className="p-3 border-t border-zinc-800">
            <button onClick={onClose}
              className="w-full text-xs py-1.5 rounded bg-zinc-700 hover:bg-zinc-600 text-zinc-300 transition-colors">
              閉じる
            </button>
          </div>
        </nav>

        {/* 右: スクロールコンテンツ */}
        <main className="flex-1 overflow-y-auto p-5">
          <div className="flex items-center gap-2 mb-4 pb-3 border-b border-zinc-800">
            <span className="text-xl leading-none">{chapter.icon}</span>
            <h2 className="text-base font-bold text-zinc-100">{chapter.label}</h2>
          </div>
          <div className="[&_.katex]:text-zinc-100 [&_.katex-display]:overflow-x-auto [&_.katex-display]:py-1">
            <Content />
          </div>
        </main>
      </div>
    </div>,
    document.body,
  );
}
