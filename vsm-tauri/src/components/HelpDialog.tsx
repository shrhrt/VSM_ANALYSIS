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

// ── M-H ループ SVG（1枚・正確な形状） ──────────────────────────
function MHLoopSVG() {
  // 座標系: W×H のキャンバス、中心 (cx,cy)
  const W = 340, H = 240;
  const cx = W / 2, cy = H / 2;
  const Hmax = 130, Mmax = 88; // 描画スケール (px)
  const Hc = 52, Mr = 66, Hs = 88; // 特性量の位置 (px)

  // 降磁場ブランチ: (+Hmax,+Ms) → (+Hc,0) → (0,-Mr) → (-Hmax,-Ms)
  // 昇磁場ブランチ: (-Hmax,-Ms) → (-Hc,0) → (0,+Mr) → (+Hmax,+Ms)
  // Cubic Bezier で S字カーブを表現
  const down = `
    M ${cx + Hmax} ${cy - Mmax}
    C ${cx + Hmax} ${cy - Mmax * 0.3},
      ${cx + Hc + 18} ${cy + 4},
      ${cx + Hc} ${cy}
    C ${cx + Hc - 12} ${cy - 4},
      ${cx + 12} ${cy - Mr + 4},
      ${cx} ${cy - Mr}
    C ${cx - 12} ${cy - Mr - 4},
      ${cx - Hmax} ${cy - Mmax * 0.3},
      ${cx - Hmax} ${cy + Mmax}
  `.trim();

  const up = `
    M ${cx - Hmax} ${cy + Mmax}
    C ${cx - Hmax} ${cy + Mmax * 0.3},
      ${cx - Hc - 18} ${cy - 4},
      ${cx - Hc} ${cy}
    C ${cx - Hc + 12} ${cy + 4},
      ${cx - 12} ${cy + Mr - 4},
      ${cx} ${cy + Mr}
    C ${cx + 12} ${cy + Mr + 4},
      ${cx + Hmax} ${cy + Mmax * 0.3},
      ${cx + Hmax} ${cy - Mmax}
  `.trim();

  const ax = { stroke: "#52525b", strokeWidth: 1 };
  const label = { fontSize: 11, fill: "#a1a1aa" };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 200 }}>
      {/* 軸 */}
      <line x1={10} y1={cy} x2={W - 10} y2={cy} {...ax} />
      <line x1={cx} y1={8}  x2={cx}     y2={H - 8} {...ax} />
      {/* 軸矢印 */}
      <polygon points={`${W-10},${cy} ${W-18},${cy-4} ${W-18},${cy+4}`} fill="#52525b" />
      <polygon points={`${cx},8 ${cx-4},16 ${cx+4},16`} fill="#52525b" />
      {/* 軸ラベル */}
      <text x={W - 8} y={cy - 6} {...label}>H</text>
      <text x={cx + 5} y={14}    {...label}>M</text>

      {/* 特性量の補助線 */}
      {/* Ms */}
      <line x1={10} y1={cy - Mmax} x2={W - 10} y2={cy - Mmax} stroke="#10b981" strokeWidth="1" strokeDasharray="4 3" opacity="0.7" />
      <line x1={10} y1={cy + Mmax} x2={W - 10} y2={cy + Mmax} stroke="#10b981" strokeWidth="1" strokeDasharray="4 3" opacity="0.7" />
      <text x={8} y={cy - Mmax - 4} fontSize="10" fill="#10b981" fontWeight="bold">Ms</text>
      <text x={8} y={cy + Mmax + 11} fontSize="10" fill="#10b981" fontWeight="bold">−Ms</text>

      {/* Mr */}
      <line x1={cx - 6} y1={cy - Mr} x2={cx + 6} y2={cy - Mr} stroke="#f59e0b" strokeWidth="1.5" />
      <line x1={cx - 6} y1={cy + Mr} x2={cx + 6} y2={cy + Mr} stroke="#f59e0b" strokeWidth="1.5" />
      <text x={cx + 8} y={cy - Mr + 4} fontSize="10" fill="#f59e0b" fontWeight="bold">Mr</text>
      <text x={cx + 8} y={cy + Mr + 4} fontSize="10" fill="#f59e0b" fontWeight="bold">−Mr</text>

      {/* Hc */}
      <line x1={cx + Hc} y1={cy - 6} x2={cx + Hc} y2={cy + 6} stroke="#ef4444" strokeWidth="1.5" />
      <line x1={cx - Hc} y1={cy - 6} x2={cx - Hc} y2={cy + 6} stroke="#ef4444" strokeWidth="1.5" />
      <text x={cx + Hc + 3} y={cy - 8} fontSize="10" fill="#ef4444" fontWeight="bold">Hc</text>
      <text x={cx - Hc + 3} y={cy - 8} fontSize="10" fill="#ef4444" fontWeight="bold">−Hc</text>

      {/* Hs */}
      <line x1={cx + Hs} y1={cy - 5} x2={cx + Hs} y2={cy + 5} stroke="#a78bfa" strokeWidth="1.5" />
      <line x1={cx - Hs} y1={cy - 5} x2={cx - Hs} y2={cy + 5} stroke="#a78bfa" strokeWidth="1.5" />
      <text x={cx + Hs + 3} y={cy + 14} fontSize="10" fill="#a78bfa" fontWeight="bold">Hs</text>
      <text x={cx - Hs + 3} y={cy + 14} fontSize="10" fill="#a78bfa" fontWeight="bold">−Hs</text>

      {/* ループ */}
      <path d={down} fill="none" stroke="#6366f1" strokeWidth="2.2" strokeLinejoin="round" />
      <path d={up}   fill="none" stroke="#6366f1" strokeWidth="2.2" strokeLinejoin="round" />

      {/* 方向矢印 */}
      <defs>
        <marker id="arr" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill="#6366f1" opacity="0.8" />
        </marker>
      </defs>
      <line x1={cx + Hmax - 2} y1={cy - Mmax + 14} x2={cx + Hmax - 2} y2={cy - Mmax + 2} stroke="none" markerEnd="url(#arr)" />
      <line x1={cx - Hmax + 2} y1={cy + Mmax - 14} x2={cx - Hmax + 2} y2={cy + Mmax - 2} stroke="none" markerEnd="url(#arr)" />
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
          下記の処理を順番に適用して磁気特性パラメータを算出します。
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

        <p className="text-xs text-zinc-500 mb-2 text-center">代表的な M-H ヒステリシスループと各特性量の位置</p>
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
          生データは CGS 単位系（Oe・emu）で出力されます。サンプル寸法（面積・膜厚）を用いて体積磁化に換算し、SI 単位系で解析します。
        </p>

        <Section title="体積の計算" />
        <p className="text-xs text-zinc-400 mb-1">入力された面積 <IM>{"A\\ [\\mathrm{mm^2}]"}</IM> と膜厚 <IM>{"t\\ [\\mathrm{nm}]"}</IM> からサンプル体積を求めます。</p>
        <BM>{"V\\ [\\mathrm{cm^3}] = A\\ [\\mathrm{mm^2}] \\times t\\ [\\mathrm{nm}] \\times 10^{-7}"}</BM>
        <p className="text-xs text-zinc-500">（単位換算: <IM>{"\\mathrm{mm^2 \\cdot nm} = 10^{-6}\\,\\mathrm{mm^3} = 10^{-7}\\,\\mathrm{cm^3}"}</IM>）</p>

        <Section title="磁場の換算（CGS → SI）" />
        <BM>{"\\mu_0 H\\ [\\mathrm{T}] = H\\ [\\mathrm{Oe}] \\times 10^{-4}"}</BM>

        <Section title="磁化の換算（モーメント → 体積磁化）" />
        <BM>{"M\\ [\\mathrm{kA/m}] = \\frac{m\\ [\\mathrm{emu}]}{V\\ [\\mathrm{cm^3}]}"}</BM>
        <Info>1 emu/cm³ = 1 kA/m（CGS ↔ SI 変換の恒等式）</Info>

        <Section title="表示単位" />
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

        <Note>膜厚・面積の入力誤差は Ms・Mr の絶対値に直接影響します。ファイル別設定で個別に上書きできます。</Note>
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
          <span className="text-orange-300 font-semibold">線形な常磁性・反磁性成分</span>が重畳しています。
          高磁場領域（磁化が飽和した後）に線形フィッティングを行い、この成分を除去します。
        </p>

        <Section title="補正の式" />
        <BM>{"M_\\mathrm{corrected} = M_\\mathrm{raw} - \\chi \\cdot H"}</BM>
        <p className="text-xs text-zinc-400">ここで <IM>{"\\chi"}</IM> は反磁性補正の傾き（実効的な磁化率）。</p>

        <Section title="自動モード：傾きの算出" />
        <p className="text-xs text-zinc-400 mb-1">全データを H でソートし、上位・下位 15 % の領域で最小二乗線形フィッティングを行います。</p>
        <BM>{"\\chi_+ = \\mathrm{slope}\\!\\left(\\left\\{H_i, M_i\\right\\}_{H_i > 0.85\\,H_\\max}\\right)"}</BM>
        <BM>{"\\chi_- = \\mathrm{slope}\\!\\left(\\left\\{H_i, M_i\\right\\}_{H_i < 0.85\\,H_\\min}\\right)"}</BM>
        <BM>{"\\chi = \\frac{\\chi_+ + \\chi_-}{2}"}</BM>
        <Note>R² &lt; 0.99 の場合は警告を表示します。飽和が不十分、または非線形成分がある可能性があります。</Note>

        <Section title="手動モード：傾きの算出" />
        <p className="text-xs text-zinc-400 mb-1">ユーザー指定の磁場範囲 <IM>{"[H_a,\\, H_b]"}</IM> 内のデータ点のみを使ってフィッティングします。</p>
        <BM>{"\\chi_+ = \\mathrm{slope}\\!\\left(\\left\\{H_i, M_i\\right\\}_{H_a \\le H_i \\le H_b}\\right)"}</BM>
        <Info>データ点が 5 点未満の範囲を指定すると精度が低下します。ログで点数を確認してください。</Info>
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
          測定器のドリフトにより、ヒステリシスループが M 軸方向に全体的にシフトすることがあります。
          高磁場端の磁化平均を使い、ループが原点を中心に上下対称になるよう補正します。
        </p>

        <Section title="オフセット値の算出" />
        <BM>{"\\bar{M}_+ = \\left\\langle M_i \\right\\rangle_{H_i > 0.9\\,H_\\max}"}</BM>
        <BM>{"\\bar{M}_- = \\left\\langle M_i \\right\\rangle_{H_i < 0.9\\,H_\\min}"}</BM>
        <BM>{"\\delta = \\frac{\\bar{M}_+ + \\bar{M}_-}{2}"}</BM>

        <Section title="補正の適用" />
        <BM>{"M_\\mathrm{final} = M_\\mathrm{corrected} - \\delta"}</BM>

        <Note>
          交換バイアス系など、ループが意図的に非対称な試料ではオフセット補正を無効にしてください。
          補正すると見かけのシフト量が失われます。
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
          飽和磁化 <IM>{"M_s"}</IM> は、磁場が十分大きいときに磁化が達する最大値です。
          正側・負側それぞれの高磁場領域の平均から算出します。
        </p>

        <Section title="自動モード" />
        <p className="text-xs text-zinc-400 mb-1">最大磁場の 90 % 以上の領域を使用します。</p>
        <BM>{"M_{s,+} = \\left\\langle M_i \\right\\rangle_{H_i > 0.9\\,H_\\max}"}</BM>
        <BM>{"M_{s,-} = \\left\\langle |M_i| \\right\\rangle_{H_i < 0.9\\,H_\\min}"}</BM>
        <BM>{"M_s = \\frac{M_{s,+} + M_{s,-}}{2}"}</BM>

        <Section title="手動モード" />
        <p className="text-xs text-zinc-400 mb-1">ユーザー指定の磁場範囲 <IM>{"[H_a,\\, H_b]"}</IM>（正側・負側それぞれ）で平均を取ります。</p>
        <BM>{"M_{s,+} = \\left\\langle M_i \\right\\rangle_{H_a \\le H_i \\le H_b}"}</BM>
        <BM>{"M_{s,-} = \\left\\langle |M_i| \\right\\rangle_{H_c \\le H_i \\le H_d}"}</BM>

        <Note>
          磁場が不足して飽和していない場合、自動モードでは <IM>{"M_s"}</IM> が過小評価されます。
          その際は手動で高磁場端に近い範囲を指定してください。
        </Note>
        <Info>
          正側と負側のどちらか一方のデータしか得られない場合、存在する側の値のみを <IM>{"M_s"}</IM> として使用します。
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
          残留磁化 <IM>{"M_r"}</IM> は、印加磁場をゼロに戻したときに残る磁化の大きさ（<IM>{"H=0"}</IM> における <IM>{"|M|"}</IM>）です。
        </p>

        <Section title="ブランチ分割" />
        <p className="text-xs text-zinc-400 mb-1">
          全データを H の符号変化点で分割し、往路（H 減少方向 ↓）と復路（H 増加方向 ↑）に分けます。
        </p>

        <Section title="線形補間による H=0 の M の推定" />
        <p className="text-xs text-zinc-400 mb-1"><code className="text-indigo-300 text-xs bg-zinc-800 px-1 rounded">numpy.interp</code> を使い、<IM>{"H=0"}</IM> を挟む 2 点間で線形補間します。</p>
        <BM>{"M_r^\\downarrow = M^\\downarrow(H = 0) \\quad \\text{（往路）}"}</BM>
        <BM>{"M_r^\\uparrow = M^\\uparrow(H = 0) \\quad \\text{（復路）}"}</BM>

        <Section title="平均値の算出" />
        <BM>{"M_r = \\frac{|M_r^\\downarrow| + |M_r^\\uparrow|}{2}"}</BM>

        <Note>
          測定ノイズが大きいと H=0 付近でブランチの割り当てが不安定になる場合があります。
          ログの「往路/復路データ点数」を確認してください。
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
          保磁力 <IM>{"H_c"}</IM> は、磁化がゼロになるときの磁場の絶対値（<IM>{"M=0"}</IM> における <IM>{"|H|"}</IM>）です。
          磁石の「保磁力（硬さ）」を表す最重要パラメータのひとつです。
        </p>

        <Section title="線形補間による M=0 の H の推定" />
        <p className="text-xs text-zinc-400 mb-1">Mr と同様に往路・復路に分割した後、<IM>{"M=0"}</IM> を挟む 2 点間で線形補間します。</p>
        <BM>{"H_c^\\downarrow = H^\\downarrow(M = 0) \\quad \\text{（往路）}"}</BM>
        <BM>{"H_c^\\uparrow = H^\\uparrow(M = 0) \\quad \\text{（復路）}"}</BM>

        <Section title="平均値の算出" />
        <BM>{"H_c = \\frac{|H_c^\\downarrow| + |H_c^\\uparrow|}{2}"}</BM>

        <Section title="単位換算" />
        <BM>{"H_c\\ [\\mathrm{Oe}] = H_c\\ [\\mathrm{T}] \\times 10^4"}</BM>
        <BM>{"\\mu_0 H_c\\ [\\mathrm{mT}] = H_c\\ [\\mathrm{T}] \\times 10^3"}</BM>

        <Info>
          保磁力は往路・復路の非対称性により左右で値が異なることがあります（交換バイアス等）。
          ログには <IM>{"H_c^\\downarrow"}</IM> と <IM>{"H_c^\\uparrow"}</IM> の個別値も記録されます。
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
          飽和磁場 <IM>{"H_s"}</IM> は、磁化が飽和に達するのに必要な最小磁場です。
          孤立した外れ値に強い「連続点アルゴリズム」を使用しています。
        </p>

        <Section title="閾値の設定" />
        <BM>{"M_\\mathrm{th} = M_s \\left(1 - \\frac{\\varepsilon}{100}\\right)"}</BM>
        <p className="text-xs text-zinc-400">デフォルト: <IM>{"\\varepsilon = 2\\%"}</IM>（サイドバーの「解析設定」から変更可）</p>

        <Section title="アルゴリズム（各ブランチで独立に実行）" />
        <div className="space-y-2 text-sm">
          {[
            ["① ソート", "データを |H| の昇順（小さい磁場から大きい磁場へ）に並び替える。"],
            ["② スキャン", "|M| < M_th の点を「未飽和」と判定しながら走査。"],
            ["③ ランの検出", "未飽和が n_c 点（デフォルト 3 点）以上連続する「最後のまとまり」を特定。"],
            ["④ Hs の決定", "そのまとまりの直後の |H| を Hs とする。"],
          ].map(([label, desc]) => (
            <div key={label} className="flex gap-3">
              <span className="shrink-0 text-xs font-mono text-indigo-400 w-14">{label}</span>
              <span className="text-xs text-zinc-400">{desc}</span>
            </div>
          ))}
        </div>

        <Section title="正側・負側の算出" />
        <p className="text-xs text-zinc-400 mb-1">往路の <IM>{"H>0"}</IM> 領域（正側）と復路の <IM>{"H<0"}</IM> 領域（負側）でそれぞれ <IM>{"H_s"}</IM> を求め、平均します。</p>
        <BM>{"H_s = \\frac{H_{s,+} + H_{s,-}}{2}"}</BM>

        <Note>
          ノイズが大きいデータでは <IM>{"n_c"}</IM>（min_consecutive）を増やすと安定します。
          測定範囲内で飽和に到達しない場合は Hs が算出されません。
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
          角形比 <IM>{"S"}</IM>（スクエアネス）は、ヒステリシスループの「四角さ」を表す無次元指標です。
          磁気記録媒体や永久磁石の性能評価に使われます。
        </p>

        <Section title="定義" />
        <BM>{"S = \\frac{M_r}{M_s} \\quad (0 \\le S \\le 1)"}</BM>

        <Section title="値の解釈" />
        <div className="space-y-2 mt-2">
          {[
            { range: "S \\approx 1", color: "text-emerald-400 bg-emerald-900/20 border-emerald-800/30", title: "理想的な角形ループ", desc: "一軸磁気異方性が強く、スイッチングが急峻。残留磁化が飽和磁化に近い。垂直磁気記録媒体・永久磁石に有利。" },
            { range: "S \\approx 0.5", color: "text-amber-400 bg-amber-900/20 border-amber-800/30", title: "等方性（面内多結晶）", desc: "結晶方位がランダムに分布した多結晶体や面内等方性薄膜で見られる。" },
            { range: "S \\approx 0", color: "text-red-400 bg-red-900/20 border-red-800/30", title: "難磁化軸方向", desc: "印加磁場が難磁化軸に平行な場合、またはソフト磁性材料で磁場印加方向と易軸が揃っていない場合。" },
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

        <Note>S &gt; 1 は <IM>{"M_s"}</IM> の過小評価（飽和不足・範囲設定ミス）を示している可能性があります。</Note>
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
    <div className="fixed inset-0 z-[9998] flex items-center justify-center bg-black/70">
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
          {/* KaTeX のカラーをダークテーマに合わせる */}
          <div className="[&_.katex]:text-zinc-100 [&_.katex-display]:overflow-x-auto [&_.katex-display]:py-1">
            <Content />
          </div>
        </main>
      </div>
    </div>,
    document.body,
  );
}
