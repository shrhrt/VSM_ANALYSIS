import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import type { FileEntry, UnitMode, GraphSettings, PaperColorScheme } from "../App";
import { texToDisplay } from "../utils/texToDisplay";

interface Props {
  entries:          FileEntry[];
  unitMode:         UnitMode;
  graphSettings:    GraphSettings;
  onToggleExclude?: (entryIndex: number, origIdx: number) => void;
}

function convertAxes(
  H_T: number[], M_kAm: number[], Ms: number | null | undefined, mode: UnitMode
): { H: number[]; M: number[] } {
  switch (mode) {
    case "CGS":        return { H: H_T.map((h) => h * 10000), M: M_kAm };
    case "Normalized": return { H: H_T, M: Ms ? M_kAm.map((m) => m / Ms) : M_kAm };
    default:           return { H: H_T, M: M_kAm };
  }
}

function defaultLabels(mode: UnitMode): { x: string; y: string } {
  switch (mode) {
    case "CGS":        return { x: "H (Oe)",      y: "M (emu/cm³)" };
    case "Normalized": return { x: "μ₀H (T)", y: "M / Ms" };
    default:           return { x: "μ₀H (T)", y: "M (kA/m)" };
  }
}

const LEGEND_POS: Record<string, { x: number; y: number; xanchor: string; yanchor: string }> = {
  "top-left":      { x: 0,   y: 1,   xanchor: "left",   yanchor: "top"    },
  "top-center":    { x: 0.5, y: 1,   xanchor: "center", yanchor: "top"    },
  "top-right":     { x: 1,   y: 1,   xanchor: "right",  yanchor: "top"    },
  "mid-left":      { x: 0,   y: 0.5, xanchor: "left",   yanchor: "middle" },
  "center":        { x: 0.5, y: 0.5, xanchor: "center", yanchor: "middle" },
  "mid-right":     { x: 1,   y: 0.5, xanchor: "right",  yanchor: "middle" },
  "bottom-left":   { x: 0,   y: 0,   xanchor: "left",   yanchor: "bottom" },
  "bottom-center": { x: 0.5, y: 0,   xanchor: "center", yanchor: "bottom" },
  "bottom-right":  { x: 1,   y: 0,   xanchor: "right",  yanchor: "bottom" },
};

const DASH: Record<string, string> = {
  solid: "solid", dot: "dot", dash: "dash",
};

function toRange(min: string, max: string): [number, number] | undefined {
  const lo = parseFloat(min);
  const hi = parseFloat(max);
  return !isNaN(lo) && !isNaN(hi) ? [lo, hi] : undefined;
}

// ── 論文用カラースキーム ─────────────────────────────────────
const PAPER_COLORS: Record<PaperColorScheme, string[]> = {
  current:   [],
  journal:   ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"],
  grayscale: ["#000000", "#555555", "#999999", "#222222", "#777777", "#AAAAAA", "#333333", "#BBBBBB"],
};

export default function Graph({ entries, unitMode, graphSettings, onToggleExclude }: Props) {
  const hasData = entries.some((e) => e.result?.plot);
  const defs    = defaultLabels(unitMode);
  const xLabel  = texToDisplay(graphSettings.xLabelOverride || defs.x);
  const yLabel  = texToDisplay(graphSettings.yLabelOverride || defs.y);

  const {
    showLegend, showGrid, showZeroLines,
    lineWidth, markerSize, markerSymbol,
    legendPosition, legendFontSize, legendColumns,
    axisLabelSize, tickLabelSize,
    xTickFormat, yTickFormat,
    xMin, xMax, yMin, yMax,
    xDtick, yDtick,
    showMinorTicks, minorDivisions,
    zeroLineColor, zeroLineStyle,
    gridColor, gridStyle,
    paperMode, paperColorScheme,
    showAnnotations, showExcluded,
    marginB, marginL,
  } = graphSettings;

  const paper = paperMode;
  const schemeColors = PAPER_COLORS[paperColorScheme];

  const theme = {
    paperBg:    paper ? "white"    : "transparent",
    plotBg:     paper ? "white"    : "#09090b",
    fontColor:  paper ? "#111111"  : "#a1a1aa",
    fontFamily: paper ? "Times New Roman, Palatino, Georgia, serif" : "Arial, Helvetica, sans-serif",
    gridCol:    paper ? (showGrid ? "#CCCCCC" : "transparent") : (showGrid ? gridColor : "transparent"),
    zeroCol:    paper ? (showZeroLines ? "#777777" : "transparent") : (showZeroLines ? zeroLineColor : "transparent"),
    legendBg:   paper ? "rgba(255,255,255,0.9)" : "rgba(24,24,27,0.85)",
    legendBord: paper ? "#AAAAAA"  : "#3f3f46",
    legendFont: paper ? "#111111"  : "#d4d4d8",
  };

  const plotMode = markerSize > 0 ? "lines+markers" : "lines";
  const legendAnchor = LEGEND_POS[legendPosition] ?? LEGEND_POS["top-right"];

  const xRaw = toRange(xMin, xMax);
  const yRaw = toRange(yMin, yMax);

  // 論文モード: 境界値の目盛が枠の隅と重ならないようrangeを拡張
  // dtickの20%分広げることで境界目盛がフレーム端から明確に離れる
  const padRange = (r: [number, number] | undefined, dtickStr: string): [number, number] | undefined => {
    if (!r) return undefined;
    const span = r[1] - r[0];
    const dtickVal = dtickStr ? parseFloat(dtickStr) : NaN;
    const pad  = !isNaN(dtickVal) && dtickVal > 0 ? dtickVal * 0.2 : span * 0.03;
    return [r[0] - pad, r[1] + pad];
  };
  const xRange = paper ? padRange(xRaw, xDtick ?? "") : xRaw;
  const yRange = paper ? padRange(yRaw, yDtick ?? "") : yRaw;

  const dtickX = xDtick ? parseFloat(xDtick) : undefined;
  const dtickY = yDtick ? parseFloat(yDtick) : undefined;

  // 補助目盛り設定（論文モード & showMinorTicks 時のみ）
  const buildMinor = (mainDtick: number | undefined) => {
    if (!paper || !showMinorTicks) return {};
    const n = Math.max(2, minorDivisions);
    return {
      minor: {
        ticks:     "inside",
        ticklen:   3,
        tickwidth: 1,
        tickcolor: "#111111",
        ...(mainDtick !== undefined && mainDtick > 0
          ? { dtick: mainDtick / n, tickmode: "linear" as const, tick0: 0 }  // 主間隔をn等分
          : { nticks: n - 1 }),                                                // 自動モード
      },
    };
  };

  const traces: Data[] = hasData
    ? entries.flatMap((e, idx): Data[] => {
        if (!e.result?.plot) return [];
        const { H_down, M_down, H_up, M_up, idx_down, idx_up } = e.result.plot;
        const { H, M } = convertAxes(
          [...H_down, ...H_up], [...M_down, ...M_up], e.result.Ms, unitMode
        );
        const color = (paper && paperColorScheme !== "current" && schemeColors.length > 0)
          ? schemeColors[idx % schemeColors.length]
          : e.color;
        // 各点に [ファイル番号, 元データ行番号] を持たせ、クリックで除外/復帰できるようにする
        const origIdx = [...(idx_down ?? []), ...(idx_up ?? [])];
        const customdata = origIdx.length === H.length ? origIdx.map((oi) => [idx, oi]) : undefined;
        const out: Data[] = [{
          x: H, y: M,
          type: "scatter",
          mode: plotMode as "lines" | "lines+markers",
          name: texToDisplay(e.legendName || e.file.name.replace(/\.[^.]+$/, "")),
          line:   { color, width: lineWidth },
          marker: { color, size: markerSize, symbol: e.markerSymbol ?? markerSymbol },
          ...(customdata ? { customdata } : {}),
        }];
        // 除外点: 灰色×で表示（クリックで復帰）。showExcluded=false なら画面・エクスポート共に非表示
        const exc = e.result.excluded;
        if (exc && exc.idx.length > 0 && showExcluded) {
          const conv = convertAxes(exc.H, exc.M, e.result.Ms, unitMode);
          out.push({
            x: conv.H, y: conv.M,
            type: "scatter", mode: "markers",
            name: "除外点", showlegend: false,
            marker: { color: "#9ca3af", size: Math.max(8, markerSize + 3), symbol: "x", line: { width: 1, color: "#6b7280" } },
            customdata: exc.idx.map((oi) => [idx, oi]),
            hovertemplate: "除外点 (クリックで復帰)<extra></extra>",
          });
        }
        return out;
      })
    : [{
        x: [-2, -1, 0, 1, 2], y: [0, 0, 0, 0, 0],
        type: "scatter", mode: "lines",
        name: "（データなし）", line: { color: paper ? "#CCCCCC" : "#3f3f46", width: 1 },
      }];

  // ── 解析注釈オーバーレイ（Ms 準位・Hc/Mr 交点・Hs）─────────────
  const convHv = (h: number) => (unitMode === "CGS" ? h * 10000 : h);
  const convMv = (m: number, Ms: number | null | undefined) =>
    unitMode === "Normalized" ? (Ms ? m / Ms : m) : m;

  const annotTraces: Data[] = [];
  const annotShapes: NonNullable<Layout["shapes"]> = [];
  if (showAnnotations && hasData) {
    entries.forEach((e, idx) => {
      const a = e.result?.annot;
      if (!a) return;
      const Ms = e.result?.Ms;
      const color = (paper && paperColorScheme !== "current" && schemeColors.length > 0)
        ? schemeColors[idx % schemeColors.length]
        : e.color;
      // Hc 交点 (M=0)
      const hcX: number[] = [], hcY: number[] = [];
      if (a.hc_down_T != null) { hcX.push(convHv(a.hc_down_T)); hcY.push(convMv(0, Ms)); }
      if (a.hc_up_T   != null) { hcX.push(convHv(a.hc_up_T));   hcY.push(convMv(0, Ms)); }
      if (hcX.length) annotTraces.push({
        x: hcX, y: hcY, type: "scatter", mode: "markers",
        name: "Hc", showlegend: false, hoverinfo: "x",
        marker: { color, size: 12, symbol: "x-thin-open", line: { width: 2, color } },
      });
      // Mr 交点 (H=0)
      if (a.mr != null) annotTraces.push({
        x: [convHv(0), convHv(0)], y: [convMv(a.mr, Ms), convMv(-a.mr, Ms)],
        type: "scatter", mode: "markers",
        name: "Mr", showlegend: false, hoverinfo: "y",
        marker: { color, size: 9, symbol: "circle-open", line: { width: 2, color } },
      });
      // Ms 準位（水平線, プロット全幅）
      const msLine = (yv: number) => ({
        type: "line" as const, xref: "paper" as const, x0: 0, x1: 1,
        yref: "y" as const, y0: yv, y1: yv, layer: "below" as const,
        line: { color, width: 1, dash: "dot" as const },
      });
      if (a.ms_pos != null) annotShapes.push(msLine(convMv(a.ms_pos, Ms)));
      if (a.ms_neg != null) annotShapes.push(msLine(convMv(-a.ms_neg, Ms)));
      // Hs（垂直線, プロット全高）
      const hsLine = (xv: number) => ({
        type: "line" as const, yref: "paper" as const, y0: 0, y1: 1,
        xref: "x" as const, x0: xv, x1: xv, layer: "below" as const,
        line: { color, width: 1, dash: "dash" as const },
      });
      if (a.hs_pos_T != null) annotShapes.push(hsLine(convHv(a.hs_pos_T)));
      if (a.hs_neg_T != null) annotShapes.push(hsLine(-convHv(a.hs_neg_T)));
    });
  }
  const allTraces = [...traces, ...annotTraces];

  // 軸共通設定（title/tickfont は個別指定）
  const axisBase = {
    gridcolor:     theme.gridCol,
    griddash:      DASH[gridStyle] as "solid" | "dot" | "dash",
    zerolinecolor: theme.zeroCol,
    zerolinewidth: paper ? 1.0 : 1.5,
    zeroline:      showZeroLines,
    zerolinedash:  DASH[zeroLineStyle] as "solid" | "dot" | "dash",
    ...(paper ? {
      linecolor:  "#111111",
      linewidth:  1.5,
      showline:   true,
      mirror:     "ticks" as const,
      ticks:      "inside" as const,
      tickcolor:  "#111111",
      ticklen:    6,
      tickwidth:  1.5,
    } : {
      linecolor: "#52525b",
      showline:  true,
    }),
  };

  return (
    <div className={`flex-1 flex items-stretch p-3 min-h-0 transition-colors duration-300 ${
      paper ? "bg-gray-300" : "bg-zinc-950"
    }`}>
      <div className="relative flex-1"
        style={paper ? { background: "white", borderRadius: 2, boxShadow: "0 20px 60px rgba(0,0,0,0.5)" } : {}}>
        <Plot
          divId="vsm-main-plot"
          data={allTraces}
          onClick={(ev) => {
            const pt = ev.points?.[0] as { customdata?: unknown } | undefined;
            const cd = pt?.customdata as [number, number] | undefined;
            if (cd && onToggleExclude) onToggleExclude(cd[0], cd[1]);
          }}
          layout={{
            paper_bgcolor: theme.paperBg,
            plot_bgcolor:  theme.plotBg,
            font: { color: theme.fontColor, family: theme.fontFamily, size: tickLabelSize },
            showlegend: showLegend,
            legend: {
              bgcolor:       theme.legendBg,
              bordercolor:   theme.legendBord,
              borderwidth:   1,
              font:          { size: legendFontSize, color: theme.legendFont, family: theme.fontFamily },
              tracegroupgap: 4,
              ncols:         legendColumns,
              ...legendAnchor,
            },
            xaxis: {
              title:          { text: xLabel, font: { size: Math.max(6, axisLabelSize), color: theme.fontColor, family: theme.fontFamily } },
              tickfont:       { size: Math.max(6, tickLabelSize), color: theme.fontColor, family: theme.fontFamily },
              showticklabels: true,
              tickformat:     xTickFormat || undefined,
              ...(dtickX !== undefined ? { dtick: dtickX, tickmode: "linear" as const, tick0: 0 } : {}),
              ...(xRange ? { range: xRange } : {}),
              ...axisBase,
              ...buildMinor(dtickX),
            },
            yaxis: {
              title:          { text: yLabel, font: { size: Math.max(6, axisLabelSize), color: theme.fontColor, family: theme.fontFamily } },
              tickfont:       { size: Math.max(6, tickLabelSize), color: theme.fontColor, family: theme.fontFamily },
              showticklabels: true,
              tickformat:     yTickFormat || undefined,
              ...(dtickY !== undefined ? { dtick: dtickY, tickmode: "linear" as const, tick0: 0 } : {}),
              ...(yRange ? { range: yRange } : {}),
              ...axisBase,
              ...buildMinor(dtickY),
            },
            margin:   { t: 30, r: 40, b: marginB, l: marginL },
            shapes:   annotShapes,
            autosize: true,
          }}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
          config={{ displaylogo: false, responsive: true }}
        />
      </div>
    </div>
  );
}
