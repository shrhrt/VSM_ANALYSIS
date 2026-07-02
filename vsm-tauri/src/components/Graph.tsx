import Plot from "react-plotly.js";
import type { FileEntry, UnitMode, GraphSettings, PaperColorScheme } from "../App";
import { texToDisplay } from "../utils/texToDisplay";

interface Props {
  entries:       FileEntry[];
  unitMode:      UnitMode;
  graphSettings: GraphSettings;
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

export default function Graph({ entries, unitMode, graphSettings }: Props) {
  const hasData = entries.some((e) => e.result?.plot);
  const defs    = defaultLabels(unitMode);
  const xLabel  = graphSettings.xLabelOverride || defs.x;
  const yLabel  = graphSettings.yLabelOverride || defs.y;

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

  const traces = hasData
    ? entries.flatMap((e, idx) => {
        if (!e.result?.plot) return [];
        const { H_down, M_down, H_up, M_up } = e.result.plot;
        const { H, M } = convertAxes(
          [...H_down, ...H_up], [...M_down, ...M_up], e.result.Ms, unitMode
        );
        const color = (paper && paperColorScheme !== "current" && schemeColors.length > 0)
          ? schemeColors[idx % schemeColors.length]
          : e.color;
        return [{
          x: H, y: M,
          type: "scatter" as const,
          mode: plotMode as "lines" | "lines+markers",
          name: texToDisplay(e.legendName || e.file.name.replace(/\.[^.]+$/, "")),
          line:   { color, width: lineWidth },
          marker: { color, size: markerSize, symbol: e.markerSymbol ?? markerSymbol },
        }];
      })
    : [{
        x: [-2, -1, 0, 1, 2], y: [0, 0, 0, 0, 0],
        type: "scatter" as const, mode: "lines" as const,
        name: "（データなし）", line: { color: paper ? "#CCCCCC" : "#3f3f46", width: 1 },
      }];

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
          data={traces}
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
