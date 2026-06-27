import Plot from "react-plotly.js";
import type { FileEntry, UnitMode, GraphSettings } from "../App";

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
    case "CGS":        return { x: "H (Oe)",  y: "M (emu/cm³)" };
    case "Normalized": return { x: "μ₀H (T)", y: "M / Ms" };
    default:           return { x: "μ₀H (T)", y: "M (kA/m)" };
  }
}

const LEGEND_POS: Record<string, { x: number; y: number; xanchor: string; yanchor: string }> = {
  "top-right":    { x: 1, y: 1, xanchor: "right",  yanchor: "top"    },
  "top-left":     { x: 0, y: 1, xanchor: "left",   yanchor: "top"    },
  "bottom-right": { x: 1, y: 0, xanchor: "right",  yanchor: "bottom" },
  "bottom-left":  { x: 0, y: 0, xanchor: "left",   yanchor: "bottom" },
};

// Plotly dash style mapping
const DASH: Record<string, string> = {
  solid: "solid", dot: "dot", dash: "dash",
};

function toRange(min: string, max: string): [number, number] | undefined {
  const lo = parseFloat(min);
  const hi = parseFloat(max);
  return !isNaN(lo) && !isNaN(hi) ? [lo, hi] : undefined;
}

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
    zeroLineColor, zeroLineStyle,
    gridColor, gridStyle,
  } = graphSettings;

  const plotMode = markerSize > 0 ? "lines+markers" : "lines";
  const legendAnchor = LEGEND_POS[legendPosition] ?? LEGEND_POS["top-right"];
  const xRange = toRange(xMin, xMax);
  const yRange = toRange(yMin, yMax);

  const traces = hasData
    ? entries.flatMap((e) => {
        if (!e.result?.plot) return [];
        const { H_down, M_down, H_up, M_up } = e.result.plot;
        const { H, M } = convertAxes(
          [...H_down, ...H_up], [...M_down, ...M_up], e.result.Ms, unitMode
        );
        return [{
          x: H, y: M,
          type: "scatter" as const,
          mode: plotMode as "lines" | "lines+markers",
          name: e.file.name.replace(/\.VSM$/i, ""),
          line:   { color: e.color, width: lineWidth },
          marker: { color: e.color, size: markerSize, symbol: markerSymbol },
        }];
      })
    : [{
        x: [-2, -1, 0, 1, 2], y: [0, 0, 0, 0, 0],
        type: "scatter" as const, mode: "lines" as const,
        name: "（データなし）", line: { color: "#3f3f46", width: 1 },
      }];

  return (
    <div className="flex-1 flex items-center justify-center bg-zinc-950 p-4 min-h-0">
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor:  "#09090b",
          font: { color: "#a1a1aa", family: "Inter, sans-serif", size: tickLabelSize },
          showlegend: showLegend,
          legend: {
            bgcolor:     "rgba(24,24,27,0.85)",
            bordercolor: "#3f3f46",
            borderwidth: 1,
            font:        { size: legendFontSize, color: "#d4d4d8" },
            tracegroupgap: 4,
            ncols: legendColumns,
            ...legendAnchor,
          },
          xaxis: {
            title:         { text: xLabel, font: { size: axisLabelSize } },
            tickfont:      { size: tickLabelSize },
            tickformat:    xTickFormat || undefined,
            gridcolor:     showGrid      ? gridColor    : "transparent",
            griddash:      DASH[gridStyle] as "solid" | "dot" | "dash",
            zerolinecolor: showZeroLines ? zeroLineColor : "transparent",
            zerolinewidth: 1.5,
            zeroline:      showZeroLines,
            zerolinedash:  DASH[zeroLineStyle] as "solid" | "dot" | "dash",
            ...(xRange ? { range: xRange } : {}),
          },
          yaxis: {
            title:         { text: yLabel, font: { size: axisLabelSize } },
            tickfont:      { size: tickLabelSize },
            tickformat:    yTickFormat || undefined,
            gridcolor:     showGrid      ? gridColor    : "transparent",
            griddash:      DASH[gridStyle] as "solid" | "dot" | "dash",
            zerolinecolor: showZeroLines ? zeroLineColor : "transparent",
            zerolinewidth: 1.5,
            zeroline:      showZeroLines,
            zerolinedash:  DASH[zeroLineStyle] as "solid" | "dot" | "dash",
            ...(yRange ? { range: yRange } : {}),
          },
          margin:   { t: 30, r: 20, b: 60, l: 80 },
          autosize: true,
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{ displaylogo: false, responsive: true }}
      />
    </div>
  );
}
