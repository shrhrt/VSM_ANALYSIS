import Plot from "react-plotly.js";
import type { FileEntry, UnitMode } from "../App";

const COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"];

interface Props {
  entries:  FileEntry[];
  unitMode: UnitMode;
}

function convertAxes(
  H_T: number[], M_kAm: number[], Ms: number | null | undefined, mode: UnitMode
): { H: number[]; M: number[] } {
  switch (mode) {
    case "CGS":
      return { H: H_T.map((h) => h * 10000), M: M_kAm };   // T→Oe, kA/m≡emu/cm³
    case "Normalized":
      return {
        H: H_T,
        M: Ms ? M_kAm.map((m) => m / Ms) : M_kAm,
      };
    default: // SI
      return { H: H_T, M: M_kAm };
  }
}

function axisLabels(mode: UnitMode): { x: string; y: string } {
  switch (mode) {
    case "CGS":        return { x: "H (Oe)",     y: "M (emu/cm³)" };
    case "Normalized": return { x: "μ₀H (T)",    y: "M/Ms" };
    default:           return { x: "μ₀H (T)",    y: "M (kA/m)" };
  }
}

export default function Graph({ entries, unitMode }: Props) {
  const hasData = entries.some((e) => e.result?.plot);
  const { x: xLabel, y: yLabel } = axisLabels(unitMode);

  const data = hasData
    ? entries.flatMap((e, i) => {
        if (!e.result?.plot) return [];
        const { H_down, M_down, H_up, M_up } = e.result.plot;
        const Ms = e.result.Ms;
        const name = e.file.name.replace(/\.VSM$/i, "");
        const color = COLORS[i % COLORS.length];

        // 降磁場 → 昇磁場 を繋げて1本のループとして表示
        const H_all = [...H_down, ...H_up];
        const M_all = [...M_down, ...M_up];
        const { H, M } = convertAxes(H_all, M_all, Ms, unitMode);

        return [{
          x: H, y: M,
          type: "scatter" as const,
          mode: "lines" as const,
          name,
          line: { color, width: 2 },
        }];
      })
    : [
        { x: [-2, -1, 0, 1, 2], y: [0, 0, 0, 0, 0],
          type: "scatter" as const, mode: "lines" as const,
          name: "（データなし）", line: { color: "#3f3f46", width: 1 } },
      ];

  return (
    <div className="flex-1 flex items-center justify-center bg-zinc-950 p-4 min-h-0">
      <Plot
        data={data}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "#09090b",
          font: { color: "#a1a1aa", family: "Inter, sans-serif", size: 12 },
          xaxis: { title: { text: xLabel, font: { size: 13 } },
                   gridcolor: "#27272a", zerolinecolor: "#52525b" },
          yaxis: { title: { text: yLabel, font: { size: 13 } },
                   gridcolor: "#27272a", zerolinecolor: "#52525b" },
          legend: { bgcolor: "rgba(0,0,0,0)", bordercolor: "#3f3f46" },
          margin: { t: 30, r: 20, b: 60, l: 70 },
          autosize: true,
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{ displaylogo: false, responsive: true }}
      />
    </div>
  );
}
