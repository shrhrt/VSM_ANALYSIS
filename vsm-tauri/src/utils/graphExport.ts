import { save } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";

export const GRAPH_DIV_ID = "vsm-main-plot";

export interface ExportOptions {
  format:        "png" | "svg" | "jpeg";
  scale:         number;
  useCustomSize: boolean;
  width:         number;
  height:        number;
}

async function getGraphSVG(): Promise<{ svg: SVGSVGElement; w: number; h: number }> {
  const gd = document.getElementById(GRAPH_DIV_ID);
  if (!gd) throw new Error("グラフ要素が見つかりません。ファイルを読み込んでください。");

  const svg = gd.querySelector<SVGSVGElement>(".main-svg");
  if (!svg) throw new Error("グラフの SVG が見つかりません。");

  const rect = svg.getBoundingClientRect();
  return { svg, w: rect.width || 800, h: rect.height || 600 };
}

export async function exportGraphBlob(opts: ExportOptions): Promise<Blob> {
  const { svg, w: srcW, h: srcH } = await getGraphSVG();

  const outW = opts.useCustomSize && opts.width  > 0 ? opts.width  : Math.round(srcW * opts.scale);
  const outH = opts.useCustomSize && opts.height > 0 ? opts.height : Math.round(srcH * opts.scale);

  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("width",  String(outW));
  clone.setAttribute("height", String(outH));
  clone.setAttribute("xmlns",  "http://www.w3.org/2000/svg");

  const svgData = new XMLSerializer().serializeToString(clone);
  const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });

  if (opts.format === "svg") return svgBlob;

  const url = URL.createObjectURL(svgBlob);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width  = outW;
      canvas.height = outH;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = "#09090b";
      ctx.fillRect(0, 0, outW, outH);
      ctx.drawImage(img, 0, 0, outW, outH);
      URL.revokeObjectURL(url);
      const mime = opts.format === "jpeg" ? "image/jpeg" : "image/png";
      canvas.toBlob(
        (blob) => blob ? resolve(blob) : reject(new Error("画像変換に失敗しました")),
        mime, 0.95
      );
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("SVG の変換に失敗しました")); };
    img.src = url;
  });
}

/** 保存したら true、キャンセルなら false を返す */
export async function downloadGraphImage(opts: ExportOptions): Promise<boolean> {
  const blob = await exportGraphBlob(opts);

  const defaultName = `vsm_${new Date().toISOString().slice(0, 10)}.${opts.format}`;

  const filePath = await save({
    defaultPath: defaultName,
    filters: [{ name: opts.format.toUpperCase(), extensions: [opts.format] }],
  });
  if (!filePath) return false; // キャンセル

  const arrayBuf = await blob.arrayBuffer();
  await writeFile(filePath, new Uint8Array(arrayBuf));
  return true;
}

export async function copyGraphToClipboard(scale = 2): Promise<void> {
  const blob = await exportGraphBlob({
    format: "png", scale,
    useCustomSize: false, width: 0, height: 0,
  });
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}
