import { save } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";

export const GRAPH_DIV_ID = "vsm-main-plot";

export interface ExportOptions {
  format:        "png" | "svg" | "jpeg" | "pdf";
  scale:         number;
  useCustomSize: boolean;
  width:         number;
  height:        number;
}

const BG      = "#ffffff";
const PDF_DPI = 300;
const NS      = "http://www.w3.org/2000/svg";

function getContainer(): HTMLElement {
  const el = document.getElementById(GRAPH_DIV_ID);
  if (!el) throw new Error("グラフが見つかりません。ファイルを読み込んでください。");
  return el;
}

/**
 * Plotly が描画する全 .main-svg を重ね合わせた合成 SVG 文字列を返す。
 * infolayer（軸タイトル・凡例）は 2 枚目の SVG にあるため、
 * querySelectorAll で全て取得して 1 枚に合成する。
 */
function buildCompositeSvg(container: HTMLElement, outW: number, outH: number): string {
  const cRect = container.getBoundingClientRect();
  const srcW  = cRect.width  || 800;
  const srcH  = cRect.height || 600;

  const svgs = Array.from(container.querySelectorAll<SVGSVGElement>(".main-svg"));
  if (svgs.length === 0) throw new Error("グラフの SVG が見つかりません");

  const root = document.createElementNS(NS, "svg") as SVGSVGElement;
  root.setAttribute("xmlns",       NS);
  root.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  root.setAttribute("width",       String(outW));
  root.setAttribute("height",      String(outH));
  root.setAttribute("viewBox",     `0 0 ${srcW} ${srcH}`);

  // 白背景
  const bg = document.createElementNS(NS, "rect");
  bg.setAttribute("width",  "100%");
  bg.setAttribute("height", "100%");
  bg.setAttribute("fill",   BG);
  root.appendChild(bg);

  for (const svg of svgs) {
    const sRect = svg.getBoundingClientRect();
    const dx    = sRect.left - cRect.left;
    const dy    = sRect.top  - cRect.top;

    const g = document.createElementNS(NS, "g");
    if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
      g.setAttribute("transform", `translate(${dx.toFixed(1)},${dy.toFixed(1)})`);
    }

    // SVG の子ノードをクローンして g に追加
    const clone = svg.cloneNode(true) as SVGSVGElement;
    Array.from(clone.childNodes).forEach(n => g.appendChild(n));
    root.appendChild(g);
  }

  return new XMLSerializer().serializeToString(root);
}

/** SVG 文字列 → Canvas → PNG/JPEG Blob
 *  blob URL はTauri WebViewのCSPで弾かれる場合があるため base64 data URL を使用
 */
function svgToCanvasBlob(
  svgStr: string, outW: number, outH: number, mime: "image/png" | "image/jpeg",
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    // Unicode対応の base64 エンコード
    const b64     = btoa(unescape(encodeURIComponent(svgStr)));
    const dataUrl = `data:image/svg+xml;base64,${b64}`;

    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width  = outW;
        canvas.height = outH;
        const ctx = canvas.getContext("2d");
        if (!ctx) { reject(new Error("Canvas コンテキスト取得失敗")); return; }
        ctx.fillStyle = BG;
        ctx.fillRect(0, 0, outW, outH);
        ctx.drawImage(img, 0, 0, outW, outH);
        canvas.toBlob(
          b => b ? resolve(b) : reject(new Error("Canvas.toBlob が null を返しました")),
          mime, 0.95,
        );
      } catch (e) {
        reject(new Error(`Canvas 描画エラー: ${(e as Error).message}`));
      }
    };
    img.onerror = () => reject(new Error("SVG を Image に読み込めませんでした"));
    img.src = dataUrl;
  });
}

export async function exportGraphBlob(opts: ExportOptions): Promise<Blob> {
  const container = getContainer();
  const cRect     = container.getBoundingClientRect();
  const outW = opts.useCustomSize && opts.width  > 0 ? opts.width  : Math.round((cRect.width  || 800) * opts.scale);
  const outH = opts.useCustomSize && opts.height > 0 ? opts.height : Math.round((cRect.height || 600) * opts.scale);

  const svgStr = buildCompositeSvg(container, outW, outH);

  if (opts.format === "svg") {
    return new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
  }

  if (opts.format === "png" || opts.format === "jpeg") {
    const mime = opts.format === "jpeg" ? "image/jpeg" : "image/png";
    return svgToCanvasBlob(svgStr, outW, outH, mime);
  }

  if (opts.format === "pdf") {
    // PNG を Canvas で生成してから jsPDF に埋め込む
    const pngBlob = await svgToCanvasBlob(svgStr, outW, outH, "image/png");
    const pngB64  = await new Promise<string>((res) => {
      const reader = new FileReader();
      reader.onloadend = () => res(reader.result as string);
      reader.readAsDataURL(pngBlob);
    });
    const { jsPDF } = await import("jspdf");
    const pxPerMm  = PDF_DPI / 25.4;
    const wMm      = outW / pxPerMm;
    const hMm      = outH / pxPerMm;
    const doc      = new jsPDF({ unit: "mm", format: [wMm, hMm], orientation: wMm >= hMm ? "l" : "p" });
    doc.addImage(pngB64, "PNG", 0, 0, wMm, hMm, undefined, "FAST");
    return new Blob([doc.output("arraybuffer")], { type: "application/pdf" });
  }

  throw new Error("未対応のフォーマット");
}

/** プレビュー用: 合成 SVG の blob URL を返す（PNG より高速） */
export async function getPreviewUrl(outW: number, outH: number): Promise<string> {
  const container = getContainer();
  // プレビューは 700px に縮小
  const scale = Math.min(1, 700 / Math.max(outW, outH, 1));
  const svgStr = buildCompositeSvg(container, Math.round(outW * scale), Math.round(outH * scale));
  const blob   = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
  return URL.createObjectURL(blob);
}

/** 保存したら true、キャンセルなら false */
export async function downloadGraphImage(opts: ExportOptions): Promise<boolean> {
  // 1) グラフを Blob に変換
  let blob: Blob;
  try {
    blob = await exportGraphBlob(opts);
  } catch (e) {
    throw new Error(`グラフ変換失敗: ${(e as Error).message}`);
  }

  // 2) 保存先を選択
  const ext  = opts.format;
  const name = `vsm_${new Date().toISOString().slice(0, 10)}.${ext}`;
  let filePath: string | null;
  try {
    filePath = await save({
      defaultPath: name,
      filters: [{ name: ext.toUpperCase(), extensions: [ext] }],
    });
  } catch (e) {
    throw new Error(`ダイアログエラー: ${(e as Error).message}`);
  }
  if (!filePath) return false;  // キャンセル

  // 3) ファイルに書き込み
  try {
    await writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
  } catch (e) {
    // Tauri バックエンドは string を throw することがある
    const msg = e instanceof Error ? e.message : (typeof e === "string" ? e : JSON.stringify(e));
    throw new Error(`書き込み失敗: ${msg}`);
  }
  return true;
}

export async function copyGraphToClipboard(
  scale = 2, width?: number, height?: number,
): Promise<void> {
  const blob = await exportGraphBlob({
    format: "png", scale,
    useCustomSize: !!(width && height),
    width:  width  ?? 0,
    height: height ?? 0,
  });
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}
