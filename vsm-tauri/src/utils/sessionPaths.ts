// Session path computation utilities (OS-agnostic, handles both / and \)

function normSep(p: string): string {
  return p.replace(/\\/g, "/");
}

function dirOf(filePath: string): string {
  const p = normSep(filePath);
  const idx = p.lastIndexOf("/");
  return idx > 0 ? p.slice(0, idx) : ".";
}

/**
 * Compute a relative path from a session file to a VSM file.
 * e.g. sessionPath = "C:/sessions/my.vsm_session"
 *      filePath    = "C:/data/vsm/sample.VSM"
 *   →  "../../data/vsm/sample.VSM"
 */
export function computeRelativePath(sessionPath: string, filePath: string): string {
  const fromDir = normSep(dirOf(sessionPath));
  const target  = normSep(filePath);

  const fromParts = fromDir.split("/").filter(Boolean);
  const toParts   = target.split("/").filter(Boolean);

  // Case-insensitive comparison for Windows drive letters / folder names
  let common = 0;
  while (
    common < fromParts.length &&
    common < toParts.length &&
    fromParts[common].toLowerCase() === toParts[common].toLowerCase()
  ) common++;

  const ups  = fromParts.length - common;
  const down = toParts.slice(common);
  return [...Array(ups).fill(".."), ...down].join("/") || "./";
}

/**
 * Compute the OneDrive-relative path for a file.
 * Returns null if the file is not under the given OneDrive root.
 */
export function computeOnedrivePath(filePath: string, onedriveRoot: string): string | null {
  if (!onedriveRoot) return null;
  const norm     = normSep(filePath);
  const normRoot = normSep(onedriveRoot).replace(/\/$/, "");
  if (norm.toLowerCase().startsWith(normRoot.toLowerCase())) {
    return norm.slice(normRoot.length).replace(/^\//, "");
  }
  return null;
}
