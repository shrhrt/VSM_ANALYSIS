/**
 * 凡例名に使われる $...$ TeX 記法を Unicode 文字列に変換する。
 * MathJax なしで動作し、物理・磁性系で一般的な記号に対応。
 */

const GREEK: Record<string, string> = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε",
  varepsilon: "ε", zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ",
  iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ",
  pi: "π", varpi: "ϖ", rho: "ρ", varrho: "ϱ", sigma: "σ",
  varsigma: "ς", tau: "τ", upsilon: "υ", phi: "φ", varphi: "φ",
  chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ",
  Pi: "Π", Sigma: "Σ", Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
};

const MISC: Record<string, string> = {
  mu_0: "μ₀", hbar: "ℏ", infty: "∞", cdot: "·", times: "×",
  div: "÷", pm: "±", mp: "∓", leq: "≤", geq: "≥", neq: "≠",
  approx: "≈", propto: "∝", in: "∈", nabla: "∇", partial: "∂",
  circ: "°", degree: "°", AA: "Å", angstrom: "Å",
};

const SUB: Record<string, string> = {
  "0":"₀","1":"₁","2":"₂","3":"₃","4":"₄",
  "5":"₅","6":"₆","7":"₇","8":"₈","9":"₉",
  "a":"ₐ","e":"ₑ","i":"ᵢ","o":"ₒ","u":"ᵤ",
  "n":"ₙ","m":"ₘ","r":"ᵣ","s":"ₛ","t":"ₜ",
  "+":"₊","-":"₋","=":"₌","(":"₍",")":"₎",
};

const SUP: Record<string, string> = {
  "0":"⁰","1":"¹","2":"²","3":"³","4":"⁴",
  "5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹",
  "n":"ⁿ","i":"ⁱ","a":"ᵃ","b":"ᵇ","c":"ᶜ",
  "+":"⁺","-":"⁻","=":"⁼","(":"⁽",")":"⁾",
};

function mapChars(s: string, map: Record<string, string>): string {
  return [...s].map((c) => map[c] ?? c).join("");
}

function convertExpr(expr: string): string {
  let s = expr.trim();

  // \command{...} or \command → lookup table
  s = s.replace(/\\([a-zA-Z]+)(?:\{([^}]*)\})?/g, (_, cmd, arg) => {
    if (MISC[cmd]) return MISC[cmd];
    if (GREEK[cmd]) {
      // e.g. \mu{0} → μ₀ (treat arg as subscript)
      return GREEK[cmd] + (arg ? mapChars(arg, SUB) : "");
    }
    // \text{...}
    if (cmd === "text" || cmd === "mathrm" || cmd === "mathbf" || cmd === "mathit") {
      return arg ?? "";
    }
    return arg ? arg : `\\${cmd}`;
  });

  // _{...} or _x
  s = s.replace(/_\{([^}]*)\}/g, (_, inner) => mapChars(inner, SUB));
  s = s.replace(/_([0-9a-zA-Z+\-])/g, (_, c) => SUB[c] ?? `_${c}`);

  // ^{...} or ^x
  s = s.replace(/\^\{([^}]*)\}/g, (_, inner) => mapChars(inner, SUP));
  s = s.replace(/\^([0-9a-zA-Z+\-])/g, (_, c) => SUP[c] ?? `^${c}`);

  // Remove remaining braces
  s = s.replace(/[{}]/g, "");

  return s;
}

/**
 * テキスト内の $...$ を Unicode に変換して返す。
 * $...$ 以外の部分はそのまま。
 */
export function texToDisplay(text: string): string {
  if (!text.includes("$")) return text;
  return text.replace(/\$([^$]*)\$/g, (_, expr) => convertExpr(expr));
}
