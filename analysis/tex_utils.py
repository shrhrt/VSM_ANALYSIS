# -*- coding: utf-8 -*-
import re

_GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ",
    r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ",
    r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ",
    r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
}

_SUB_TABLE = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
_SUP_TABLE = str.maketrans("0123456789+-n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ")


def tex_to_display(text: str) -> str:
    """TeX記法の文字列をUnicodeに変換して返す。

    変換対象:
      - $...$ デリミタの除去
      - ギリシャ文字 (\\gamma → γ 等)
      - 下付き文字 (_{50} → ₅₀、_2 → ₂)
      - 上付き文字 (^{2+} → ²⁺、^3 → ³)
    """
    # $ デリミタを除去
    text = text.replace("$", "")

    # ギリシャ文字（長い名前から先に置換して部分一致を防ぐ）
    for tex, uni in sorted(_GREEK.items(), key=lambda x: -len(x[0])):
        text = text.replace(tex, uni)

    # 下付き: _{...} または _数字
    text = re.sub(r"_\{([^}]+)\}", lambda m: m.group(1).translate(_SUB_TABLE), text)
    text = re.sub(r"_([0-9])", lambda m: m.group(1).translate(_SUB_TABLE), text)

    # 上付き: ^{...} または ^数字
    text = re.sub(r"\^\{([^}]+)\}", lambda m: m.group(1).translate(_SUP_TABLE), text)
    text = re.sub(r"\^([0-9])", lambda m: m.group(1).translate(_SUP_TABLE), text)

    # 未知のコマンド (\\rm 等) を除去
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    return text
