# -*- coding: utf-8 -*-
from analysis.tex_utils import tex_to_display


def test_dollar_removed():
    assert tex_to_display("$Fe_3O_4$") == "Fe₃O₄"

def test_greek_gamma():
    assert tex_to_display(r"$\gamma$") == "γ"

def test_greek_mixed_text():
    assert tex_to_display(r"$\gamma$-Fe") == "γ-Fe"

def test_subscript_braces_multidigit():
    assert tex_to_display(r"$Co_{50}Fe_{50}$") == "Co₅₀Fe₅₀"

def test_subscript_single_digit():
    assert tex_to_display(r"$H_2O$") == "H₂O"

def test_superscript_braces():
    assert tex_to_display(r"$Fe^{2+}$") == "Fe²⁺"

def test_superscript_single_digit():
    assert tex_to_display(r"$X^3$") == "X³"

def test_no_tex_unchanged():
    # アンダースコアの後ろが英字のときは変換しない
    assert tex_to_display("sample_A") == "sample_A"

def test_plain_text_unchanged():
    assert tex_to_display("permalloy") == "permalloy"

def test_complex_formula():
    # 典型的なサンプル名
    assert tex_to_display(r"$\gamma$-Fe$_2$O$_3$") == "γ-Fe₂O₃"

def test_capital_greek():
    assert tex_to_display(r"$\Sigma$") == "Σ"

def test_superscript_plus_minus():
    assert tex_to_display(r"$Fe^{3+}$") == "Fe³⁺"
