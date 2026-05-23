# -*- coding: utf-8 -*-
import pytest
from app.vsm_app import _make_sort_key


# --- 数値列のソートキー ---

def test_numeric_smaller_value_sorts_first():
    assert _make_sort_key("100.0", "ms") < _make_sort_key("200.0", "ms")


def test_numeric_negative_value():
    assert _make_sort_key("-50.0", "hc") < _make_sort_key("0.0", "hc")


def test_numeric_na_sorts_last():
    assert _make_sort_key("100.0", "ms") < _make_sort_key("N/A", "ms")


def test_numeric_na_vs_na_equal():
    assert _make_sort_key("N/A", "ms") == _make_sort_key("N/A", "ms")


# --- テキスト列（filename）のソートキー ---

def test_filename_alphabetical():
    assert _make_sort_key("abc.vsm", "filename") < _make_sort_key("xyz.vsm", "filename")


def test_filename_na_sorts_last():
    assert _make_sort_key("abc.vsm", "filename") < _make_sort_key("N/A", "filename")


# --- 列種別の区別 ---

def test_numeric_col_uses_float_comparison():
    # "9.0" > "10.0" は文字列比較だと逆転するが、数値比較なら正しい
    assert _make_sort_key("9.0", "ms") < _make_sort_key("10.0", "ms")


def test_non_numeric_col_uses_string_comparison():
    # filename 列は文字列比較
    assert _make_sort_key("9_sample", "filename") > _make_sort_key("10_sample", "filename")
