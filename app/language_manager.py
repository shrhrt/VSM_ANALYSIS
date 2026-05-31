# -*- coding: utf-8 -*-
import json
from pathlib import Path
import sys
import os
import tkinter as tk
from typing import Dict, Any, TYPE_CHECKING

# 実行時は無視され、静的型チェックツールやIDEの補完時のみインポートされます
if TYPE_CHECKING:
    from app.state_manager import StateManager


class LanguageManager:
    """
    多言語対応（国際化）を管理するクラス。
    JSONファイルから翻訳データを読み込み、指定されたキーに対応する文字列を提供する。
    """

    def __init__(self, app_state: "StateManager") -> None:
        """
        LanguageManagerのインスタンスを初期化します。

        Args:
            app_state (StateManager): アプリケーション全体の状態を管理するオブジェクト。
        """
        self.app_state: "StateManager" = app_state
        self.locales_dir: Path = self._get_base_path() / "locales"
        self.translations: Dict[str, Any] = {}
        self.load_language(self.app_state.language_var.get())

        # Listen for language changes
        self.app_state.language_var.trace_add("write", self._on_language_change)

    def _get_base_path(self) -> Path:
        """
        PyInstaller / Nuitka / 通常実行の全環境でベースパスを取得します。
        """
        if hasattr(sys, "_MEIPASS"):
            # PyInstaller: 一時解凍フォルダ
            return Path(sys._MEIPASS)
        elif "__compiled__" in globals():
            # Nuitka standalone: exe と同じフォルダ
            return Path(sys.executable).parent
        return Path(__file__).parent.parent

    def load_language(self, lang_code: str) -> None:
        """
        指定された言語コードの言語ファイル (例: 'en', 'ja') を読み込みます。

        Args:
            lang_code (str): 読み込む言語のコード。
        """
        filepath = self.locales_dir / f"{lang_code}.json"
        if not filepath.exists():
            print(f"Warning: Language file not found: {filepath}")
            # Fallback to English if the selected language is not found
            filepath = self.locales_dir / "en.json"
            if not filepath.exists():
                print("Error: Default language file 'en.json' not found.")
                self.translations = {}
                return

        with open(filepath, "r", encoding="utf-8") as f:
            self.translations = json.load(f)

    def get(self, key: str, **kwargs: Any) -> str:
        """
        指定されたキーに対応する翻訳文字列を取得します。

        Args:
            key (str): ドット区切りの翻訳キー (例: "menu.file")。
            **kwargs: 翻訳文字列のプレースホルダーに埋め込む値。

        Returns:
            str: 翻訳された文字列。キーが見つからない場合はキー自身を返します。
        """
        # Dotted key access, e.g., "menu.file"
        keys = key.split(".")
        value = self.translations
        for k in keys:
            value = value.get(k)
            if value is None:
                return key  # Return the key itself if not found
        return value.format(**kwargs)

    def _on_language_change(self, *args: Any) -> None:
        """
        状態管理の言語変数が変更された際に呼び出されるコールバック関数。

        Args:
            *args: Tkinterのtrace_addから渡される任意の引数。
        """
        self.load_language(self.app_state.language_var.get())
        # Here you would typically trigger a full UI refresh
        print(
            f"Language changed to {self.app_state.language_var.get()}. UI needs refresh."
        )
