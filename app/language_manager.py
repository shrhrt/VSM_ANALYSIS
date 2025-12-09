# -*- coding: utf-8 -*-
import json
from pathlib import Path
import tkinter as tk


class LanguageManager:
    def __init__(self, app_state):
        self.app_state = app_state
        self.locales_dir = Path(__file__).parent.parent / "locales"
        self.translations = {}
        self.load_language(self.app_state.language_var.get())

        # Listen for language changes
        self.app_state.language_var.trace_add("write", self._on_language_change)

    def load_language(self, lang_code):
        """Load the specified language file (e.g., 'en', 'ja')."""
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

    def get(self, key, **kwargs):
        """Get a translated string by its key."""
        # Dotted key access, e.g., "menu.file"
        keys = key.split(".")
        value = self.translations
        for k in keys:
            value = value.get(k)
            if value is None:
                return key  # Return the key itself if not found
        return value.format(**kwargs)

    def _on_language_change(self, *args):
        self.load_language(self.app_state.language_var.get())
        # Here you would typically trigger a full UI refresh
        print(
            f"Language changed to {self.app_state.language_var.get()}. UI needs refresh."
        )
