# -*- coding: utf-8 -*-
import tkinter as tk


class StateManager:
    """
    アプリケーション全体の状態（設定値など）を管理するクラス。
    tk.StringVarなどのUIにバインドされる変数を一元管理する。
    """

    def __init__(self):
        # --- スタイル変数 ---
        self.area_var = tk.StringVar(value="1.0")
        self.offset_correction_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)
        self.unit_mode_var = tk.StringVar(value="SI (T, kA/m)")
        self.marker_size_var = tk.StringVar(value="5")
        self.line_width_var = tk.StringVar(value="1.5")
        self.axis_label_fontsize_var = tk.StringVar(value="16")
        self.tick_label_fontsize_var = tk.StringVar(value="12")
        self.legend_fontsize_var = tk.StringVar(value="12")
        self.xlim_min_var = tk.StringVar(value="")
        self.xlim_max_var = tk.StringVar(value="")
        self.ylim_min_var = tk.StringVar(value="")
        self.ylim_max_var = tk.StringVar(value="")
        self.show_grid_var = tk.BooleanVar(value=True)
        self.show_zero_lines_var = tk.BooleanVar(value=True)
        self.zero_line_color_var = tk.StringVar(value="grey")
        self.zero_line_linestyle_var = tk.StringVar(value="-")
        self.save_width_var = tk.StringVar(value="6.0")
        self.save_height_var = tk.StringVar(value="6.0")
        self.save_dpi_var = tk.StringVar(value="300")

        # --- 詳細スタイル変数 ---
        self.x_step_var = tk.StringVar(value="")
        self.y_step_var = tk.StringVar(value="")
        self.x_format_var = tk.StringVar(value="%.1f")
        self.y_format_var = tk.StringVar(value="%.1f")
        self.grid_style_var = tk.StringVar(value=":")
        self.grid_color_var = tk.StringVar(value="#CCCCCC")

    def to_dict(self):
        """現在の状態を辞書に変換して返す"""
        state_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(
                value, (tk.StringVar, tk.BooleanVar, tk.IntVar, tk.DoubleVar)
            ):
                state_dict[key] = value.get()
        return state_dict

    def from_dict(self, state_dict):
        """辞書から状態を復元する"""
        for key, value in state_dict.items():
            if hasattr(self, key):
                var = getattr(self, key)
                if isinstance(
                    var, (tk.StringVar, tk.BooleanVar, tk.IntVar, tk.DoubleVar)
                ):
                    try:
                        var.set(value)
                    except tk.TclError as e:
                        print(f"Warning: Could not set state for '{key}': {e}")
