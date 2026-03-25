# -*- coding: utf-8 -*-
import tkinter as tk
from typing import Dict, Any


class StateManager:
    """
    アプリケーション全体の状態（設定値など）を管理するクラス。
    tk.StringVarなどのUIにバインドされる変数を一元管理する。
    """

    def __init__(self) -> None:
        """状態管理変数の初期化を行います。"""
        # --- スタイル変数 ---
        self.area_var: tk.StringVar = tk.StringVar(value="1.0")
        self.offset_correction_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.show_legend_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.unit_mode_var: tk.StringVar = tk.StringVar(value="SI (T, kA/m)")
        self.marker_size_var: tk.StringVar = tk.StringVar(
            value="0"
        )  # マーカーサイズ初期値を0に設定
        self.line_width_var: tk.StringVar = tk.StringVar(value="1.5")
        self.axis_label_fontsize_var: tk.StringVar = tk.StringVar(value="24")
        self.tick_label_fontsize_var: tk.StringVar = tk.StringVar(value="16")
        self.legend_fontsize_var: tk.StringVar = tk.StringVar(value="16")
        self.legend_location_var: tk.StringVar = tk.StringVar(value="best")
        self.legend_show_frame_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.legend_alpha_var: tk.DoubleVar = tk.DoubleVar(value=1.0)
        self.legend_columns_var: tk.StringVar = tk.StringVar(value="1")

        self.xlim_min_var: tk.StringVar = tk.StringVar(value="-1.0")
        self.xlim_max_var: tk.StringVar = tk.StringVar(value="1.0")
        self.ylim_min_var: tk.StringVar = tk.StringVar(value="")
        self.ylim_max_var: tk.StringVar = tk.StringVar(value="")
        self.show_grid_var: tk.BooleanVar = tk.BooleanVar(value=False)
        self.show_zero_lines_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.zero_line_color_var: tk.StringVar = tk.StringVar(value="grey")
        self.zero_line_linestyle_var: tk.StringVar = tk.StringVar(value=":")
        self.save_width_var: tk.StringVar = tk.StringVar(value="6.0")
        self.save_height_var: tk.StringVar = tk.StringVar(value="6.0")
        self.save_dpi_var: tk.StringVar = tk.StringVar(value="300")

        # --- 詳細スタイル変数 ---
        self.x_format_si_var: tk.StringVar = tk.StringVar(value="%.1f")  # T (Tesla)
        self.x_format_cgs_var: tk.StringVar = tk.StringVar(value="%.0f")  # Oe (Oersted)
        self.y_format_si_var: tk.StringVar = tk.StringVar(value="%.0f")  # kA/m
        self.y_format_cgs_var: tk.StringVar = tk.StringVar(value="%.1f")  # emu/cm^3
        self.y_format_norm_var: tk.StringVar = tk.StringVar(value="%.2f")  # M/Ms
        self.grid_style_var: tk.StringVar = tk.StringVar(value=":")
        self.grid_color_var: tk.StringVar = tk.StringVar(value="#CCCCCC")

    def to_dict(self) -> Dict[str, Any]:
        """
        現在の状態を辞書に変換して返します。

        Returns:
            Dict[str, Any]: 全ての状態変数の名前と値のペア
        """
        state_dict: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(
                value, (tk.StringVar, tk.BooleanVar, tk.IntVar, tk.DoubleVar)
            ):
                state_dict[key] = value.get()
        return state_dict

    def from_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        辞書データから状態を復元します。

        Args:
            state_dict (Dict[str, Any]): 復元する状態変数の名前と値のペアが含まれた辞書
        """
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
