# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk  # tk.StringVarのため
import io  # StringIOのため
from contextlib import redirect_stdout  # redirect_stdoutのため
import sys  # tracebackのため

import analysis.file_io as file_io
import analysis.calculations as vsm_logic


def format_axis(ax, fig, style_params, unit_mode="SI (T, kA/m)"):
    """グラフの軸や目盛りなどを整形"""
    # 単位モードに応じて軸ラベルを設定
    if "CGS" in unit_mode:
        ax.set_xlabel(
            r"$H$ (Oe)",
            fontsize=style_params.get("axis_label_fontsize", 16),
        )
        ax.set_ylabel(
            r"$M$ (emu/cm$^3$)",
            fontsize=style_params.get("axis_label_fontsize", 16),
        )
    elif "Normalized" in unit_mode:
        ax.set_xlabel(
            r"$\mu_0H$ (T)",
            fontsize=style_params.get("axis_label_fontsize", 16),
        )
        ax.set_ylabel(
            r"$M/M_s$",  # This one has no units, so it's fine as is
            fontsize=style_params.get("axis_label_fontsize", 16),
        )
    else:  # SI (T, kA/m)
        ax.set_xlabel(
            r"$\mu_0H$ (T)",
            fontsize=style_params.get("axis_label_fontsize", 16),
        )
        ax.set_ylabel(
            r"$M$ (kA/m)",
            fontsize=style_params.get("axis_label_fontsize", 16),
        )

    if style_params.get("show_zero_lines", True):
        ax.axhline(0, color="#AAAAAA", linestyle="-", linewidth=1.0)
        ax.axvline(0, color="#AAAAAA", linestyle="-", linewidth=1.0)
    if style_params.get("show_grid", True):
        ax.grid(
            True,
            linestyle=style_params.get("grid_style", ":"),
            color=style_params.get("grid_color", "#CCCCCC"),
        )
    if (
        style_params.get("xlim_min") is not None
        and style_params.get("xlim_max") is not None
    ):
        ax.set_xlim(style_params["xlim_min"], style_params["xlim_max"])
    if (
        style_params.get("ylim_min") is not None
        and style_params.get("ylim_max") is not None
    ):
        ax.set_ylim(style_params["ylim_min"], style_params["ylim_max"])

    # --- 新しい軸設定 ---
    if style_params.get("xaxis_step"):
        try:
            step = float(style_params["xaxis_step"])
            if step > 0:
                ax.xaxis.set_major_locator(MultipleLocator(step))
        except (ValueError, TypeError):
            pass  # 無効な値は無視
    if style_params.get("yaxis_step"):
        try:
            step = float(style_params["yaxis_step"])
            if step > 0:
                ax.yaxis.set_major_locator(MultipleLocator(step))
        except (ValueError, TypeError):
            pass

    if style_params.get("xaxis_format"):
        try:
            ax.xaxis.set_major_formatter(
                FormatStrFormatter(style_params["xaxis_format"])
            )
        except (ValueError, TypeError):
            print(f"警告: 無効なX軸フォーマットです: {style_params['xaxis_format']}")
    if style_params.get("yaxis_format"):
        try:
            ax.yaxis.set_major_formatter(
                FormatStrFormatter(style_params["yaxis_format"])
            )
        except (ValueError, TypeError):
            print(f"警告: 無効なY軸フォーマットです: {style_params['yaxis_format']}")

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=style_params.get("tick_label_fontsize", 12),
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.0,
        colors="black",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8,
        colors="black",
    )
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]


class GraphManager:
    def __init__(self, app):
        self.app = app

    def update_graph(self):
        self.app.log_text.delete(1.0, tk.END)
        self.app.ax.clear()
        self.app.all_metadata = {}  # app.all_metadata はVsmAppで管理すべきだが、一時的にここに

        unit_mode = self.app.state.unit_mode_var.get()

        try:
            params = {
                "Area": float(self.app.state.area_var.get()),
                "marker_size": float(self.app.state.marker_size_var.get()),
                "line_width": float(self.app.state.line_width_var.get()),
                "axis_label_fontsize": int(
                    self.app.state.axis_label_fontsize_var.get()
                ),
                "tick_label_fontsize": int(
                    self.app.state.tick_label_fontsize_var.get()
                ),
                "legend_fontsize": int(self.app.state.legend_fontsize_var.get()),
                "show_grid": self.app.state.show_grid_var.get(),
                "show_zero_lines": self.app.state.show_zero_lines_var.get(),
                "xlim_min": float(v)
                if (v := self.app.state.xlim_min_var.get())
                else None,
                "xlim_max": float(v)
                if (v := self.app.state.xlim_max_var.get())
                else None,
                "ylim_min": float(v)
                if (v := self.app.state.ylim_min_var.get())
                else None,
                "ylim_max": float(v)
                if (v := self.app.state.ylim_max_var.get())
                else None,
            }
        except ValueError:
            return

        if not self.app.vsm_data:
            format_axis(self.app.ax, self.app.fig, params, unit_mode)
            self.app.ax.text(
                0.5,
                0.5,
                "ファイルを選択してください",
                ha="center",
                va="center",
                color="gray",
                fontsize=16,
                transform=self.app.ax.transAxes,
            )
            self.app.canvas.draw()
            self.app._update_demag_settings_ui()  # VSMAppのメソッド
            self.app._update_thickness_settings_ui()  # VSMAppのメソッド
            return

        output_stream = io.StringIO()
        with redirect_stdout(output_stream):
            print(f"解析開始: 面積={params['Area']} cm²\n")
            self._process_and_plot(params, unit_mode)

        self.app.log_message(output_stream.getvalue())
        self.app.log_message("\n描画完了。\n")
        self.app.canvas.draw()
        self.app.info_button.config(
            state=tk.NORMAL if self.app.all_metadata else tk.DISABLED
        )

    def _process_and_plot(self, params, unit_mode):
        self.app.analysis_results = []
        h_min_global, h_max_global = float("inf"), float("-inf")
        print("読み込みファイル:")
        [print(f" {i + 1}: {d['path'].name}") for i, d in enumerate(self.app.vsm_data)]

        for idx, data in enumerate(self.app.vsm_data):
            file, df = data["path"], data["df"]
            try:
                thick_nm = float(data["thickness_var"].get())
                Vol = params["Area"] * thick_nm * 1e-7

                self.app.all_metadata[file.name] = file_io.parse_metadata(file)
                min_H_idx = df["H(Oe)"].idxmin()
                if df["H(Oe)"].iloc[min_H_idx:].empty:
                    raise ValueError("不完全なデータ。復路が見つかりません。")
                max_H_idx2 = min_H_idx + df["H(Oe)"].iloc[min_H_idx:].idxmax()
                df_loop = df.iloc[: max_H_idx2 + 1]
                H_raw, M_raw = df_loop["H(Oe)"] * 1e-4, df_loop["M(emu)"] / Vol
                print(
                    f"\n--- 解析: {file.stem} (膜厚: {thick_nm} nm, データ点: {len(H_raw)}) ---"
                )

                # --- 反磁性補正 ---
                slope, r2_pos, r2_neg = 0, 0, 0
                file_specific_settings = data.get("demag_settings")

                if file_specific_settings and file_specific_settings["enabled"]:
                    print("  反磁性補正: 有効")
                    if file_specific_settings["manual"]:
                        print("    傾き計算: 手動設定モード")
                        try:
                            pos_range = (
                                float(file_specific_settings["pos_range"][0]),
                                float(file_specific_settings["pos_range"][1]),
                            )
                            neg_range = (
                                float(file_specific_settings["neg_range"][0]),
                                float(file_specific_settings["neg_range"][1]),
                            )
                            slope, r2_pos, r2_neg = vsm_logic.find_demag_slope_manual(
                                H_raw, M_raw, pos_range, neg_range
                            )
                        except (ValueError, IndexError):
                            print("  エラー: 手動設定の磁場範囲が無効。")
                            slope, r2_pos, r2_neg = 0, 0, 0
                    else:
                        print("    傾き計算: 自動検出モード")
                        slope, r2_pos, r2_neg = vsm_logic.find_demag_slope_auto(
                            H_raw, M_raw
                        )
                    print(
                        f"    補正傾き S: {slope:.6f}, R^2: [正 {r2_pos:.4f}], [負 {r2_neg:.4f}]"
                    )
                else:
                    print("  反磁性補正: 無効")

                M_corrected = M_raw - H_raw * slope
                if self.app.state.offset_correction_var.get():
                    print("  磁化オフセット補正: 有効")
                    H_np, M_np = H_raw.values, M_corrected.values
                    Ms_pos = (
                        np.mean(M_np[H_np > np.max(H_np) * 0.9])
                        if np.any(H_np > np.max(H_np) * 0.9)
                        else 0
                    )
                    Ms_neg = (
                        np.mean(M_np[H_np < np.min(H_np) * 0.9])
                        if np.any(H_np < np.min(H_np) * 0.9)
                        else 0
                    )
                    offset = (Ms_pos + Ms_neg) / 2
                    M_final = M_corrected - offset
                    print(f"    補正値: {offset:.4f} kA/m")
                else:
                    print("  磁化オフセット補正: 無効")
                    M_final = M_corrected

                min_H_idx_loop = H_raw.idxmin()
                H_down, M_down = (
                    H_raw.iloc[: min_H_idx_loop + 1].values,
                    M_final.iloc[: min_H_idx_loop + 1].values,
                )
                H_up, M_up = (
                    H_raw.iloc[min_H_idx_loop:].values,
                    M_final.iloc[min_H_idx_loop:].values,
                )

                # Create a dictionary to store results for the current file.
                file_results = {
                    "filename": file.stem,
                    "Ms": None,
                    "Mr": None,
                    "Hc_Oe": None,
                    "squareness": None,
                }

                # --- 飽和磁化 (Ms) 計算 ---
                ms_settings = data.get("ms_calc_settings")
                ms_pos_range, ms_neg_range = None, None

                try:
                    if ms_settings and ms_settings.get("manual"):
                        # Use manual ranges if they are set
                        print("    Ms計算: 手動範囲を使用")
                        ms_pos_range = (
                            float(ms_settings["pos_range"][0]),
                            float(ms_settings["pos_range"][1]),
                        )
                        ms_neg_range = (
                            float(ms_settings["neg_range"][0]),
                            float(ms_settings["neg_range"][1]),
                        )
                    elif (
                        params.get("xlim_max") is not None
                        and params.get("xlim_min") is not None
                    ):
                        # Otherwise, use plot limits if they are set
                        print("    Ms計算: 描画範囲を使用")
                        h_max_limit = params["xlim_max"]
                        h_min_limit = params["xlim_min"]
                        # Use the outer 10% of the *visible* range if the range is positive
                        if h_max_limit > 0 and h_min_limit < h_max_limit:
                            pos_start = h_max_limit - (h_max_limit - h_min_limit) * 0.1
                            neg_end = h_min_limit + (h_max_limit - h_min_limit) * 0.1
                            ms_pos_range = (pos_start, h_max_limit)
                            ms_neg_range = (h_min_limit, neg_end)

                except (ValueError, IndexError, TypeError):
                    print(
                        "  エラー: Ms計算の範囲が無効です。自動計算にフォールバックします。",
                    )
                    ms_pos_range, ms_neg_range = None, None

                Ms_results = vsm_logic.calculate_saturation_magnetization(
                    H_raw, M_final, pos_range=ms_pos_range, neg_range=ms_neg_range
                )
                if Ms_results:
                    file_results["Ms"] = Ms_results.get("avg")

                Mr_avg = vsm_logic.calculate_remanence(H_down, M_down, H_up, M_up)
                file_results["Mr"] = Mr_avg

                Hc_results = vsm_logic.calculate_coercivity(H_down, M_down, H_up, M_up)
                if Hc_results:
                    file_results["Hc_Oe"] = Hc_results.get("Oe")

                if (
                    Mr_avg is not None
                    and file_results["Ms"] is not None
                    and file_results["Ms"] > 0
                ):
                    squareness = Mr_avg / file_results["Ms"]
                    file_results["squareness"] = squareness

                self.app.analysis_results.append(file_results)

                # --- グラフ描画単位の処理 ---
                Ms_avg = file_results["Ms"]  # Keep Ms_avg for normalization logic
                if "CGS" in unit_mode:
                    H_plot_down, H_plot_up = H_down * 10000, H_up * 10000
                    M_plot_down, M_plot_up = M_down, M_up
                elif "Normalized" in unit_mode:
                    H_plot_down, H_plot_up = H_down, H_up
                    if Ms_avg is not None and Ms_avg > 0:
                        M_plot_down, M_plot_up = M_down / Ms_avg, M_up / Ms_avg
                    else:
                        M_plot_down, M_plot_up = M_down, M_up
                else:  # SI
                    H_plot_down, H_plot_up = H_down, H_up
                    M_plot_down, M_plot_up = M_down, M_up

                color = self.app.file_color_vars[idx].get()
                plot_kwargs = {
                    "marker": "o",
                    "markersize": params["marker_size"],
                    "linestyle": "-",
                    "linewidth": params["line_width"],
                }
                self.app.ax.plot(H_plot_down, M_plot_down, color=color, **plot_kwargs)
                self.app.ax.plot(
                    H_plot_up, M_plot_up, color=color, label=file.stem, **plot_kwargs
                )

                if "CGS" in unit_mode:
                    h_min_global = min(h_min_global, H_plot_down.min())
                    h_max_global = max(h_max_global, H_plot_up.max())
                else:
                    h_min_global = min(h_min_global, H_down.min())
                    h_max_global = max(h_max_global, H_up.max())

            except Exception as e:
                print(f"\nエラー: '{file.name}'の処理中に問題発生: {e}")
                import traceback

                traceback.print_exc(file=sys.stdout)
                continue

        if params["xlim_min"] is None and params["xlim_max"] is None:
            params["xlim_min"], params["xlim_max"] = h_min_global, h_max_global

        format_axis(self.app.ax, self.app.fig, params, unit_mode)
        if self.app.state.show_legend_var.get() and any(
            self.app.ax.get_legend_handles_labels()[1]
        ):
            self.app.ax.legend(
                fontsize=params["legend_fontsize"],
                loc="best",
                facecolor="white",
                edgecolor="#DDDDDD",
                labelcolor="black",
            )
        self.app.fig.tight_layout()
        self.app._update_results_table()
        self.app.main_notebook.tab(self.app.results_tab, text="解析結果 *")
