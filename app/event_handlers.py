# -*- coding: utf-8 -*-
import tkinter as tk
from pathlib import Path
import os
import json
from tkinter import TclError, filedialog, messagebox, scrolledtext, ttk, colorchooser
from typing import TYPE_CHECKING, Any

import analysis.file_io as file_io
import app.theme as theme
import tools.dat_to_VSM as dat_converter

if TYPE_CHECKING:
    from app.vsm_app import VSMApp


class EventHandlers:
    """
    GUI上のユーザーアクション（ボタンクリック、ファイル操作など）を処理するクラス。
    """

    def __init__(self, app: "VSMApp") -> None:
        """
        EventHandlersのインスタンスを初期化します。

        Args:
            app (VSMApp): メインアプリケーションのインスタンス。
        """
        self.app: "VSMApp" = app

    def convert_dat_file(self) -> None:
        """PPMSの.datファイルを.VSM形式に変換するダイアログを開きます。"""
        dat_path = filedialog.askopenfilename(
            title="変換する PPMS .dat ファイルを選択",
            filetypes=(("DAT files", "*.dat"), ("All files", "*.*")),
            parent=self.app.root,
        )
        if not dat_path:
            return

        # 拡張子を .vsm に変更したデフォルトファイル名を生成
        default_vsm_name = Path(dat_path).with_suffix(".vsm").name

        vsm_path = filedialog.asksaveasfilename(
            title="VSMファイルの保存先",
            initialfile=default_vsm_name,
            defaultextension=".vsm",
            filetypes=(("VSM files", "*.vsm"), ("All files", "*.*")),
            parent=self.app.root,
        )
        if not vsm_path:
            return

        try:
            dat_converter.convert_dat_to_vsm(dat_path, vsm_path)
            messagebox.showinfo(
                "成功",
                f"変換が完了しました。\n\n保存先: {vsm_path}",
                parent=self.app.root,
            )
        except Exception as e:
            messagebox.showerror(
                "エラー", f"変換中にエラーが発生しました:\n{e}", parent=self.app.root
            )

    def show_metadata_window(self) -> None:
        """読み込まれたファイルの測定情報（メタデータ）を表示するウィンドウを開きます。"""
        if not self.app.all_metadata:
            messagebox.showinfo(
                "情報", "表示できる測定情報がありません。", parent=self.app.root
            )
            return
        info_window = tk.Toplevel(self.app.root)
        info_window.title("測定情報")
        info_window.geometry("500x650")
        info_window.configure(bg=self.app.get_bg_color())
        top_frame = ttk.Frame(info_window, padding="10 10 10 0")
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="ファイルを選択:").pack(side=tk.LEFT, padx=(0, 10))
        file_names = list(self.app.all_metadata.keys())
        selected_file = tk.StringVar(value=file_names[0])
        file_menu = ttk.Combobox(
            top_frame, textvariable=selected_file, values=file_names, state="readonly"
        )
        file_menu.pack(fill=tk.X, expand=True)
        text_widget = scrolledtext.ScrolledText(
            info_window, wrap=tk.WORD, font=theme.FONT_BODY
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        def update_display(event=None):
            filename = selected_file.get()
            metadata = self.app.all_metadata.get(filename, {})
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            display_keys = {
                "date": "測定日",
                "sample name": "サンプル名",
                "comment": "コメント",
                "lock-in amp. sensitivity": "感度(mV)",
                "lock-in amp. time constant": "時定数(msec)",
                "measuring points": "測定点数",
                "max magnetic field": "最大磁場(Oe)",
                "max magnetization": "最大磁化(emu)",
                "lock-in amp. phase": "位相(deg)",
                "pole piece gap": "磁極間距離(mm)",
                "calibration value": "校正値",
            }
            info_text = f"ファイル: {filename}\n\n--- 測定パラメータ ---\n\n"
            for key, desc in display_keys.items():
                info_text += f"■ {desc}\n  {metadata.get(key, '---')}\n\n"
            text_widget.insert(tk.END, info_text)
            text_widget.config(state=tk.DISABLED)

        file_menu.bind("<<ComboboxSelected>>", update_display)
        update_display()

    def show_calculation_logic_window(self) -> None:
        """解析ロジック（計算方法）の解説ウィンドウを表示します。"""
        win = tk.Toplevel(self.app.root)
        win.title("計算ロジックの解説")
        win.geometry("800x600")
        win.configure(bg=self.app.get_bg_color())
        win.transient(self.app.root)

        # --- トピックと説明文を定義 ---
        topics = {
            "単位換算と体積磁化": """PPMSから出力される生データである磁場 H (Oe) と磁気モーメント M (emu) は、サンプルの寸法を用いて体積磁化に変換される。

■ 体積の計算
体積 (cm³) = 面積 (mm²) × 膜厚 (nm) × 10⁻⁹

■ 磁場のSI単位系への換算
H (T) = H (Oe) × 10⁻⁴

■ 磁化の換算
M (kA/m) = M (emu) / 体積 (cm³)
※ 1 emu/cm³ = 1 kA/m である。""",
            "反磁性補正": """強磁性体のヒステリシスループに乗っている、サンプルホルダーや基板に由来する線形な反磁性または常磁性成分を取り除くための補正である。

■ 自動モード
M-Hカーブの正負両端の領域（デフォルトでは全データ点の外側15%）に対し、最小二乗法による線形フィッティングを行い、その傾き χ を算出する。

■ 手動モード
ユーザーが指定した正負の磁場範囲内のデータ点に対し、同様に線形フィッティングを行い、傾き χ を算出する。

■ 補正計算
補正後の磁化 M_corrected は、元の磁化 M_raw から線形成分を差し引くことで得られる。
M_corrected = M_raw - χ × H""",
            "磁化オフセット補正": """測定機器のドリフト等に起因する、M軸方向の全体的なズレを補正し、ヒステリシスループが原点に対して対称になるように調整する。

■ オフセット値の計算
高磁場領域における磁化の平均値を計算する。
・M_pos: 最大磁場の90%以上の領域におけるMの平均値
・M_neg: 最小磁場の90%以下の領域におけるMの平均値
オフセット値 = (M_pos + M_neg) / 2

■ 補正計算
最終的な磁化 M_final は、反磁性補正後の磁化 M_corrected からオフセット値を差し引くことで得られる。
M_final = M_corrected - オフセット値""",
            "磁気特性の算出 (Ms, Mr, Hc)": """各種補正が完了した最終的な磁化 M_final を用いて、主要な磁気特性パラメータを算出する。

■ 飽和磁化 (Ms)
高磁場領域（デフォルトでは最大磁場の90%以上、手動指定も可）における正負両側の磁化の平均値を計算し、その絶対値の平均を Ms とする。

■ 残留磁化 (Mr)
ヒステリシスループの往路（Hが減少）と復路（Hが増加）のデータを分割する。
その後、`numpy.interp` を用いた線形補間により、磁場 H = 0 を横切る際の磁化 M の値をそれぞれ求め、それらの絶対値の平均を Mr とする。

■ 保磁力 (Hc)
同様に、線形補間を用いて磁化 M = 0 を横切る際の磁場 H の値をそれぞれ求め、それらの絶対値の平均を Hc とする。

■ 角形比 (S)
S = Mr / Ms""",
        }

        # --- メインレイアウト ---
        main_paned_window = ttk.PanedWindow(win, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- 左ペイン (トピックリスト) ---
        left_frame = ttk.Frame(main_paned_window, padding=0)
        main_paned_window.add(left_frame, weight=1)
        listbox = tk.Listbox(
            left_frame, font=(theme.FONT_FAMILY, 11), selectmode=tk.SINGLE, exportselection=False
        )
        for topic in topics:
            listbox.insert(tk.END, topic)
        listbox.pack(fill=tk.BOTH, expand=True)

        # --- 右ペイン (テキスト表示) ---
        right_frame = ttk.Frame(main_paned_window, padding=0)
        main_paned_window.add(right_frame, weight=3)
        text_widget = scrolledtext.ScrolledText(
            right_frame, wrap=tk.WORD, font=(theme.FONT_FAMILY, 11), padx=15, pady=15
        )
        text_widget.pack(fill=tk.BOTH, expand=True)

        # --- イベント処理 ---
        def update_text(event=None):
            selected_indices = listbox.curselection()
            if not selected_indices:
                return
            selected_topic = listbox.get(selected_indices[0])
            content = topics.get(selected_topic, "トピックが見つかりません。")
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)

        listbox.bind("<<ListboxSelect>>", update_text)

        # --- 初期状態 ---
        listbox.selection_set(0)
        update_text()

        theme.accent_button(win, text="閉じる", command=win.destroy).pack(pady=(0, 10))

    def on_drop_files(self, event: Any) -> None:
        """
        ドラッグ＆ドロップされたファイルを読み込み、既存のデータに追加します。

        Args:
            event (Any): ドロップイベントオブジェクト。
        """
        # Tkinterの機能を使って、スペースを含むファイルパスを安全にリスト化
        files = self.app.root.tk.splitlist(event.data)
        if not files:
            return
        self._process_loaded_files(files, append=True)

    def _process_loaded_files(self, files: tuple, append: bool = False) -> None:
        """
        ファイルパスのリストを処理し、アプリケーションのデータモデルに読み込みます。

        Args:
            files (tuple): 読み込むファイルのパスのタプル。
            append (bool): 既存のデータに追加するかどうか。Falseの場合は既存のデータをクリアします。
        """
        if not append:
            self.app.vsm_data.clear()
            self.app.file_color_vars.clear()
            start_idx = 0
        else:
            start_idx = len(self.app.vsm_data)

        loaded_names = []
        for i, file_path in enumerate(files):
            df, error_message = file_io.load_vsm_file(file_path)

            if error_message:
                messagebox.showerror("読込エラー", error_message, parent=self.app.root)
                self.app.log_message(f"読込失敗: {Path(file_path).name} — {error_message}", level="error")
                continue

            path = Path(file_path)
            file_data = {
                "path": path,
                "df": df,
                "thickness_var": tk.StringVar(value="30.0"),
                "marker_style_var": tk.StringVar(value="o"),
                "legend_name_var": tk.StringVar(value=path.stem),
            }
            self.app.vsm_data.append(file_data)
            loaded_names.append(path.name)

            # 色を順番に割り当て
            color_idx = (start_idx + i) % len(self.app.base_colors)
            color_var = tk.StringVar(value=self.app.base_colors[color_idx])
            self.app.file_color_vars.append(color_var)

        if loaded_names:
            mode = "追加" if append else "読込"
            names_str = "\n".join(f"  {n}" for n in loaded_names)
            self.app.log_message(f"ファイル{mode} ({len(loaded_names)}件):\n{names_str}")

        self.app.info_button.config(
            state=tk.NORMAL if self.app.vsm_data else tk.DISABLED
        )
        self.app._update_file_list_ui()
        self.app._update_demag_settings_ui()
        self.app._update_thickness_settings_ui()
        self.app.graph_manager.update_graph()

    def load_files(self) -> None:
        """ファイル選択ダイアログを開き、新しいVSMファイルを読み込みます（既存データはクリア）。"""
        files = filedialog.askopenfilenames(
            title="解析したいVSMファイルを選択",
            filetypes=[("VSM files", "*.VSM"), ("All files", "*.*")],
            parent=self.app.root,
        )
        if files:
            self._process_loaded_files(files, append=False)

    def add_files(self) -> None:
        """既存のデータを保持したまま、追加でVSMファイルを読み込みます。"""
        files = filedialog.askopenfilenames(
            title="追加するVSMファイルを選択",
            filetypes=[("VSM files", "*.VSM"), ("All files", "*.*")],
            parent=self.app.root,
        )
        if files:
            self._process_loaded_files(files, append=True)

    def clear_all_files(self) -> None:
        """全てのファイルをリストから削除し、UIを初期化。"""
        if not self.app.vsm_data:
            return

        # ユーザーに確認ダイアログを表示
        if not messagebox.askyesno(
            "確認", "全てのファイルを削除しますか？", parent=self.app.root
        ):
            return

        # アプリケーションが保持している各種データモデルをメモリ上からクリア
        self.app.vsm_data.clear()
        self.app.file_color_vars.clear()
        self.app.all_metadata.clear()
        self.app.analysis_results.clear()

        # UIコンポーネントを再構築・初期化
        self.app._update_file_list_ui()
        self.app._update_demag_settings_ui()
        self.app._update_thickness_settings_ui()
        self.app._update_results_table()
        self.app.info_button.config(state=tk.DISABLED)
        self.app.graph_manager.update_graph()

    def remove_file(self, index: int) -> None:
        """
        指定されたインデックスのファイルをリストから削除します。

        Args:
            index (int): 削除するファイルのリスト内インデックス。
        """
        if 0 <= index < len(self.app.vsm_data):
            filename = self.app.vsm_data[index]["path"].name
            if not messagebox.askyesno(
                "削除確認",
                f"'{filename}' をリストから削除しますか？",
                parent=self.app.root,
            ):
                return

            del self.app.vsm_data[index]
            del self.app.file_color_vars[index]
            self.app.log_message(f"ファイル削除: {filename}", level="error")

            self.app.info_button.config(
                state=tk.NORMAL if self.app.vsm_data else tk.DISABLED
            )
            self.app._update_file_list_ui()
            self.app._update_demag_settings_ui()
            self.app._update_thickness_settings_ui()
            self.app.graph_manager.update_graph()

    def save_figure(self) -> None:
        """
        現在のグラフを指定された画像フォーマットとサイズ設定で保存します。
        """
        try:
            w, h, dpi = (
                float(self.app.state.save_width_var.get()),
                float(self.app.state.save_height_var.get()),
                int(self.app.state.save_dpi_var.get()),
            )
            if w <= 0 or h <= 0 or dpi <= 0:
                raise ValueError(
                    "値は正数である必要があります。",
                )
        except ValueError as e:
            messagebox.showerror(
                "入力エラー",
                f"幅,高さ,DPIには有効な正数を入力してください。\n({e})",
                parent=self.app.root,
            )
            return
        file_path = filedialog.asksaveasfilename(
            title="画像を保存",
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("JPEG", "*.jpg"),
            ],
            defaultextension=".png",
            parent=self.app.root,
        )
        if not file_path:
            return
        original_size = self.app.fig.get_size_inches()
        try:
            self.app.log_message(
                f"画像を保存中: {Path(file_path).name}\n  サイズ: {w}x{h} inches, DPI: {dpi}\n"
            )
            self.app.fig.set_size_inches(w, h)
            self.app.fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
            self.app.log_message("保存が完了しました。\n", level="success")
            messagebox.showinfo(
                "成功", f"画像を保存しました:\n{file_path}", parent=self.app.root
            )
        except Exception as e:
            self.app.log_message(f"エラー: 画像保存失敗 - {e}\n", level="error")
            messagebox.showerror(
                "保存エラー", f"画像保存中にエラーが発生:\n{e}", parent=self.app.root
            )
        finally:
            self.app.fig.set_size_inches(original_size[0], original_size[1])
            self.app.canvas.draw_idle()

    def copy_graph_to_clipboard(self) -> None:
        """現在のグラフをDIB形式でクリップボードにコピーする（PowerPoint/Word等に直接貼り付け可能）。"""
        import io
        try:
            import win32clipboard
            import win32con
        except ImportError:
            messagebox.showerror(
                "エラー",
                "pywin32がインストールされていません。\npip install pywin32 を実行してください。",
                parent=self.app.root,
            )
            return

        try:
            dpi = float(self.app.state.save_dpi_var.get())
        except ValueError:
            dpi = 150.0

        try:
            buf = io.BytesIO()
            self.app.fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)

            from PIL import Image
            img = Image.open(buf).convert("RGB")

            bmp_buf = io.BytesIO()
            img.save(bmp_buf, format="BMP")
            dib_data = bmp_buf.getvalue()[14:]  # BMP ファイルヘッダ(14byte)を除いたDIB形式

            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32con.CF_DIB, dib_data)
            finally:
                win32clipboard.CloseClipboard()

            self.app.log_message("グラフをクリップボードにコピーしました。", level="success")
        except Exception as e:
            self.app.log_message(f"クリップボードコピー失敗: {e}", level="error")
            messagebox.showerror("エラー", f"コピーに失敗しました:\n{e}", parent=self.app.root)

    def show_advanced_style_window(self) -> None:
        """凡例名、マーカー、軸の数値フォーマットなどを設定する詳細スタイルウィンドウを表示します。"""
        if not self.app.vsm_data:
            messagebox.showinfo(
                "情報", "設定対象のファイルがありません。", parent=self.app.root
            )
            return

        win = tk.Toplevel(self.app.root)
        win.title("詳細スタイル設定")
        # ウィンドウのサイズを調整
        screen_height = win.winfo_screenheight()
        win_height = int(screen_height * 0.85)  # 画面の85%の高さ
        win.geometry(f"700x{win_height}")
        win.transient(self.app.root)
        win.grab_set()

        main_frame = ttk.Frame(win, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 凡例設定フレーム ---
        legend_section, legend_frame = theme.make_section(main_frame, "凡例名の設定")
        legend_section.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        canvas = tk.Canvas(
            legend_frame,
            borderwidth=0,
            background=self.app.get_bg_color(),
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(legend_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=(0, 0, 10, 0))

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for i, data in enumerate(self.app.vsm_data):
            row = ttk.Frame(scrollable_frame)
            row.pack(fill=tk.X, pady=3)
            row.grid_columnconfigure(1, weight=1)

            ttk.Label(row, text=f"ファイル{i + 1}:").grid(
                row=0, column=0, sticky="w", padx=(0, 10)
            )
            entry = ttk.Entry(row, textvariable=data["legend_name_var"], width=50)
            entry.grid(row=0, column=1, sticky="ew")
            if not any(t[0] == "write" for t in data["legend_name_var"].trace_info()):
                data["legend_name_var"].trace_add("write", self.app._schedule_update)

        # --- ヒントを追加 ---
        hint_label = ttk.Label(
            scrollable_frame,
            text="ヒント: 下付き文字やギリシャ文字(γ)は TeX 形式で入力できます (例: $H_2O$, $\gamma$)",
            font=theme.FONT_HINT,
            foreground="gray",
        )
        hint_label.pack(fill=tk.X, pady=(10, 0), padx=5)

        # --- 凡例ボックス設定フレーム ---
        legend_box_section, legend_box_frame = theme.make_section(main_frame, "凡例ボックス設定")
        legend_box_section.pack(fill=tk.X, expand=True, pady=(0, 10))
        legend_box_frame.grid_columnconfigure(1, weight=1)

        # 配置場所
        ttk.Label(legend_box_frame, text="配置場所:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        location_options = [
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center",
        ]
        ttk.Combobox(
            legend_box_frame,
            textvariable=self.app.state.legend_location_var,
            values=location_options,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        # 枠線
        ttk.Checkbutton(
            legend_box_frame,
            text="枠線を表示",
            variable=self.app.state.legend_show_frame_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        # 背景の透明度
        ttk.Label(legend_box_frame, text="背景の透明度:").grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Scale(
            legend_box_frame,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.app.state.legend_alpha_var,
            command=lambda s: self.app.state.legend_alpha_var.set(float(s)),
        ).grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        # 列数
        ttk.Label(legend_box_frame, text="列数:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        ttk.Entry(
            legend_box_frame, textvariable=self.app.state.legend_columns_var, width=10
        ).grid(row=3, column=1, sticky="ew", padx=5, pady=2)

        # --- マーカー設定フレーム ---
        marker_section, marker_frame = theme.make_section(main_frame, "マーカーの形状")
        marker_section.pack(fill=tk.X, expand=True, pady=(0, 10))

        marker_options = ["o", "s", "^", "v", "D", "p", "*", "x", "+"]
        num_columns = 4

        for i, data in enumerate(self.app.vsm_data):
            row_idx = i // num_columns
            col_idx = i % num_columns

            # Create a sub-frame for each item
            item_frame = ttk.Frame(marker_frame)
            item_frame.grid(row=row_idx, column=col_idx, padx=10, pady=5, sticky="ew")

            ttk.Label(item_frame, text=f"ファイル{i + 1}:").pack(
                side=tk.LEFT, padx=(0, 5)
            )
            combo = ttk.Combobox(
                item_frame,
                textvariable=data["marker_style_var"],
                values=marker_options,
                state="readonly",
                width=5,
            )
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if not any(t[0] == "write" for t in data["marker_style_var"].trace_info()):
                data["marker_style_var"].trace_add("write", self.app._schedule_update)

        # --- 軸フォーマット設定フレーム ---
        axis_format_section, axis_format_frame = theme.make_section(main_frame, "軸の数値フォーマット")
        axis_format_section.pack(fill=tk.X, expand=True, pady=(0, 10))
        axis_format_frame.grid_columnconfigure(1, weight=1)
        axis_format_frame.grid_columnconfigure(3, weight=1)

        # X軸フォーマット
        ttk.Label(axis_format_frame, text="X軸 (T):").grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Entry(
            axis_format_frame, textvariable=self.app.state.x_format_si_var, width=10
        ).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(axis_format_frame, text="X軸 (Oe):").grid(
            row=0, column=2, sticky="w", pady=2
        )
        ttk.Entry(
            axis_format_frame, textvariable=self.app.state.x_format_cgs_var, width=10
        ).grid(row=0, column=3, sticky="ew", padx=5, pady=2)

        # Y軸フォーマット
        ttk.Label(axis_format_frame, text="Y軸 (kA/m):").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Entry(
            axis_format_frame, textvariable=self.app.state.y_format_si_var, width=10
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(axis_format_frame, text="Y軸 (emu/cm³):").grid(
            row=1, column=2, sticky="w", pady=2
        )
        ttk.Entry(
            axis_format_frame, textvariable=self.app.state.y_format_cgs_var, width=10
        ).grid(row=1, column=3, sticky="ew", padx=5, pady=2)

        ttk.Label(axis_format_frame, text="Y軸 (M/Ms):").grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Entry(
            axis_format_frame, textvariable=self.app.state.y_format_norm_var, width=10
        ).grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        # フォーマット文字列のヒント
        ttk.Label(
            axis_format_frame,
            text="例: %.2f (小数点以下2桁), %.0f (整数), %.1e (指数表記)",
            font=theme.FONT_HINT,
            foreground="gray",
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(5, 0))

        # --- ボタンフレーム ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        def on_ok():
            self.app.graph_manager.update_graph()
            win.destroy()

        theme.accent_button(button_frame, text="OK", command=on_ok).pack()

    def show_ms_settings_window(self) -> None:
        """飽和磁化 (Ms) を計算するための磁場範囲を手動設定するウィンドウを表示します。"""
        settings_window = tk.Toplevel(self.app.root)
        settings_window.title("飽和磁化(Ms) 計算範囲設定")
        settings_window.geometry("650x500")
        settings_window.transient(self.app.root)
        settings_window.grab_set()

        main_frame = ttk.Frame(settings_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(
            main_frame,
            borderwidth=0,
            background=self.app.get_bg_color(),
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=(0, 0, 10, 0))

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        file_settings_vars = []

        def _toggle_manual_entries(widgets, manual_var):
            state = tk.NORMAL if manual_var.get() else tk.DISABLED
            for widget in widgets:
                widget.config(state=state)

        for i, data in enumerate(self.app.vsm_data):
            file_path = data["path"]
            current_settings = data.get("ms_calc_settings", {})

            manual_var = tk.BooleanVar(value=current_settings.get("manual", False))
            link_var = tk.BooleanVar(value=current_settings.get("link_ranges", True))
            pos_min_var = tk.StringVar(
                value=current_settings.get("pos_range", ("1.5", "2.0"))[0]
            )
            pos_max_var = tk.StringVar(
                value=current_settings.get("pos_range", ("1.5", "2.0"))[1]
            )
            neg_min_var = tk.StringVar(
                value=current_settings.get("neg_range", ("-2.0", "-1.5"))[0]
            )
            neg_max_var = tk.StringVar(
                value=current_settings.get("neg_range", ("-2.0", "-1.5"))[1]
            )

            file_settings_vars.append(
                {
                    "manual": manual_var,
                    "link": link_var,
                    "pos_min": pos_min_var,
                    "pos_max": pos_max_var,
                    "neg_min": neg_min_var,
                    "neg_max": neg_max_var,
                }
            )

            frame_section, frame = theme.make_section(scrollable_frame, file_path.name)
            frame_section.pack(fill=tk.X, expand=True, pady=5)
            frame.grid_columnconfigure(1, weight=1)
            frame.grid_columnconfigure(3, weight=1)

            manual_check = ttk.Checkbutton(
                frame, text="手動範囲で計算", variable=manual_var
            )
            manual_check.grid(row=0, column=0, columnspan=2, sticky="w")

            link_check = ttk.Checkbutton(
                frame, text="正負の範囲を連動", variable=link_var
            )
            link_check.grid(row=0, column=2, columnspan=2, sticky="w")

            pos_min_entry = ttk.Entry(frame, textvariable=pos_min_var, width=7)
            pos_min_entry.grid(row=1, column=1, sticky="ew", padx=(0, 2))
            ttk.Label(frame, text="～").grid(row=1, column=2)
            pos_max_entry = ttk.Entry(frame, textvariable=pos_max_var, width=7)
            pos_max_entry.grid(row=1, column=3, sticky="ew", padx=(2, 0))
            ttk.Label(frame, text=" (正 H)").grid(row=1, column=4, sticky="w")

            neg_min_entry = ttk.Entry(frame, textvariable=neg_min_var, width=7)
            neg_min_entry.grid(row=2, column=1, sticky="ew", pady=(5, 0), padx=(0, 2))
            ttk.Label(frame, text="～").grid(row=2, column=2, pady=(5, 0))
            neg_max_entry = ttk.Entry(frame, textvariable=neg_max_var, width=7)
            neg_max_entry.grid(row=2, column=3, sticky="ew", pady=(5, 0), padx=(2, 0))
            ttk.Label(frame, text=" (負 H)").grid(
                row=2, column=4, sticky="w", pady=(5, 0)
            )

            manual_entries = [
                pos_min_entry,
                pos_max_entry,
                neg_min_entry,
                neg_max_entry,
            ]

            def on_pos_change(
                *args,
                p_min_v=pos_min_var,
                p_max_v=pos_max_var,
                n_min_v=neg_min_var,
                n_max_v=neg_max_var,
                l_v=link_var,
            ):
                if l_v.get():
                    try:
                        p_min = float(p_min_v.get())
                        n_max_v.set(str(-p_min))
                    except (ValueError, TclError):
                        pass
                    try:
                        p_max = float(p_max_v.get())
                        n_min_v.set(str(-p_max))
                    except (ValueError, TclError):
                        pass

            pos_min_var.trace_add("write", on_pos_change)
            pos_max_var.trace_add("write", on_pos_change)

            manual_var.trace_add(
                "write",
                lambda *a, w=manual_entries, v=manual_var: _toggle_manual_entries(w, v),
            )
            _toggle_manual_entries(manual_entries, manual_var)

        button_frame = ttk.Frame(settings_window, padding=(10, 0, 10, 10))
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        def save_settings():
            try:
                for i, data in enumerate(self.app.vsm_data):
                    vars_dict = file_settings_vars[i]
                    settings = {
                        "manual": vars_dict["manual"].get(),
                        "link_ranges": vars_dict["link"].get(),
                        "pos_range": (
                            vars_dict["pos_min"].get(),
                            vars_dict["pos_max"].get(),
                        ),
                        "neg_range": (
                            vars_dict["neg_min"].get(),
                            vars_dict["neg_max"].get(),
                        ),
                    }
                    data["ms_calc_settings"] = settings
                settings_window.destroy()
                self.app.graph_manager.update_graph()
                messagebox.showinfo(
                    "成功", "飽和磁化の計算設定を保存しました。", parent=settings_window
                )
            except ValueError:
                messagebox.showerror(
                    "入力エラー",
                    "磁場範囲には有効な数値を入力してください。",
                    parent=settings_window,
                )

        theme.neutral_button(
            button_frame, text="キャンセル", command=settings_window.destroy,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        theme.accent_button(
            button_frame, text="OK & 保存", command=save_settings,
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def move_file_up(self, index: int) -> None:
        """
        ファイルリスト内で指定されたインデックスのファイルを1つ上に移動します。

        Args:
            index (int): 移動させるファイルのインデックス。
        """
        if index > 0:
            # Swap data
            self.app.vsm_data.insert(index - 1, self.app.vsm_data.pop(index))
            self.app.file_color_vars.insert(
                index - 1, self.app.file_color_vars.pop(index)
            )

            # Rebuild all dynamic UIs and update graph
            self.app._update_file_list_ui()
            self.app._update_demag_settings_ui()
            self.app._update_thickness_settings_ui()
            self.app.graph_manager.update_graph()

    def move_file_down(self, index: int) -> None:
        """
        ファイルリスト内で指定されたインデックスのファイルを1つ下に移動します。

        Args:
            index (int): 移動させるファイルのインデックス。
        """
        if index < len(self.app.vsm_data) - 1:
            # Swap data
            self.app.vsm_data.insert(index + 1, self.app.vsm_data.pop(index))
            self.app.file_color_vars.insert(
                index + 1, self.app.file_color_vars.pop(index)
            )

            # Rebuild all dynamic UIs and update graph
            self.app._update_file_list_ui()
            self.app._update_demag_settings_ui()
            self.app._update_thickness_settings_ui()
            self.app.graph_manager.update_graph()

    def choose_individual_color(self, index: int) -> None:
        """
        カラーピッカーダイアログを開き、指定されたファイルのプロット色を選択します。

        Args:
            index (int): 色を変更するファイルのインデックス。
        """
        if index >= len(self.app.file_color_vars):
            return
        color_var = self.app.file_color_vars[index]
        path_name = self.app.vsm_data[index]["path"].name
        title_name = (path_name[:40] + "..") if len(path_name) > 42 else path_name
        color_code = colorchooser.askcolor(
            title=f"'{title_name}' の色を選択", initialcolor=color_var.get()
        )
        if color_code and color_code[1]:
            color_var.set(color_code[1])

    def apply_first_file_settings_to_all(self) -> None:
        """リストの一番上にあるファイルの反磁性補正設定を、他の全てのファイルに適用します。"""
        if len(self.app.vsm_data) < 2:
            return

        # 1. Get settings from the first file's data model
        first_file_settings = self.app.vsm_data[0]["demag_settings"].copy()

        # 2. Apply settings to all other files
        for i in range(1, len(self.app.vsm_data)):
            data = self.app.vsm_data[i]

            # Update the data model
            data["demag_settings"] = first_file_settings.copy()

            # Update the UI variables, which will trigger traces
            vars_dict = data.get("demag_vars")
            if vars_dict:
                vars_dict["enabled"].set(first_file_settings["enabled"])
                vars_dict["manual"].set(first_file_settings["manual"])
                vars_dict["link"].set(first_file_settings["link_ranges"])
                vars_dict["pos_min"].set(first_file_settings["pos_range"][0])
                vars_dict["pos_max"].set(first_file_settings["pos_range"][1])
                vars_dict["neg_min"].set(first_file_settings["neg_range"][0])
                vars_dict["neg_max"].set(first_file_settings["neg_range"][1])

        messagebox.showinfo(
            "成功",
            "一番上のファイルの設定をすべてのファイルに適用しました。",
            parent=self.app.root,
        )

    def save_session(self) -> None:
        """現在のセッション（グローバル設定、ファイルリスト、個別設定）をJSONファイルに保存します。"""
        if not self.app.vsm_data:
            messagebox.showwarning(
                "保存エラー", "保存するデータがありません。", parent=self.app.root
            )
            return

        filepath = filedialog.asksaveasfilename(
            title="セッションを保存",
            filetypes=[("VSM Session Files", "*.vsm_session"), ("All files", "*.*")],
            defaultextension=".vsm_session",
            parent=self.app.root,
        )
        if not filepath:
            return

        try:
            session_file_path = Path(filepath)
            session_dir = session_file_path.parent

            session_data = {
                "global_settings": self.app.state.to_dict(),
                "file_specific_data": [],
            }

            for i, data in enumerate(self.app.vsm_data):
                data_path = data["path"].resolve()
                try:
                    rel_path = os.path.relpath(str(data_path), str(session_dir))
                except ValueError:
                    rel_path = str(data_path)

                # OneDrive相対パスを計算（別PCでの復元用）
                onedrive_rel = None
                for env_key in ("OneDriveCommercial", "OneDrive"):
                    od_root = os.environ.get(env_key, "")
                    if od_root:
                        try:
                            candidate = os.path.relpath(str(data_path), od_root)
                            if not candidate.startswith(".."):
                                onedrive_rel = candidate
                                break
                        except ValueError:
                            continue

                file_data = {
                    "path": str(data_path),  # 後方互換性用
                    "relative_path": rel_path,
                    "onedrive_relative_path": onedrive_rel,
                    "thickness": data["thickness_var"].get(),
                    "area": data["area_var"].get(),
                    "marker_style": data["marker_style_var"].get(),
                    "legend_name": data["legend_name_var"].get(),
                    "color": self.app.file_color_vars[i].get(),
                    "demag_settings": data.get("demag_settings", {}),
                    "ms_calc_settings": data.get("ms_calc_settings", {}),
                }
                session_data["file_specific_data"].append(file_data)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=4, ensure_ascii=False)

            self.app.log_message(f"セッション保存: {session_file_path.name}", level="success")
            messagebox.showinfo(
                "成功", "セッションを保存しました。", parent=self.app.root
            )

        except Exception as e:
            messagebox.showerror(
                "保存エラー",
                f"セッションの保存中にエラーが発生しました:\n{e}",
                parent=self.app.root,
            )

    def load_session(self) -> None:
        """保存されたJSONファイルからセッション（設定とファイルリスト）を読み込み、状態を復元します。"""
        filepath = filedialog.askopenfilename(
            title="セッションを読み込み",
            filetypes=[("VSM Session Files", "*.vsm_session"), ("All files", "*.*")],
            parent=self.app.root,
        )
        if not filepath:
            return

        try:
            session_file_path = Path(filepath)
            session_dir = session_file_path.parent

            with open(filepath, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # グローバル設定を復元
            self.app.state.from_dict(session_data.get("global_settings", {}))

            # ファイルリストをクリアして再構築
            self.app.vsm_data = []
            self.app.file_color_vars = []

            missing_files = []
            for file_data in session_data.get("file_specific_data", []):
                rel_path_str = file_data.get("relative_path")
                abs_path_str = file_data.get("path")
                od_rel_str = file_data.get("onedrive_relative_path")
                path = None

                # 1. 相対パスで探す（セッションファイルと同じPCで使う場合）
                if rel_path_str:
                    target_path = (session_dir / rel_path_str).resolve()
                    if target_path.exists():
                        path = target_path
                # 2. 絶対パスで探す（同じPCでの読み込みや古いセッションファイル用）
                if not path and abs_path_str:
                    target_path = Path(abs_path_str)
                    if target_path.exists():
                        path = target_path
                # 3. 同じフォルダ内を探す（ファイルをまとめて移動した場合用）
                if not path and abs_path_str:
                    target_path = session_dir / Path(abs_path_str).name
                    if target_path.exists():
                        path = target_path
                # 4. OneDriveルート＋相対パスで探す（別PCで同期している場合用）
                if not path and od_rel_str:
                    for env_key in ("OneDriveCommercial", "OneDrive"):
                        od_root = os.environ.get(env_key, "")
                        if od_root:
                            candidate = Path(od_root) / od_rel_str
                            if candidate.exists():
                                path = candidate
                                break

                if not path:
                    missing_files.append(abs_path_str or rel_path_str or "(不明)")
                    continue

                df, _ = file_io.load_vsm_file(path)
                new_data = {"path": path, "df": df}
                new_data["thickness_var"] = tk.StringVar(
                    value=file_data.get("thickness", "30.0")
                )
                new_data["area_var"] = tk.StringVar(
                    value=file_data.get("area", "90")
                )
                new_data["marker_style_var"] = tk.StringVar(
                    value=file_data.get("marker_style", "o")
                )
                new_data["legend_name_var"] = tk.StringVar(
                    value=file_data.get("legend_name", path.stem)
                )
                new_data["demag_settings"] = file_data.get("demag_settings", {})
                new_data["ms_calc_settings"] = file_data.get("ms_calc_settings", {})
                self.app.vsm_data.append(new_data)
                self.app.file_color_vars.append(
                    tk.StringVar(value=file_data.get("color", "#000000"))
                )

            if missing_files:
                file_list = "\n".join(f"  • {p}" for p in missing_files)
                messagebox.showwarning(
                    "ファイル欠落",
                    f"以下のファイルが見つからなかったためスキップしました:\n\n{file_list}",
                    parent=self.app.root,
                )

            # UIを更新してグラフを再描画
            self.app._update_file_list_ui()
            self.app._update_demag_settings_ui()
            self.app._update_thickness_settings_ui()
            n = len(self.app.vsm_data)
            self.app.log_message(
                f"セッション読込: {session_file_path.name}  ({n}件のファイル)", level="success"
            )
            self.app.graph_manager.update_graph()

        except Exception as e:
            messagebox.showerror(
                "読込エラー",
                f"セッションの読み込み中にエラーが発生しました:\n{e}",
                parent=self.app.root,
            )
