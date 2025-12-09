# -*- coding: utf-8 -*-
import tkinter as tk
from pathlib import Path
from tkinter import TclError, filedialog, messagebox, scrolledtext, ttk, colorchooser

import analysis.file_io as file_io


class EventHandlers:
    def __init__(self, app):
        self.app = app

    def show_metadata_window(self):
        if not self.app.all_metadata:
            messagebox.showinfo(
                "情報", "表示できる測定情報がありません。", parent=self.app.root
            )
            return
        info_window = tk.Toplevel(self.app.root)
        info_window.title("測定情報")
        info_window.geometry("500x650")
        info_window.configure(bg=self.app.style.lookup(".", "background"))
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
            info_window, wrap=tk.WORD, font=("Arial", 10), bg="white", fg="black"
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

    def load_files(self):
        files = filedialog.askopenfilenames(
            title="解析したいVSMファイルを選択",
            filetypes=[("VSM files", "*.VSM"), ("All files", "*.*")],
            parent=self.app.root,
        )
        if not files:
            return

        self.app.vsm_data, self.app.file_color_vars = [], []
        for i, file_path in enumerate(files):
            df, error_message = file_io.load_vsm_file(file_path)

            if error_message:
                messagebox.showerror("読込エラー", error_message, parent=self.app.root)
                if "必要な列" in error_message:
                    messagebox.showwarning(
                        "形式エラー", error_message, parent=self.app.root
                    )
                else:
                    messagebox.showerror(
                        "読込エラー", error_message, parent=self.app.root
                    )
                continue

            path = Path(file_path)
            file_data = {
                "path": path,
                "df": df,
                "thickness_var": tk.StringVar(value="100.0"),
            }
            self.app.vsm_data.append(file_data)

            color_var = tk.StringVar(
                value=self.app.base_colors[i % len(self.app.base_colors)]
            )
            self.app.file_color_vars.append(color_var)

        self.app.info_button.config(
            state=tk.NORMAL if self.app.vsm_data else tk.DISABLED
        )
        self.app._update_file_list_ui()
        self.app._update_demag_settings_ui()
        self.app._update_thickness_settings_ui()
        self.app.graph_manager.update_graph()

    def save_figure(self):
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
            self.app.log_message("保存が完了しました。\n")
            messagebox.showinfo(
                "成功", f"画像を保存しました:\n{file_path}", parent=self.app.root
            )
        except Exception as e:
            self.app.log_message(f"エラー: 画像保存失敗 - {e}\n")
            messagebox.showerror(
                "保存エラー", f"画像保存中にエラーが発生:\n{e}", parent=self.app.root
            )
        finally:
            self.app.fig.set_size_inches(original_size)
            self.app.canvas.draw_idle()

    def show_ms_settings_window(self):
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
            background=self.app.style.lookup(".", "background"),
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

            frame = ttk.LabelFrame(scrollable_frame, text=file_path.name, padding=10)
            frame.pack(fill=tk.X, expand=True, pady=5)
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

        ttk.Button(
            button_frame, text="キャンセル", command=settings_window.destroy
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(button_frame, text="OK & 保存", command=save_settings).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )

    def move_file_up(self, index):
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

    def move_file_down(self, index):
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

    def choose_individual_color(self, index):
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

    def apply_first_file_settings_to_all(self):
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
