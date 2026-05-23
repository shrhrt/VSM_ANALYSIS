# -*- coding: utf-8 -*-
import platform
import tkinter as tk
from tkinter import ttk

_OS = platform.system()

if _OS == "Windows":
    FONT_FAMILY = "Segoe UI"
elif _OS == "Darwin":
    FONT_FAMILY = "SF Pro Display"
else:
    FONT_FAMILY = "Ubuntu"

FONT_BODY    = (FONT_FAMILY, 10)
FONT_SMALL   = (FONT_FAMILY, 9)
FONT_HEADING = (FONT_FAMILY, 10, "bold")
FONT_HINT    = (FONT_FAMILY, 9, "italic")
FONT_LOG     = ("Consolas", 10)

ACCENT_COLOR  = "#3D5A99"   # Indigo
ACCENT_DARK   = "#2E4A80"   # Indigo (hover)
ACCENT_LIGHT  = "#EEF1F8"   # 薄いインディゴ（Treeview 偶数行など）
DANGER_COLOR  = "#DC2626"   # Red
DANGER_DARK   = "#B91C1C"   # Red (hover)
SECTION_PADDING = 10


def _theme_bg() -> str:
    """現在の sv_ttk テーマに応じた背景色を返す。"""
    import sv_ttk
    return "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa"


def apply_theme(style: ttk.Style) -> None:
    """sv_ttk 適用後に呼び出してフォント・スペーシングを上書きする。
    テーマ切り替え後にも再呼び出しすること。
    """
    style.configure(".", font=FONT_BODY)
    style.configure("TNotebook.Tab", padding=(14, 5))
    style.configure("Treeview", font=FONT_BODY, rowheight=26)
    style.configure("Treeview.Heading", font=FONT_HEADING, padding=(4, 6))
    style.configure("Status.TLabel", font=FONT_SMALL)

    # Danger.TButton ─ 削除・全消去などの破壊的操作
    style.configure("Danger.TButton",
        foreground=DANGER_COLOR,
        background="#FFF1F1",
    )
    style.map("Danger.TButton",
        foreground=[("active", "white"), ("disabled", "#AAAAAA"), ("!active", DANGER_COLOR)],
        background=[("active", DANGER_DARK),  ("disabled", "#F3F4F6"), ("!active", "#FFF1F1")],
    )

    # Accent.TButton の色をインディゴに上書き
    style.configure("Accent.TButton", background=ACCENT_COLOR)
    style.map("Accent.TButton",
        background=[("active", ACCENT_DARK), ("!active", ACCENT_COLOR)],
    )


def make_section(parent: tk.Widget, text: str, padding: int = SECTION_PADDING) -> tuple:
    """モダンなカードスタイルのセクションを生成する。
    ttk.LabelFrame の代替。ボーダーなし、タイトルは太字＋アクセントカラー。
    Returns (outer_frame, content_frame)。子ウィジェットは content_frame に追加する。

    NOTE: sv_ttk が ttk.Label の foreground をエレメントレベルで上書きするため、
    タイトルラベルには tk.Label を使い fg を直接指定する。
    """
    outer = ttk.Frame(parent)

    header = ttk.Frame(outer)
    header.pack(fill=tk.X, pady=(4, 0))

    # tk.Label で fg を直接指定（sv_ttk の ttk.Label foreground 上書き問題を回避）
    title_lbl = tk.Label(
        header,
        text=text,
        font=FONT_HEADING,
        fg=ACCENT_COLOR,
        bg=_theme_bg(),
    )
    title_lbl.pack(side=tk.LEFT, padx=(2, 0))

    ttk.Separator(header, orient="horizontal").pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), pady=5
    )

    content = ttk.Frame(outer, padding=padding)
    content.pack(fill=tk.BOTH, expand=True)

    return outer, content


def setup_treeview_tags(tree: ttk.Treeview) -> None:
    """Treeview に交互行カラー（zebra striping）のタグを設定する。"""
    tree.tag_configure("oddrow",  background="#ffffff")
    tree.tag_configure("evenrow", background=ACCENT_LIGHT)


def refresh_section_title_bg(root: tk.Misc) -> None:
    """テーマ切り替え後、make_section で作成した tk.Label の背景色を再設定する。"""
    bg = _theme_bg()
    _walk_and_update(root, bg)


def _walk_and_update(widget: tk.Misc, bg: str) -> None:
    if isinstance(widget, tk.Label):
        try:
            if widget.cget("fg") == ACCENT_COLOR:
                widget.configure(bg=bg)
        except tk.TclError:
            pass
    for child in widget.winfo_children():
        _walk_and_update(child, bg)
