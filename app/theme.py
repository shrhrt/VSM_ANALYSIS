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

ACCENT_COLOR  = "#00704A"   # Starbucks Emerald Green
ACCENT_DARK   = "#005A3C"   # Dark Green (hover)
ACCENT_LIGHT  = "#E8F5EF"   # Pale Green（Treeview 偶数行など）
DANGER_COLOR  = "#DC2626"   # Red
DANGER_DARK   = "#B91C1C"   # Red (hover)
SECTION_PADDING = 10

# ログ用テキスト色（白背景向け）
LOG_FG_TS      = "#999999"   # タイムスタンプ（グレー）
LOG_FG_INFO    = "#222222"   # 通常テキスト
LOG_FG_SUCCESS = "#1a7f37"   # 成功（深い緑）
LOG_FG_ERROR   = "#cf222e"   # エラー（深い赤）


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

    # Accent.TButton の色を上書き（sv_ttk 非対応だが一応設定）
    style.configure("Accent.TButton", background=ACCENT_COLOR)
    style.map("Accent.TButton",
        background=[("active", ACCENT_DARK), ("!active", ACCENT_COLOR)],
    )


def make_section(parent: tk.Widget, text: str, padding: int = SECTION_PADDING) -> tuple:
    """モダンなカードスタイルのセクションを生成する。
    左端にアクセントカラーの縦ストライプ、タイトルは太字＋アクセントカラー。
    Returns (outer_frame, content_frame)。子ウィジェットは content_frame に追加する。

    NOTE: sv_ttk が ttk.Label の foreground をエレメントレベルで上書きするため、
    タイトルラベルには tk.Label を使い fg を直接指定する。
    """
    outer = ttk.Frame(parent)

    # 左端の緑ストライプ（モダンカードUIのアクセント）
    stripe = tk.Frame(outer, bg=ACCENT_COLOR, width=4)
    stripe.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
    stripe.pack_propagate(False)

    inner = ttk.Frame(outer)
    inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    header = ttk.Frame(inner)
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

    content = ttk.Frame(inner, padding=padding)
    content.pack(fill=tk.BOTH, expand=True)

    return outer, content


def accent_button(parent: tk.Widget, **kwargs) -> tk.Button:
    """緑のプライマリボタン。ttk では sv_ttk が色を上書きするため tk.Button を使用。
    ホバー時に暗い緑に変化するエフェクト付き。
    """
    kw = dict(
        bg=ACCENT_COLOR, fg="white",
        activebackground=ACCENT_DARK, activeforeground="white",
        relief="flat", font=FONT_BODY, cursor="hand2",
        padx=12, pady=6, borderwidth=0,
    )
    kw.update(kwargs)
    btn = tk.Button(parent, **kw)
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT_DARK)   if btn["state"] != "disabled" else None)
    btn.bind("<Leave>", lambda e: btn.config(bg=ACCENT_COLOR)  if btn["state"] != "disabled" else None)
    return btn


def danger_button(parent: tk.Widget, **kwargs) -> tk.Button:
    """赤の危険操作ボタン。ホバー時に暗い赤に変化するエフェクト付き。"""
    kw = dict(
        bg=DANGER_COLOR, fg="white",
        activebackground=DANGER_DARK, activeforeground="white",
        relief="flat", font=FONT_BODY, cursor="hand2",
        padx=12, pady=6, borderwidth=0,
    )
    kw.update(kwargs)
    btn = tk.Button(parent, **kw)
    btn.bind("<Enter>", lambda e: btn.config(bg=DANGER_DARK)   if btn["state"] != "disabled" else None)
    btn.bind("<Leave>", lambda e: btn.config(bg=DANGER_COLOR)  if btn["state"] != "disabled" else None)
    return btn


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
