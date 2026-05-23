# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VSM Data Analyzer** は、磁性材料の磁気特性測定データ（VSM, PPMS）を解析・可視化するPython製デスクトップGUIアプリケーション。

## Commands

```bash
# アプリ起動
python main.py

# テスト実行
pytest -v

# 単一テスト実行
pytest test_calculations.py::test_calculate_coercivity -v

# 実行ファイルのビルド (PyInstaller)
pyinstaller main.spec
```

依存ライブラリのインストール:
```bash
pip install -r requirements.txt
```

## Architecture

### レイヤー構造

```
main.py                     # エントリーポイント (TkinterDnD.Tk 初期化)
└── app/vsm_app.py          # VSMApp: メインGUIクラス・タブ管理・中央データ保持
    ├── app/state_manager.py    # StateManager: 全 tk.Var を一元管理
    ├── app/graph_manager.py    # GraphManager: Matplotlib グラフ描画ロジック
    ├── app/event_handlers.py   # EventHandlers: ユーザー操作イベントのハンドラ群
    ├── app/analysis_tab.py     # AnalysisTab: 解析タブのUI構築
    └── analysis/
        ├── calculations.py     # 純粋な数学関数 (Ms/Mr/Hc 計算、反磁性補正)
        └── file_io.py          # .VSM ファイル読み込み・メタデータ解析
tools/dat_to_VSM.py             # PPMS (.dat) → VSM形式 変換ツール
```

### 中心データ構造

`VSMApp.vsm_data` は読み込み済みファイルの辞書リスト。各要素の主要キー:

| キー | 内容 |
|---|---|
| `path` | `pathlib.Path` ファイルパス |
| `df` | 生の測定データ (`pd.DataFrame`, 列: `H(Oe)`, `M(emu)`) |
| `demag_settings` | 反磁性補正の設定値 (有効/手動/範囲) |
| `demag_vars` | 補正設定に対応する `tk.Var` オブジェクト群 |
| `thickness_var` | 膜厚 (nm) の `tk.StringVar` |
| `area_var` | 面積 (mm²) の `tk.StringVar` |

### 重要な設計パターン

**デバウンス更新**: UI変数の変更 → `_schedule_update()` → `root.after(500ms, update_graph)` でグラフ再描画。連続入力時に過剰な再描画を防ぐ。

**グラフ再描画トリガー**: `StateManager` の各 `tk.Var` に `trace_add("write", _schedule_update)` を設定 (`_add_traces()`)。カラーや補正設定の変数も同様のトレースを個別設定。

**単位変換**: 内部計算はすべてSI単位系 (T, kA/m)。生データ (`H(Oe)`, `M(emu)`) は `GraphManager` 内で `state.unit_mode_var` に応じてSI/CGS/規格化へ変換して描画。

**セッション保存**: ファイルパスリストと各ファイルの設定 (`demag_settings`, `thickness`, `color`等) を `.vsm_session` ファイルに保存。相対パスで復元するため別PCでも再現可能。

### テスト対象

`test_calculations.py` が `analysis/calculations.py` の純粋関数を網羅。GUIコード (`app/`) にはテストなし。
