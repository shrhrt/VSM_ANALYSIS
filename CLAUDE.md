# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VSM Data Analyzer** は、磁性材料の磁気特性測定データ（VSM, PPMS）を解析・可視化するデスクトップアプリ。
**Tauri v2 (Rust) + React/TypeScript フロントエンド** と **Python (FastAPI) バックエンド** で構成される。

> 注: 旧版は Tkinter 製の単体 GUI (`app/`) だったが、現在は Tauri + FastAPI 構成へ完全移行済み。
> 旧 Tkinter コードは削除済み（必要なら git 履歴で参照できる）。

## Commands

```bash
# 開発起動 (バックエンド + Tauri dev ウィンドウをまとめて起動し、閉じると後片付けまで行う)
python main.py

# 配布用ビルド (backend.exe サイドカー同梱の .msi/.exe インストーラを生成)
python build.py

# Python テスト実行
pytest -v

# 単一テスト実行
pytest test_calculations.py::test_calculate_coercivity -v
```

依存のインストール:
```bash
pip install -r requirements.txt          # Python (backend + テスト)
cd vsm-tauri && npm install               # フロントエンド (Tauri + React)
```

Rust ツールチェーン (rustup/cargo) と Node.js が必要。

## Architecture

### 全体構成

```
main.py              # 【開発起動】uvicorn バックエンド + `npm run tauri dev` を起動し、
                     #   ウィンドウを閉じたらプロセスツリーごと後片付け (ゾンビ防止)
build.py             # 【本番ビルド】backend_server.py を PyInstaller で backend.exe 化 →
                     #   vsm-tauri/src-tauri/binaries/ にサイドカー配置 → `tauri build`
backend_server.py    # 本番バックエンドのエントリ (PyInstaller で固める。削除禁止)

backend/             # FastAPI サーバ (http://localhost:8000)
  main.py            #   app 定義・CORS・Chrome PNA ミドルウェア・GET /health
  routers/
    analysis.py      #   POST /api/analyze — ファイルを受け取り解析結果 JSON を返す
    session.py       #   GET /api/session/env, POST /api/session/resolve — パス解決

analysis/            # 純粋ロジック層 (バックエンドとテストが共用する)
  calculations.py    #   Ms/Mr/Hc/Hs 計算・反磁性補正・オフセット補正 (test_calculations.py が網羅)
  file_io.py         #   .VSM ファイル読み込み・メタデータ解析

vsm-tauri/           # Tauri v2 + React/TS フロントエンド
  src/
    App.tsx          #   ルート。ファイルリスト・解析結果・グラフ設定の中央 state を保持
    api/client.ts    #   バックエンド呼び出し (fetch to :8000) と型定義
    components/      #   Graph / Sidebar / StatusBar / HelpDialog / ResultsTable ...
    data/trivia.ts   #   ステータスバーの豆知識 (ガチャは現在無効化。StatusBar.tsx の GACHA_ENABLED)
    utils/           #   texToDisplay 等 (TeX → Unicode 変換など)
  src-tauri/
    src/lib.rs       #   Rust: backend.exe サイドカーの起動と、アプリ終了時の kill を管理

tools/dat_to_VSM.py  # PPMS (.dat) → VSM 形式 変換ツール
```

### データフロー

React (Tauri WebView) → `api/client.ts` が HTTP fetch → FastAPI (:8000) →
`analysis/calculations.py` で計算 → JSON を返す → React が SVG でグラフ描画。

グラフ描画は **フロントエンド (SVG) が担当**。バックエンドは matplotlib を使わない
(`build.py` でも `--exclude-module matplotlib` で明示的に除外している)。

### 中心データ構造: 解析結果 (`AnalysisResult`)

`POST /api/analyze` は 1 ファイルを受け取り、次の JSON を返す
(型定義: [client.ts](vsm-tauri/src/api/client.ts) の `AnalysisResult`):

| キー | 内容 |
|---|---|
| `Ms` / `Mr` | 飽和磁化 / 残留磁化 (kA/m) |
| `Hc_T` / `Hc_Oe` | 保磁力 (T / Oe) |
| `Hs_Oe` | 飽和磁場 (Oe) |
| `squareness` | 角形比 (Mr/Ms) |
| `demag_slope` | 反磁性補正の傾き |
| `plot` | M-H ループ描画用配列 `{H_down, M_down, H_up, M_up}` (往路/復路) |
| `logs` | 解析中の計算ログ (文字列配列) |
| `metadata` | ファイルのメタデータ |

解析パラメータ (膜厚 thickness[nm]・面積 area[mm²]・反磁性補正モード・Ms 計算範囲など) は
`multipart/form-data` で送る。ファイル別設定 (`per_demag_mode`, `ms_manual` 等) が指定されれば
グローバル設定より優先される。

### 単位変換

入力の生データは `H(Oe)`, `M(emu)`。バックエンドは内部で SI 系へ変換して計算する
(`analysis.py`: `H×1e-4 → T`、`M / 体積 → kA/m`。体積は膜厚[nm]×面積[mm²]から算出)。
フロントエンドは `unit_mode` に応じて表示単位 (SI/CGS/規格化) を切り替えて描画する。

### 重要な設計パターン

**バックエンドプロセスのライフサイクル**:
- 開発時 ([main.py](main.py)): `.venv\Scripts\python.exe` はシム (本体 python を孫として起動する中継役)
  のため、単純な `terminate()` では孫プロセスが残る。Windows では `taskkill /F /T` でツリーごと終了。
  起動前に `_free_port()` でポート 8000 の残骸も掃除する。
- 本番時 ([lib.rs](vsm-tauri/src-tauri/src/lib.rs)): `Child` を Tauri の `.manage()` 状態に保持し、
  `RunEvent::ExitRequested` で `kill()` する (Rust の `Child` は drop では自動 kill されないため)。

**数式表記**: 軸ラベル・凡例は TeX ライクな入力 (`$M$ (emu/cm$^3$)` など) を
`utils` の `texToDisplay()` で Unicode に変換して描画。ヘルプ内の数式は KaTeX で表示。

**セッション保存**: ファイルパスと各ファイルの設定を `.vsm_session` に保存。相対パス／OneDrive パスで
復元するため別 PC でも再現可能。パス解決は `POST /api/session/resolve` がバックエンドで行う。

### テスト対象

- `test_calculations.py` — `analysis/calculations.py` の純粋関数を網羅 (Ms/Mr/Hc/Hs, 反磁性補正)。
  **バックエンドの解析ロジックはこの層を再利用するため、実質的に本番の計算経路をカバーしている。**
- `test_tex_utils.py` — TeX 変換ロジック。

FastAPI ルーター層 (HTTP 契約) と React フロントエンドには自動テストなし。
