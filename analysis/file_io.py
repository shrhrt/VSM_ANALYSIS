# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

def find_header_row(file_path, default_row=40):
    """
    ファイル内のデータヘッダー行を自動的に検出する。
    'H(Oe)'と'M(emu)'を含む行を探し、その行番号（0-indexed）を返す。
    """
    encodings_to_try = ["shift-jis", "utf-8"]
    for encoding in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for i, line in enumerate(f):
                    if "H(Oe)" in line and "M(emu)" in line:
                        print(f"  情報: ヘッダーを {i + 1} 行目で検出。")
                        return i
                    if i > 100:
                        break
        except (UnicodeDecodeError, IOError):
            continue

    print(
        f"  警告: ヘッダー行を自動検出できず。デフォルト値({default_row + 1}行目)を使用。"
    )
    return default_row


def parse_metadata(file_path):
    """
    ファイルのヘッダーから測定メタデータを抽出し、辞書として返す。
    """
    metadata = {}
    try:
        encodings_to_try = ["shift-jis", "utf-8"]
        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    for i, line in enumerate(f):
                        if i > 40:
                            break
                        try:
                            line = line.strip()
                            if "=" in line:
                                parts = line.split("=", 1)
                                key = parts[0].strip()
                                value_part = parts[1]
                                if value_part.startswith(","):
                                    value_parts = value_part.split(",")
                                    if len(value_parts) > 1:
                                        value = value_parts[1].strip()
                                        if key and value:
                                            metadata[key] = value
                        except IndexError:
                            continue
                if metadata:
                    return metadata
            except (UnicodeDecodeError, IOError):
                continue
    except Exception as e:
        print(f"  警告: メタデータ読み取り中に予期せぬエラー発生: {e}。")
    return metadata

def load_vsm_file(file_path):
    """
    一つのVSMファイルを読み込み、DataFrameを返す。
    エラーが発生した場合は、(None, error_message)を返す。
    """
    path = Path(file_path)
    header_row = find_header_row(path)
    try:
        try:
            df = pd.read_csv(path, header=header_row, encoding="shift-jis")
        except UnicodeDecodeError:
            df = pd.read_csv(path, header=header_row, encoding="utf-8")
        
        df.dropna(inplace=True)
        
        if not {"H(Oe)", "M(emu)"}.issubset(df.columns):
            return None, f"ファイル '{path.name}' に必要な列 ('H(Oe)', 'M(emu)') がありません。"
            
        return df, None # Success
    except Exception as e:
        return None, f"'{path.name}'の読み込みに失敗しました:\n{e}"