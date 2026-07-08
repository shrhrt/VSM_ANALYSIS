# -*- coding: utf-8 -*-
import pytest
import numpy as np

# テスト対象の関数をインポート
from analysis.calculations import (
    calculate_saturation_magnetization,
    calculate_coercivity,
    calculate_remanence,
    find_demag_slope_auto,
    find_demag_slope_manual,
)


def test_calculate_saturation_magnetization_auto():
    """飽和磁化(Ms)の自動計算機能が正しく動作するかテストする"""

    # 1. 準備 (Arrange): ダミーデータの作成
    # 磁場Hが -10 から 10 まで変化すると仮定します。
    H = np.array([-10.0, -9.0, -5.0, 0.0, 5.0, 9.0, 10.0])

    # 磁化Mは、最大磁場(10)で100、最小磁場(-10)で-100に飽和しているダミーデータを作ります。
    M = np.array([-100.0, -95.0, -50.0, 0.0, 50.0, 95.0, 100.0])

    # 2. 実行 (Act): テスト対象の関数を呼び出す
    # 自動範囲計算では、H_max(10)*0.9=9.0より大きいデータ(H=10, M=100)の平均と
    # H_min(-10)*0.9=-9.0より小さいデータ(H=-10, M=-100 -> 絶対値で100)の平均を取るはずです。
    result = calculate_saturation_magnetization(H, M)

    # 3. 検証 (Assert): 結果が期待通りか確認する
    assert result["pos"] == 100.0  # 正側のMsは100であるべき
    assert result["neg"] == 100.0  # 負側のMs(絶対値)は100であるべき
    assert result["avg"] == 100.0  # 全体の平均値は100であるべき


def test_calculate_coercivity():
    """保磁力(Hc)の計算が正しく動作するかテストする"""

    # 1. 準備 (Arrange)
    # 往路(H_down)と復路(H_up)のダミーデータ。
    # M=0を横切る（保磁力）場所を、わかりやすく H=2.0 と H=-2.0 に設定します。
    H_down = np.array([5.0, 2.0, 0.0, -5.0])
    M_down = np.array([100.0, 0.0, -50.0, -100.0])  # H=2.0 のとき M=0

    H_up = np.array([-5.0, -2.0, 0.0, 5.0])
    M_up = np.array([-100.0, 0.0, 50.0, 100.0])  # H=-2.0 のとき M=0

    # 2. 実行 (Act)
    result = calculate_coercivity(H_down, M_down, H_up, M_up)

    # 3. 検証 (Assert)
    # Hcは往路(2.0)と復路(絶対値で2.0)の平均なので 2.0 T になるはずです。
    assert result is not None
    assert result["T"] == 2.0
    # 1 T = 10000 Oe なので、Oe単位では 20000 になるはずです。
    assert result["Oe"] == 20000.0


def test_symmetric_loop_zero_exchange_bias():
    """対称なループでは交換バイアス(Heb)が 0 になることをテストする"""
    H_down = np.array([5.0, 2.0, 0.0, -5.0])
    M_down = np.array([100.0, 0.0, -50.0, -100.0])
    H_up = np.array([-5.0, -2.0, 0.0, 5.0])
    M_up = np.array([-100.0, 0.0, 50.0, 100.0])

    result = calculate_coercivity(H_down, M_down, H_up, M_up)

    assert result is not None
    # 対称なので Heb = (2 + -2)/2 = 0
    assert result["Heb_T"] == pytest.approx(0.0)


def test_calculate_exchange_bias_shifted_loop():
    """水平にシフトしたループで交換バイアス(Heb)が正しく算出されるかテストする"""

    # 往路は M=0 を H=3 で、復路は M=0 を H=-1 で横切る（+1 方向にシフトしたループ）
    H_down = np.array([5.0, 3.0, 0.0, -5.0])
    M_down = np.array([100.0, 0.0, -30.0, -100.0])  # H=3.0 のとき M=0
    H_up = np.array([-5.0, -1.0, 0.0, 5.0])
    M_up = np.array([-100.0, 0.0, 40.0, 100.0])     # H=-1.0 のとき M=0

    result = calculate_coercivity(H_down, M_down, H_up, M_up)

    assert result is not None
    # Hc = (|3| + |-1|)/2 = 2.0（保磁力はシフトの影響を受けない）
    assert result["T"] == pytest.approx(2.0)
    # 符号付き交差点
    assert result["down_T"] == pytest.approx(3.0)
    assert result["up_T"] == pytest.approx(-1.0)
    # Heb = (3 + -1)/2 = 1.0（ループ中心のシフト量）
    assert result["Heb_T"] == pytest.approx(1.0)
    assert result["Heb_Oe"] == pytest.approx(10000.0)


def test_calculate_remanence():
    """残留磁化(Mr)の計算が正しく動作するかテストする"""

    # 1. 準備 (Arrange)
    # H=0 を横切る（残留磁化）場所を、M=50 と M=-50 に設定します。
    H_down = np.array([2.0, 0.0, -2.0])
    M_down = np.array([100.0, 50.0, 0.0])  # H=0 のとき M=50

    H_up = np.array([-2.0, 0.0, 2.0])
    M_up = np.array([-100.0, -50.0, 0.0])  # H=0 のとき M=-50

    # 2. 実行 (Act)
    result = calculate_remanence(H_down, M_down, H_up, M_up)

    # 3. 検証 (Assert)
    # 往路(50)と復路(絶対値で50)の平均なので 50.0 になるはずです。
    assert result == 50.0


def test_find_demag_slope_auto():
    """反磁性補正の傾き自動検出が正しく動作するかテストする"""

    # 1. 準備 (Arrange)
    # 傾きが -2.0 の完全な直線データを作ります。
    # H_data が増えると M_data が減る（反磁性の特徴）
    H_data = np.linspace(-10, 10, 100)
    # y = -2x + 0 (傾き -2.0)
    M_data = -2.0 * H_data

    # 2. 実行 (Act)
    slope, r2_pos, r2_neg = find_demag_slope_auto(H_data, M_data)

    # 3. 検証 (Assert)
    # 期待される傾きは -2.0 に近いはずです（浮動小数点の誤差を考慮して丸めるか、pytest.approxを使います）
    assert pytest.approx(slope, rel=1e-5) == -2.0
    assert pytest.approx(r2_pos, rel=1e-5) == 1.0  # 完全な直線なのでR^2は1.0
    assert pytest.approx(r2_neg, rel=1e-5) == 1.0


def test_find_demag_slope_manual():
    """手動指定範囲での反磁性補正傾き計算が正しく動作するかテストする"""

    # 1. 準備 (Arrange)
    # 傾きが -3.0 のデータを作ります。
    H_data = np.array([-10.0, -8.0, 0.0, 8.0, 10.0])
    M_data = -3.0 * H_data

    # 2. 実行 (Act)
    # 正側を 5.0〜15.0、負側を -15.0〜-5.0 の範囲で指定します。
    slope, r2_pos, r2_neg = find_demag_slope_manual(
        H_data, M_data, pos_range=(5.0, 15.0), neg_range=(-15.0, -5.0)
    )

    # 3. 検証 (Assert)
    assert pytest.approx(slope, rel=1e-5) == -3.0
