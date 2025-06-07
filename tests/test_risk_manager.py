import pytest

from trading.risk_manager import calculate_position_size, calculate_sl_tp_levels


def test_calculate_position_size_percent_inputs():
    size = calculate_position_size(balance=1000, entry_price=100, stop_loss_pct=2, risk_pct=1)
    assert size == pytest.approx(5)

    size = calculate_position_size(balance=1000, entry_price=100, stop_loss_pct=3, risk_pct=1.5)
    assert size == pytest.approx(5)


def test_calculate_position_size_zero_or_negative_distance():
    assert calculate_position_size(1000, 100, 0, 1) == 0
    assert calculate_position_size(1000, 100, -2, 1) == 0


def test_calculate_position_size_zero_risk():
    assert calculate_position_size(1000, 100, 2, 0) == 0


def test_calculate_sl_tp_levels_percent_inputs():
    sl, tp = calculate_sl_tp_levels(100, 2)
    assert sl == pytest.approx(98)
    assert tp == pytest.approx(104)

    sl, tp = calculate_sl_tp_levels(100, 2, tp_ratio=3)
    assert sl == pytest.approx(98)
    assert tp == pytest.approx(106)


def test_calculate_sl_tp_levels_zero_or_negative_sl_pct():
    sl, tp = calculate_sl_tp_levels(100, 0)
    assert sl == pytest.approx(100)
    assert tp == pytest.approx(100)

    sl, tp = calculate_sl_tp_levels(100, -2)
    assert sl == pytest.approx(102)
    assert tp == pytest.approx(96)
