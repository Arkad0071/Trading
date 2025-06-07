# trading/risk_manager.py

def calculate_position_size(balance, entry_price, stop_loss_pct, risk_pct):
    """
    balance: текущий баланс в USDT
    entry_price: цена входа
    stop_loss_pct: допустимый SL в процентах от entry_price
    risk_pct: процент риска от баланса (например 1 для 1%)
    """
    dollar_risk = balance * (risk_pct / 100)
    distance_to_sl = entry_price * (stop_loss_pct / 100)
    if distance_to_sl <= 0:
        return 0
    return dollar_risk / distance_to_sl


def calculate_sl_tp_levels(entry_price, stop_loss_pct, tp_ratio=2.0):
    """
    Возвращает кортеж (stop_price, take_price)
    """
    stop_price = entry_price * (1 - stop_loss_pct / 100)
    take_price = entry_price * (1 + stop_loss_pct * tp_ratio / 100)
    return stop_price, take_price
