# trading/risk_manager.py

def calculate_position_size(balance, entry_price, stop_loss_pct, risk_pct):
    """
    balance: текущий баланс в USDT
    entry_price: цена входа
    stop_loss_pct: допустимый SL в процентах от entry_price
    risk_pct: процент риска от баланса (например 1 для 1%)
    """
    # Input validation
    if balance <= 0:
        raise ValueError("Balance must be positive")
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")
    if stop_loss_pct <= 0:
        raise ValueError("Stop loss percentage must be positive")
    if risk_pct <= 0 or risk_pct > 100:
        raise ValueError("Risk percentage must be between 0 and 100")
    
    dollar_risk = balance * (risk_pct / 100)
    distance_to_sl = entry_price * (stop_loss_pct / 100)
    
    if distance_to_sl <= 0:
        raise ValueError("Invalid stop loss configuration: distance to stop loss must be positive")
    
    position_size = dollar_risk / distance_to_sl
    
    # Additional safety check
    if position_size <= 0:
        raise ValueError("Calculated position size is invalid")
    
    return position_size


def calculate_sl_tp_levels(entry_price, stop_loss_pct, tp_ratio=2.0):
    """
    Возвращает кортеж (stop_price, take_price)
    """
    # Input validation
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")
    if stop_loss_pct <= 0:
        raise ValueError("Stop loss percentage must be positive")
    if tp_ratio <= 0:
        raise ValueError("Take profit ratio must be positive")
    
    stop_price = entry_price * (1 - stop_loss_pct / 100)
    take_price = entry_price * (1 + stop_loss_pct * tp_ratio / 100)
    
    # Ensure stop price is positive
    if stop_price <= 0:
        raise ValueError("Calculated stop price is invalid (must be positive)")
    
    return stop_price, take_price
