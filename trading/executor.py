# trading/executor.py

import ccxt
import logging
from utils.config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    MARGIN_MODE,
    LEVERAGE,
    COMMISSION_RATE
)
from positions_db import (
    add_open_position,
    get_open_positions,
    remove_open_position,
    log_trade,
    load_bot_state,
    save_bot_state
)

logger = logging.getLogger(__name__)


def init_trading_client(symbol: str = "BTC/USDT"):
    """
    Приватный клиент Bybit для торговли:
    - переключает маржинальный режим (isolated/cross)
    - устанавливает плечо
    """
    exchange = ccxt.bybit({
        'apiKey': BYBIT_API_KEY,
        'secret': BYBIT_API_SECRET,
        'enableRateLimit': True,
    })
    # Переключаем режим маржи
    if exchange.has.get('setMarginMode'):
        exchange.setMarginMode(MARGIN_MODE, symbol)
    else:
        exchange.private_post_position_switch_isolated({
            'symbol': symbol,
            'is_isolated': (MARGIN_MODE == 'isolated')
        })
    # Устанавливаем плечо
    if exchange.has.get('setLeverage'):
        exchange.setLeverage(LEVERAGE, symbol)
    else:
        exchange.private_post_position_leverage_save({
            'symbol': symbol,
            'buy_leverage': LEVERAGE,
            'sell_leverage': LEVERAGE
        })
    return exchange


def place_order(symbol: str, side: str, amount: float, price: float = None):
    """
    Отправляет ордер на Bybit.
    side: 'Buy' или 'Sell'; amount: базовый актив; price=None => рыночный.
    """
    client = init_trading_client(symbol)
    order_type = 'Market' if price is None else 'Limit'
    params = {
        'symbol': symbol,
        'side': side.upper(),
        'orderType': order_type,
        'qty': amount,
    }
    if price is not None:
        params['price'] = price
        params['timeInForce'] = 'GoodTillCancel'
    logger.info(f"Placing order: {params}")
    resp = client.private_linear_post_order_create(**params)
    logger.info(f"Order response: {resp}")
    return resp


def execute_entry(symbol: str,
                  entry_price: float,
                  position_size: float,
                  stop_loss: float,
                  take_profit: float):
    """
    Открывает позицию: ордер Buy, добавление в open_positions, обновление bot_state.
    """
    # 1) Отправляем ордер
    resp = place_order(symbol, side='Buy', amount=position_size)

    # 2) Рассчитываем затраченные средства и комиссию
    notional = entry_price * position_size
    margin_used = notional / LEVERAGE
    commission = notional * COMMISSION_RATE

    # 3) Добавляем запись в open_positions
    add_open_position(
        symbol=symbol,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size
    )

    # 4) Обновляем свободный баланс
    state = load_bot_state()
    new_usd_balance = state['usd_balance'] - margin_used - commission
    save_bot_state(
        usd_balance=new_usd_balance,
        btc_balance=state['btc_balance'],
        entry_price=state['entry_price'],
        stop_loss=state['stop_loss'],
        take_profit=state['take_profit'],
        fraction=state['fraction'],
        risk_per_trade=state['risk_per_trade'],
        in_trade=state['in_trade']
    )
    logger.info(
        f"Executed entry: margin_used={margin_used:.2f}, commission={commission:.2f}, new USD={new_usd_balance:.2f}"
    )
    return resp


def execute_exit(symbol: str, position_id: int, exit_price: float):
    """
    Закрывает позицию по её ID: ордер Sell, лог трейда и удаление open_position, обновление bot_state.
    """
    # 1) Получаем позицию из БД
    positions = get_open_positions()
    pos = next((p for p in positions if p['id'] == position_id), None)
    if not pos:
        logger.error(f"Position {position_id} not found")
        return None
    entry_price = pos['entry_price']
    position_size = pos['position_size']

    # 2) Отправляем Sell-ордер
    resp = place_order(symbol, side='Sell', amount=position_size)

    # 3) Рассчитываем прибыль и комиссию
    commission = (entry_price + exit_price) * position_size * COMMISSION_RATE
    profit = (exit_price - entry_price) * position_size - commission

    # 4) Логируем и удаляем позицию
    log_trade(entry_price, exit_price, position_size, profit)
    remove_open_position(position_id)

    # 5) Освобождаем маржу обратно в баланс
    notional = entry_price * position_size
    margin_used = notional / LEVERAGE
    state = load_bot_state()
    new_usd_balance = state['usd_balance'] + margin_used - commission
    save_bot_state(
        usd_balance=new_usd_balance,
        btc_balance=state['btc_balance'],
        entry_price=state['entry_price'],
        stop_loss=state['stop_loss'],
        take_profit=state['take_profit'],
        fraction=state['fraction'],
        risk_per_trade=state['risk_per_trade'],
        in_trade=state['in_trade']
    )
    logger.info(
        f"Executed exit: profit={profit:.2f}, commission={commission:.2f}, new USD={new_usd_balance:.2f}"
    )
    return resp
