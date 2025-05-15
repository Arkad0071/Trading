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


def init_trading_client(symbol: str = "BTC/USDT:USDT"):
    """
    Private Bybit client for USDT futures trading:
    - uses linear USDT futures endpoint
    - sets margin mode (cross/isolated) if supported
    - sets leverage if supported
    """
    exchange = ccxt.bybit({
        'apiKey': BYBIT_API_KEY,
        'secret': BYBIT_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })
    # Switch margin mode if supported
    if exchange.has.get('setMarginMode'):
        try:
            exchange.setMarginMode(MARGIN_MODE, symbol)
        except Exception as e:
            logger.warning(f"Margin mode setup failed: {e}")
    # Set leverage if supported
    if exchange.has.get('setLeverage'):
        try:
            exchange.setLeverage(LEVERAGE, symbol)
        except Exception as e:
            logger.warning(f"Leverage setup failed for {symbol}: {e}")
    return exchange


def place_order(symbol: str, side: str, amount: float, price: float = None):
    """
    Places an order on Bybit USDT futures:
    - market order if price is None
    - limit order otherwise
    """
    client = init_trading_client(symbol)
    logger.info(f"Placing {side.upper()} order: symbol={symbol}, amount={amount}, price={price}")
    if price is None:
        resp = client.create_market_order(symbol, side.upper(), amount)
    else:
        resp = client.create_limit_order(symbol, side.upper(), amount, price)
    logger.info(f"Order response: {resp}")
    return resp


def execute_entry(symbol: str, entry_price: float, position_size: float, stop_loss: float, take_profit: float):
    """
    Execute Market Buy: sends order, records position, updates balance.
    """
    resp = place_order(symbol, 'buy', position_size)
    notional = entry_price * position_size
    margin_used = notional / LEVERAGE
    commission = notional * COMMISSION_RATE
    position_id = add_open_position(symbol, entry_price, stop_loss, take_profit, position_size)
    state = load_bot_state()
    new_usdt = state['usd_balance'] - margin_used - commission
    save_bot_state(
        usd_balance=new_usdt,
        btc_balance=state['btc_balance'],
        entry_price=state['entry_price'],
        stop_loss=state['stop_loss'],
        take_profit=state['take_profit'],
        fraction=state['fraction'],
        risk_per_trade=state['risk_per_trade'],
        in_trade=state['in_trade']
    )
    logger.info(f"Executed entry #{position_id}: margin_used={margin_used:.2f}, commission={commission:.2f}, new USD={new_usdt:.2f}")
    return {'id': position_id, 'response': resp}


def execute_exit(symbol: str, position_id: int, exit_price: float):
    """
    Execute Market Sell: sends order, logs trade, removes position, updates balance.
    """
    positions = get_open_positions()
    pos = next((p for p in positions if p['id'] == position_id), None)
    if not pos:
        logger.error(f"Position {position_id} not found")
        return None
    entry_price = pos['entry_price']
    size = pos['position_size']
    resp = place_order(symbol, 'sell', size)
    commission = (entry_price + exit_price) * size * COMMISSION_RATE
    profit = (exit_price - entry_price) * size - commission
    log_trade(entry_price, exit_price, size, profit)
    remove_open_position(position_id)
    margin_used = entry_price * size / LEVERAGE
    state = load_bot_state()
    new_usdt = state['usd_balance'] + margin_used - commission
    save_bot_state(
        usd_balance=new_usdt,
        btc_balance=state['btc_balance'],
        entry_price=state['entry_price'],
        stop_loss=state['stop_loss'],
        take_profit=state['take_profit'],
        fraction=state['fraction'],
        risk_per_trade=state['risk_per_trade'],
        in_trade=state['in_trade']
    )
    logger.info(f"Executed exit #{position_id}: profit={profit:.2f}, commission={commission:.2f}, new USD={new_usdt:.2f}")
    return {'id': position_id, 'response': resp, 'profit': profit}


