# trading/executor.py

import ccxt
import logging
from utils.config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
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
    Private Bybit client for trading:
    - sets leverage only (margin mode should be set in UI)
    """
    exchange = ccxt.bybit({
        'apiKey': BYBIT_API_KEY,
        'secret': BYBIT_API_SECRET,
        'enableRateLimit': True,
    })
    # Monkey-patch fetch_currencies to avoid private endpoint calls
    exchange.has['fetchCurrencies'] = False
    exchange.fetch_currencies = lambda params=None: {}

    # Set leverage if supported
    try:
        if exchange.has.get('setLeverage'):
            exchange.setLeverage(LEVERAGE, symbol)
        else:
            exchange.private_post_position_leverage_save({
                'symbol': symbol,
                'buy_leverage': LEVERAGE,
                'sell_leverage': LEVERAGE
            })
    except Exception as e:
        logger.warning(f"Leverage setup failed: {e}")

    return exchange


def place_order(symbol: str, side: str, amount: float, price: float = None):
    """
    Places an order on Bybit.
    side: 'Buy' or 'Sell'; amount: base asset; price=None => market order.
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
    Executes Market Buy: sends order, records position, updates balance.
    """
    resp = place_order(symbol, side='Buy', amount=position_size)

    notional = entry_price * position_size
    margin_used = notional / LEVERAGE
    commission = notional * COMMISSION_RATE

    add_open_position(
        symbol=symbol,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size
    )

    state = load_bot_state()
    new_usd = state['usd_balance'] - margin_used - commission
    save_bot_state(
        usd_balance=new_usd,
        btc_balance=state['btc_balance'],
        entry_price=state['entry_price'],
        stop_loss=state['stop_loss'],
        take_profit=state['take_profit'],
        fraction=state['fraction'],
        risk_per_trade=state['risk_per_trade'],
        in_trade=state['in_trade']
    )

    logger.info(
        f"Executed entry: margin_used={margin_used:.2f}, commission={commission:.2f}, new USD={new_usd:.2f}"
    )
    return resp


def execute_exit(symbol: str, position_id: int, exit_price: float):
    """
    Executes Market Sell: sends order, logs trade, removes position, updates balance.
    """
    positions = get_open_positions()
    pos = next((p for p in positions if p['id'] == position_id), None)
    if not pos:
        logger.error(f"Position {position_id} not found")
        return None

    entry_price = pos['entry_price']
    position_size = pos['position_size']

    resp = place_order(symbol, side='Sell', amount=position_size)

    commission = (entry_price + exit_price) * position_size * COMMISSION_RATE
    profit = (exit_price - entry_price) * position_size - commission

    log_trade(entry_price, exit_price, position_size, profit)
    remove_open_position(position_id)

    notional = entry_price * position_size
    margin_used = notional / LEVERAGE
    state = load_bot_state()
    new_usd = state['usd_balance'] + margin_used - commission
    save_bot_state(
        usd_balance=new_usd,
        btc_balance=state['btc_balance'],
        entry_price=state['entry_price'],
        stop_loss=state['stop_loss'],
        take_profit=state['take_profit'],
        fraction=state['fraction'],
        risk_per_trade=state['risk_per_trade'],
        in_trade=state['in_trade']
    )

    logger.info(
        f"Executed exit: profit={profit:.2f}, commission={commission:.2f}, new USD={new_usd:.2f}"
    )
    return resp
