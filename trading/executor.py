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
    Private Bybit client for trading: sets API keys and leverage.
    """
    exchange = ccxt.bybit({
        'apiKey': BYBIT_API_KEY,
        'secret': BYBIT_API_SECRET,
        'enableRateLimit': True,
    })
    # Monkey-patch fetch_currencies and unified checks to avoid private endpoint calls
    exchange.has['fetchCurrencies'] = False
    exchange.fetch_currencies = lambda params=None: {}
    # Disable unified API to skip is_unified_enabled calls
    exchange.has['fetchOHLCV'] = True  # keep fetchOHLCV
    exchange.is_unified_enabled = lambda: (False, False)

    # Set leverage if supported
    try:
        if exchange.has.get('setLeverage'):
            exchange.setLeverage(LEVERAGE, symbol)
    except Exception as e:
        logger.warning(f"Leverage setup failed: {e}")

    return exchange


def place_order(symbol: str, side: str, amount: float, price: float = None):
    """
    Places an order on Bybit using unified create_order.
    side: 'Buy' or 'Sell'; amount: base asset; price=None => market order.
    """
    client = init_trading_client(symbol)
    order_type = 'market' if price is None else 'limit'
    # create_order(symbol, type, side, amount, price, params)
    params = {}
    logger.info(f"Placing order: {{'symbol': symbol, 'type': order_type, 'side': side.upper(), 'amount': amount, 'price': price}}")
    resp = client.create_order(symbol, order_type, side.upper(), amount, price, params)
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
    # Send order
    resp = place_order(symbol, side='Buy', amount=position_size)

    # Calculate margin and commission
    notional = entry_price * position_size
    margin_used = notional / LEVERAGE
    commission = notional * COMMISSION_RATE

    # Record open position
    position_id = add_open_position(
        symbol=symbol,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size
    )

    # Update free USD balance
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
        f"Executed entry (ID {position_id}): margin_used={margin_used:.2f}, commission={commission:.2f}, new USD={new_usd:.2f}"
    )
    return {'id': position_id, 'response': resp}


def execute_exit(symbol: str, position_id: int, exit_price: float):
    """
    Executes Market Sell: sends order, logs trade, removes position, updates balance.
    """
    # Find position
    positions = get_open_positions()
    pos = next((p for p in positions if p['id'] == position_id), None)
    if not pos:
        logger.error(f"Position {position_id} not found")
        return None

    entry_price = pos['entry_price']
    position_size = pos['position_size']

    # Send sell order
    resp = place_order(symbol, side='Sell', amount=position_size)

    # Calculate profit and commission
    commission = (entry_price + exit_price) * position_size * COMMISSION_RATE
    profit = (exit_price - entry_price) * position_size - commission

    # Log trade and remove position
    log_trade(entry_price, exit_price, position_size, profit)
    remove_open_position(position_id)

    # Return margin to balance
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
    return {'id': position_id, 'response': resp, 'profit': profit}
