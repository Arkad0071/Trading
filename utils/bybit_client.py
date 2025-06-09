import ccxt
from .config import BYBIT_API_KEY, BYBIT_API_SECRET


def get_public_client():
    """Return a CCXT Bybit client for public endpoints."""
    exchange = ccxt.bybit({
        'enableRateLimit': True,
    })
    # Avoid unnecessary private calls
    exchange.options['fetchCurrencies'] = False
    return exchange


def get_private_client():
    """Return an authenticated CCXT Bybit client."""
    exchange = ccxt.bybit({
        'apiKey': BYBIT_API_KEY,
        'secret': BYBIT_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })
    exchange.options['fetchCurrencies'] = False
    return exchange
