# utils/config.py
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

MARGIN_MODE = os.getenv("BYBIT_MARGIN_MODE", "cross")
LEVERAGE = int(os.getenv("BYBIT_LEVERAGE", "5"))
COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.001"))

DEFAULT_RISK_PCT = 1.0    # риск 1% от баланса
DEFAULT_SL_PCT   = 2.0    # стоп-лосс 2% от цены входа
DEFAULT_TP_RATIO = 2.0    # тейк-профит в 2 раза дальше SL
