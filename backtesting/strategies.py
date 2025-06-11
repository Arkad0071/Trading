import pandas as pd

# Thresholds for RSI based signals.  Higher values make BUY/SELL entries
# appear more often compared to the classic 30/70 levels.
RSI_BUY_THRESHOLD = 40
RSI_SELL_THRESHOLD = 60

def rsi_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Generate BUY/SELL/HOLD signals based on RSI."""
    df["signal"] = df["RSI"].apply(
        lambda rsi: "BUY" if rsi < RSI_BUY_THRESHOLD else (
            "SELL" if rsi > RSI_SELL_THRESHOLD else "HOLD"
        )
    )
    return df


def macd_rsi_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Combine MACD crossovers with RSI filters."""
    df["signal"] = "HOLD"
    df.loc[
        (df["MACD"] > df["MACD_signal"]) & (df["RSI"] < RSI_BUY_THRESHOLD),
        "signal",
    ] = "BUY"
    df.loc[
        (df["MACD"] < df["MACD_signal"]) & (df["RSI"] > RSI_SELL_THRESHOLD),
        "signal",
    ] = "SELL"
    return df


def bollinger_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Buy when price is below the lower band, sell above the upper band."""
    df["signal"] = "HOLD"
    df.loc[df["close"] < df["BB_lower"], "signal"] = "BUY"
    df.loc[df["close"] > df["BB_upper"], "signal"] = "SELL"
    return df


def macd_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Signals based solely on MACD crossovers."""
    df["signal"] = "HOLD"
    df.loc[df["MACD"] > df["MACD_signal"], "signal"] = "BUY"
    df.loc[df["MACD"] < df["MACD_signal"], "signal"] = "SELL"
    return df


def stochastic_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Use the Stochastic oscillator for signals."""
    df["signal"] = "HOLD"
    df.loc[df["STOCH_K"] > df["STOCH_D"], "signal"] = "BUY"
    df.loc[df["STOCH_K"] < df["STOCH_D"], "signal"] = "SELL"
    return df


def sma_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Price above SMA -> BUY, below -> SELL."""
    df["signal"] = "HOLD"
    df.loc[df["close"] > df["SMA_20"], "signal"] = "BUY"
    df.loc[df["close"] < df["SMA_20"], "signal"] = "SELL"
    return df


def ema_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Price above EMA -> BUY, below -> SELL."""
    df["signal"] = "HOLD"
    df.loc[df["close"] > df["EMA_20"], "signal"] = "BUY"
    df.loc[df["close"] < df["EMA_20"], "signal"] = "SELL"
    return df


def macd_bollinger_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """MACD crossovers filtered by Bollinger bands."""
    df["signal"] = "HOLD"
    buy = (df["MACD"] > df["MACD_signal"]) & (df["close"] < df["BB_lower"])
    sell = (df["MACD"] < df["MACD_signal"]) & (df["close"] > df["BB_upper"])
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def macd_stochastic_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Combine MACD crossovers with Stochastic filter."""
    df["signal"] = "HOLD"
    buy = (df["MACD"] > df["MACD_signal"]) & (df["STOCH_K"] > df["STOCH_D"])
    sell = (df["MACD"] < df["MACD_signal"]) & (df["STOCH_K"] < df["STOCH_D"])
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def rsi_bollinger_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """RSI oversold/overbought plus Bollinger band filter."""
    df["signal"] = "HOLD"
    buy = (df["RSI"] < RSI_BUY_THRESHOLD) & (df["close"] < df["BB_lower"])
    sell = (df["RSI"] > RSI_SELL_THRESHOLD) & (df["close"] > df["BB_upper"])
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def rsi_stochastic_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """RSI with Stochastic confirmation."""
    df["signal"] = "HOLD"
    buy = (df["RSI"] < RSI_BUY_THRESHOLD) & (df["STOCH_K"] > df["STOCH_D"])
    sell = (df["RSI"] > RSI_SELL_THRESHOLD) & (df["STOCH_K"] < df["STOCH_D"])
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def bollinger_stochastic_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger band extremes confirmed by Stochastic."""
    df["signal"] = "HOLD"
    buy = (df["close"] < df["BB_lower"]) & (df["STOCH_K"] > df["STOCH_D"])
    sell = (df["close"] > df["BB_upper"]) & (df["STOCH_K"] < df["STOCH_D"])
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def sma_ema_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Cross of SMA and EMA."""
    df["signal"] = "HOLD"
    buy = df["SMA_20"] > df["EMA_20"]
    sell = df["SMA_20"] < df["EMA_20"]
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def rsi_macd_bollinger_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Combine RSI, MACD and Bollinger Bands."""
    df["signal"] = "HOLD"
    buy = (
        (df["MACD"] > df["MACD_signal"])
        & (df["RSI"] < RSI_BUY_THRESHOLD)
        & (df["close"] < df["BB_lower"])
    )
    sell = (
        (df["MACD"] < df["MACD_signal"])
        & (df["RSI"] > RSI_SELL_THRESHOLD)
        & (df["close"] > df["BB_upper"])
    )
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def rsi_macd_stochastic_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """RSI and MACD with Stochastic confirmation."""
    df["signal"] = "HOLD"
    buy = (
        (df["MACD"] > df["MACD_signal"])
        & (df["RSI"] < RSI_BUY_THRESHOLD)
        & (df["STOCH_K"] > df["STOCH_D"])
    )
    sell = (
        (df["MACD"] < df["MACD_signal"])
        & (df["RSI"] > RSI_SELL_THRESHOLD)
        & (df["STOCH_K"] < df["STOCH_D"])
    )
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def rsi_bollinger_stochastic_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """RSI with Bollinger band and Stochastic filters."""
    df["signal"] = "HOLD"
    buy = (
        (df["RSI"] < RSI_BUY_THRESHOLD)
        & (df["close"] < df["BB_lower"])
        & (df["STOCH_K"] > df["STOCH_D"])
    )
    sell = (
        (df["RSI"] > RSI_SELL_THRESHOLD)
        & (df["close"] > df["BB_upper"])
        & (df["STOCH_K"] < df["STOCH_D"])
    )
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def macd_bollinger_stochastic_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """MACD plus Bollinger and Stochastic filters."""
    df["signal"] = "HOLD"
    buy = (
        (df["MACD"] > df["MACD_signal"])
        & (df["close"] < df["BB_lower"])
        & (df["STOCH_K"] > df["STOCH_D"])
    )
    sell = (
        (df["MACD"] < df["MACD_signal"])
        & (df["close"] > df["BB_upper"])
        & (df["STOCH_K"] < df["STOCH_D"])
    )
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def rsi_macd_bollinger_stochastic_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """All indicators combined."""
    df["signal"] = "HOLD"
    buy = (
        (df["MACD"] > df["MACD_signal"])
        & (df["RSI"] < RSI_BUY_THRESHOLD)
        & (df["close"] < df["BB_lower"])
        & (df["STOCH_K"] > df["STOCH_D"])
    )
    sell = (
        (df["MACD"] < df["MACD_signal"])
        & (df["RSI"] > RSI_SELL_THRESHOLD)
        & (df["close"] > df["BB_upper"])
        & (df["STOCH_K"] < df["STOCH_D"])
    )
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def rsi_ema_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Price above EMA with RSI filter."""
    df["signal"] = "HOLD"
    buy = (df["close"] > df["EMA_20"]) & (df["RSI"] < RSI_BUY_THRESHOLD)
    sell = (df["close"] < df["EMA_20"]) & (df["RSI"] > RSI_SELL_THRESHOLD)
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


def sma_macd_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """SMA trend with MACD confirmation."""
    df["signal"] = "HOLD"
    buy = (df["close"] > df["SMA_20"]) & (df["MACD"] > df["MACD_signal"])
    sell = (df["close"] < df["SMA_20"]) & (df["MACD"] < df["MACD_signal"])
    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"
    return df


STRATEGIES = {
    "RSI only": rsi_strategy,
    "MACD only": macd_strategy,
    "Bollinger": bollinger_strategy,
    "Stochastic": stochastic_strategy,
    "SMA": sma_strategy,
    "EMA": ema_strategy,
    "MACD + RSI": macd_rsi_strategy,
    "MACD + Bollinger": macd_bollinger_strategy,
    "MACD + Stochastic": macd_stochastic_strategy,
    "RSI + Bollinger": rsi_bollinger_strategy,
    "RSI + Stochastic": rsi_stochastic_strategy,
    "Bollinger + Stochastic": bollinger_stochastic_strategy,
    "SMA + EMA": sma_ema_strategy,
    "RSI + MACD + Bollinger": rsi_macd_bollinger_strategy,
    "RSI + MACD + Stochastic": rsi_macd_stochastic_strategy,
    "RSI + Bollinger + Stochastic": rsi_bollinger_stochastic_strategy,
    "MACD + Bollinger + Stochastic": macd_bollinger_stochastic_strategy,
    "All indicators": rsi_macd_bollinger_stochastic_strategy,
    "RSI + EMA": rsi_ema_strategy,
    "SMA + MACD": sma_macd_strategy,

}

# --- Programmatically generated strategies ---------------------------------

from itertools import combinations

COMBO_INDICATORS = [
    "RSI",
    "MACD",
    "ATR",
    "SMA",
    "EMA",
    "Bollinger",
    "Stochastic",
]


def _create_combo_strategy(indicators: tuple[str, ...]):
    """Return a strategy that requires all *indicators* to agree."""

    def strategy(df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = "HOLD"
        buy = pd.Series(True, index=df.index)
        sell = pd.Series(True, index=df.index)

        for ind in indicators:
            if ind == "RSI":
                buy &= df["RSI"] < RSI_BUY_THRESHOLD
                sell &= df["RSI"] > RSI_SELL_THRESHOLD
            elif ind == "MACD":
                buy &= df["MACD"] > df["MACD_signal"]
                sell &= df["MACD"] < df["MACD_signal"]
            elif ind == "ATR":
                buy &= df["ATR"] > df["ATR"].shift()
                sell &= df["ATR"] < df["ATR"].shift()
            elif ind == "SMA":
                buy &= df["close"] > df["SMA_20"]
                sell &= df["close"] < df["SMA_20"]
            elif ind == "EMA":
                buy &= df["close"] > df["EMA_20"]
                sell &= df["close"] < df["EMA_20"]
            elif ind == "Bollinger":
                buy &= df["close"] < df["BB_lower"]
                sell &= df["close"] > df["BB_upper"]
            elif ind == "Stochastic":
                buy &= df["STOCH_K"] > df["STOCH_D"]
                sell &= df["STOCH_K"] < df["STOCH_D"]

        df.loc[buy, "signal"] = "BUY"
        df.loc[sell, "signal"] = "SELL"
        return df

    return strategy


def generate_all_indicator_strategies() -> dict[str, callable]:
    """Return strategy functions for every non-empty indicator combination."""
    strategies = {}
    for r in range(1, len(COMBO_INDICATORS) + 1):
        for combo in combinations(COMBO_INDICATORS, r):
            name = " + ".join(combo)
            strategies[name] = _create_combo_strategy(combo)
    return strategies


STRATEGIES.update(generate_all_indicator_strategies())
