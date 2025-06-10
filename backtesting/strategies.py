import pandas as pd


def rsi_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Generate BUY/SELL/HOLD signals based on RSI."""
    df["signal"] = df["RSI"].apply(
        lambda rsi: "BUY" if rsi < 30 else ("SELL" if rsi > 70 else "HOLD")
    )
    return df


def macd_rsi_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Combine MACD crossovers with RSI filters."""
    df["signal"] = "HOLD"
    df.loc[(df["MACD"] > df["MACD_signal"]) & (df["RSI"] < 30), "signal"] = "BUY"
    df.loc[(df["MACD"] < df["MACD_signal"]) & (df["RSI"] > 70), "signal"] = "SELL"
    return df


def bollinger_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Buy when price is below the lower band, sell above the upper band."""
    df["signal"] = "HOLD"
    df.loc[df["close"] < df["BB_lower"], "signal"] = "BUY"
    df.loc[df["close"] > df["BB_upper"], "signal"] = "SELL"
    return df


STRATEGIES = {
    "RSI only": rsi_strategy,
    "MACD + RSI": macd_rsi_strategy,
    "Bollinger": bollinger_strategy,
}
