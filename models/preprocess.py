# models/preprocess.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_features(df, features, sequence_length=50, external_scaler=None):
    """
    Подготавливает входные данные (X) и целевую переменную (y) для обучения модели.

    - df: DataFrame с рыночными данными.
    - features: Список столбцов, которые будут использоваться в модели.
    - sequence_length: Длина окна последовательностей.
    - external_scaler: При наличии передает уже обученный scaler.
    """
    df_filtered = df[features].copy()

    # Добавляем целевой признак (напр., направление движения цены)
    df_filtered["future_close"] = df["close"].shift(-1)
    df_filtered["direction"] = (df_filtered["future_close"] > df_filtered["close"]).astype(int)

    if external_scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_filtered[features])
    else:
        scaler = external_scaler
        scaled_data = scaler.transform(df_filtered[features])

    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - 1):
        X.append(scaled_data[i - sequence_length:i])
        y.append(df_filtered["direction"].iloc[i])

    return np.array(X), np.array(y), scaler
