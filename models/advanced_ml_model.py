#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Продвинутые модели машинного обучения для торговли
Включает LSTM, GRU, Transformer и ансамблевые методы
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Пытаемся импортировать библиотеки ML
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn не установлен. Установите: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, BatchNormalization, 
        Bidirectional, Attention, MultiHeadAttention,
        LayerNormalization, Input, Concatenate
    )
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow не установлен. Установите: pip install tensorflow")

logger = logging.getLogger(__name__)

class AdvancedMLModel:
    """
    Продвинутая модель машинного обучения для прогнозирования цен
    """
    
    def __init__(self, sequence_length=60, prediction_horizon=1):
        """
        Инициализация модели
        
        Args:
            sequence_length: Длина последовательности для обучения
            prediction_horizon: Горизонт прогнозирования (количество шагов вперед)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = None
        self.model = None
        self.feature_columns = []
        self.model_type = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготавливает признаки для обучения
        """
        logger.info("Подготовка признаков для ML модели...")
        
        df_features = df.copy()
        
        # Базовые признаки цены
        df_features['price_change'] = df_features['close'].pct_change()
        df_features['price_change_5'] = df_features['close'].pct_change(5)
        df_features['price_change_10'] = df_features['close'].pct_change(10)
        
        # Волатильность
        df_features['volatility'] = df_features['price_change'].rolling(20).std()
        df_features['volatility_ratio'] = df_features['volatility'] / df_features['volatility'].rolling(60).mean()
        
        # Объемные индикаторы
        df_features['volume_change'] = df_features['volume'].pct_change()
        df_features['volume_sma_ratio'] = df_features['volume'] / df_features['volume'].rolling(20).mean()
        
        # Технические индикаторы (если есть)
        tech_indicators = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'BB_upper', 'BB_lower']
        for indicator in tech_indicators:
            if indicator in df_features.columns:
                # Нормализованные значения
                df_features[f'{indicator}_norm'] = (df_features[indicator] - df_features[indicator].rolling(60).mean()) / df_features[indicator].rolling(60).std()
                # Изменения
                df_features[f'{indicator}_change'] = df_features[indicator].pct_change()
        
        # Временные признаки
        if 'start_at' in df_features.columns:
            df_features['start_at'] = pd.to_datetime(df_features['start_at'])
            df_features['hour'] = df_features['start_at'].dt.hour
            df_features['day_of_week'] = df_features['start_at'].dt.dayofweek
            df_features['month'] = df_features['start_at'].dt.month
            
            # Циклические признаки
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
            df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # Лаговые признаки
        for lag in [1, 2, 3, 5, 10]:
            df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
            df_features[f'volume_lag_{lag}'] = df_features['volume'].shift(lag)
            if 'RSI' in df_features.columns:
                df_features[f'RSI_lag_{lag}'] = df_features['RSI'].shift(lag)
        
        # Скользящие средние соотношения
        for window in [5, 10, 20, 50]:
            df_features[f'price_sma_{window}_ratio'] = df_features['close'] / df_features['close'].rolling(window).mean()
            df_features[f'volume_sma_{window}_ratio'] = df_features['volume'] / df_features['volume'].rolling(window).mean()
        
        # Убираем NaN и сохраняем список признаков
        df_features = df_features.dropna()
        
        # Выбираем числовые колонки для обучения
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Подготовлено {len(self.feature_columns)} признаков для обучения")
        return df_features
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создает последовательности для обучения RNN/LSTM
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i:i+self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Создает улучшенную LSTM модель
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        model = Sequential([
            # Первый LSTM слой
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            # Второй LSTM слой
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # Третий LSTM слой
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Полносвязные слои
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(8, activation='relu'),
            
            # Выходной слой
            Dense(self.prediction_horizon, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_gru_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Создает GRU модель
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        model = Sequential([
            # Bidirectional GRU слои
            Bidirectional(GRU(64, return_sequences=True), input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(GRU(32, return_sequences=False)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Полносвязные слои
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            Dense(self.prediction_horizon, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_transformer_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Создает Transformer модель для временных рядов
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8, 
            key_dim=64
        )(inputs, inputs)
        
        # Residual connection
        attention_output = LayerNormalization()(inputs + attention_output)
        
        # Feed forward
        ffn_output = Dense(128, activation='relu')(attention_output)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        ffn_output = LayerNormalization()(attention_output + ffn_output)
        
        # Global average pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        
        # Final layers
        dense1 = Dense(64, activation='relu')(pooled)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.2)(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(0.1)(dense2)
        
        outputs = Dense(self.prediction_horizon, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_ensemble_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Создает ансамблевую модель из LSTM и GRU
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        inputs = Input(shape=input_shape)
        
        # LSTM ветка
        lstm_branch = LSTM(64, return_sequences=True)(inputs)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = Dropout(0.2)(lstm_branch)
        lstm_branch = LSTM(32, return_sequences=False)(lstm_branch)
        lstm_branch = Dense(16, activation='relu')(lstm_branch)
        
        # GRU ветка
        gru_branch = GRU(64, return_sequences=True)(inputs)
        gru_branch = BatchNormalization()(gru_branch)
        gru_branch = Dropout(0.2)(gru_branch)
        gru_branch = GRU(32, return_sequences=False)(gru_branch)
        gru_branch = Dense(16, activation='relu')(gru_branch)
        
        # Объединение
        combined = Concatenate()([lstm_branch, gru_branch])
        combined = Dense(32, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.2)(combined)
        
        combined = Dense(16, activation='relu')(combined)
        combined = Dropout(0.1)(combined)
        
        outputs = Dense(self.prediction_horizon, activation='linear')(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_neural_network(self, df: pd.DataFrame, model_type='lstm', validation_split=0.2, epochs=100):
        """
        Обучает нейронную сеть
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow не установлен. Используйте train_sklearn_model")
            return None
        
        logger.info(f"Обучение {model_type.upper()} модели...")
        
        # Подготовка данных
        df_features = self.prepare_features(df)
        
        # Целевая переменная - будущая цена
        target_col = 'close'
        df_features[f'{target_col}_future'] = df_features[target_col].shift(-self.prediction_horizon)
        df_features = df_features.dropna()
        
        # Выбор признаков и целевой переменной
        X = df_features[self.feature_columns].values
        y = df_features[f'{target_col}_future'].values
        
        # Нормализация
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Создание последовательностей
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        if len(X_seq) == 0:
            logger.error("Недостаточно данных для создания последовательностей")
            return None
        
        logger.info(f"Создано {len(X_seq)} последовательностей для обучения")
        
        # Разделение на train/val
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Создание модели
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        
        if model_type == 'lstm':
            self.model = self.build_lstm_model(input_shape)
        elif model_type == 'gru':
            self.model = self.build_gru_model(input_shape)
        elif model_type == 'transformer':
            self.model = self.build_transformer_model(input_shape)
        elif model_type == 'ensemble':
            self.model = self.build_ensemble_model(input_shape)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        self.model_type = model_type
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
        ]
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Оценка качества
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        logger.info(f"Обучение завершено:")
        logger.info(f"  Финальный loss (train): {train_loss[0]:.6f}")
        logger.info(f"  Финальный loss (val): {val_loss[0]:.6f}")
        
        return history
    
    def train_sklearn_model(self, df: pd.DataFrame, model_type='random_forest'):
        """
        Обучает модель на основе sklearn
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn не установлен")
            return None
        
        logger.info(f"Обучение {model_type} модели...")
        
        # Подготовка данных
        df_features = self.prepare_features(df)
        
        # Целевая переменная
        target_col = 'close'
        df_features[f'{target_col}_future'] = df_features[target_col].shift(-self.prediction_horizon)
        df_features = df_features.dropna()
        
        X = df_features[self.feature_columns]
        y = df_features[f'{target_col}_future']
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Нормализация
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Выбор модели
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        self.model_type = model_type
        
        # Обучение
        self.model.fit(X_train_scaled, y_train)
        
        # Оценка качества
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Обучение завершено:")
        logger.info(f"  R² (train): {train_score:.4f}")
        logger.info(f"  R² (test): {test_score:.4f}")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'mse': mse,
            'mae': mae
        }
    
    def predict(self, df: pd.DataFrame, steps_ahead: int = 1) -> np.ndarray:
        """
        Делает прогноз на основе обученной модели
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Подготовка данных
        df_features = self.prepare_features(df)
        X = df_features[self.feature_columns].values
        
        if self.scaler is None:
            raise ValueError("Скейлер не инициализирован")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type in ['lstm', 'gru', 'transformer', 'ensemble']:
            # Для нейронных сетей нужны последовательности
            if len(X_scaled) < self.sequence_length:
                raise ValueError(f"Недостаточно данных. Нужно минимум {self.sequence_length} записей")
            
            # Берем последние sequence_length записей
            X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            prediction = self.model.predict(X_seq, verbose=0)
        else:
            # Для sklearn моделей
            X_last = X_scaled[-1:] # Последняя запись
            prediction = self.model.predict(X_last)
        
        return prediction.flatten()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Возвращает важность признаков (для sklearn моделей)
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning("Модель не поддерживает feature importance")
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """
        Сохраняет модель
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        try:
            if self.model_type in ['lstm', 'gru', 'transformer', 'ensemble']:
                self.model.save(f"{filepath}_{self.model_type}.h5")
            else:
                import joblib
                joblib.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'model_type': self.model_type
                }, f"{filepath}_{self.model_type}.pkl")
            
            logger.info(f"Модель сохранена: {filepath}_{self.model_type}")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
    
    def load_model(self, filepath: str):
        """
        Загружает модель
        """
        try:
            if filepath.endswith('.h5'):
                self.model = tf.keras.models.load_model(filepath)
                self.model_type = 'neural_network'
            else:
                import joblib
                data = joblib.load(filepath)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_columns = data['feature_columns']
                self.model_type = data['model_type']
            
            logger.info(f"Модель загружена: {filepath}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")

