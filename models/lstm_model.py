# models/lstm_model.py
import tensorflow as tf
from tensorflow import keras
import os
import logging

logger = logging.getLogger(__name__)


class LSTMModel:
    def __init__(self, sequence_length=50, num_features=6):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.sequence_length, self.num_features)),
            keras.layers.LSTM(100, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(100, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info("Модель LSTM построена и скомпилирована.")
        return model

    def train(self, X, y, epochs=50, batch_size=32, callbacks=None):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)

    def evaluate_model(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return loss, acc

    def predict(self, X_input):
        return self.model.predict(X_input)

    def save_model(self, file_path="lstm_model.h5"):
        self.model.save(file_path)
        logger.info(f"Модель сохранена в {file_path}")

    def load_model(self, file_path="lstm_model.h5"):
        if os.path.exists(file_path):
            self.model = keras.models.load_model(file_path)
            logger.info(f"Модель загружена из {file_path}")
        else:
            logger.error("Файл модели не найден!")
