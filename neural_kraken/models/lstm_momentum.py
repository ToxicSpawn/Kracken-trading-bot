"""LSTM-based momentum trading model."""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class MomentumLSTM:
    """LSTM model for momentum trading."""

    def __init__(
        self,
        input_shape: Tuple[int, int] = (60, 2),
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Initialize LSTM model.

        Args:
            input_shape: (timesteps, features)
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model: Optional[Sequential] = None
        self.scaler = MinMaxScaler()
        self.is_trained = False

    def build_model(self) -> Sequential:
        """Build the LSTM model."""
        model = Sequential([
            LSTM(
                self.lstm_units,
                return_sequences=True,
                input_shape=self.input_shape,
            ),
            Dropout(self.dropout_rate),
            LSTM(32, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),  # 1 for buy, 0 for sell
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        logger.info("Built LSTM model")
        return model

    def preprocess_data(
        self,
        data: pd.DataFrame,
        fit_scaler: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for LSTM.

        Args:
            data: DataFrame with price and volume columns
            fit_scaler: Whether to fit the scaler

        Returns:
            (X, y) arrays for training or (X, None) for prediction
        """
        # Select features
        features = ["price", "volume"]
        if not all(col in data.columns for col in features):
            raise ValueError(f"Data must contain columns: {features}")

        feature_data = data[features].copy()

        # Normalize
        if fit_scaler:
            feature_data = pd.DataFrame(
                self.scaler.fit_transform(feature_data),
                columns=features,
                index=feature_data.index,
            )
        else:
            feature_data = pd.DataFrame(
                self.scaler.transform(feature_data),
                columns=features,
                index=feature_data.index,
            )

        # Create sequences
        X = []
        y = []
        sequence_length = self.input_shape[0]

        for i in range(len(feature_data) - sequence_length):
            X.append(feature_data.iloc[i : i + sequence_length].values)
            # Target: 1 if next price > current price, else 0
            if i + sequence_length < len(data):
                y.append(1 if data.iloc[i + sequence_length]["price"] > data.iloc[i + sequence_length - 1]["price"] else 0)

        X = np.array(X)
        y = np.array(y) if y else None

        return X, y

    def train(
        self,
        data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
    ) -> None:
        """
        Train the LSTM model.

        Args:
            data: Training data DataFrame
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        if self.model is None:
            self.build_model()

        # Preprocess data
        X, y = self.preprocess_data(data, fit_scaler=True)

        if y is None:
            raise ValueError("Cannot train without target values")

        # Train
        logger.info(f"Training LSTM model on {len(X)} samples...")
        history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
        )

        self.is_trained = True
        logger.info("LSTM model training complete")

        return history

    def predict(self, data: pd.DataFrame) -> float:
        """
        Make prediction.

        Args:
            data: Input data DataFrame

        Returns:
            Prediction probability (0-1)
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() or train() first.")

        # Preprocess
        X, _ = self.preprocess_data(data, fit_scaler=False)

        if len(X) == 0:
            return 0.5  # Neutral prediction

        # Predict
        prediction = self.model.predict(X[-1:], verbose=0)[0][0]
        return float(prediction)

    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise RuntimeError("Model not built.")
        self.model.save(filepath)
        logger.info(f"Saved model to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from file."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required.")
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Loaded model from {filepath}")

