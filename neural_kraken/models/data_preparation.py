"""Data preparation for model training."""

from __future__ import annotations

import logging
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessor for model training."""

    def __init__(self, config: Dict) -> None:
        """
        Initialize data preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaler = MinMaxScaler()
        self.feature_columns = [
            "price",
            "volume",
            "returns",
            "ma_5",
            "ma_20",
            "rsi",
            "volatility",
        ]

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        # Price features
        df["returns"] = df["close"].pct_change()
        df["price"] = df["close"]

        # Moving averages
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volatility
        df["volatility"] = df["returns"].rolling(20).std()

        # Additional features
        df["spread"] = df["high"] - df["low"]
        df["volume_change"] = df["volume"].pct_change()

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            df: DataFrame with features
            sequence_length: Length of sequences

        Returns:
            (X, y) arrays
        """
        # Select features
        available_cols = [col for col in self.feature_columns if col in df.columns]
        if not available_cols:
            raise ValueError(f"No valid features found. Available: {df.columns.tolist()}")

        feature_data = df[available_cols].copy()

        # Scale features
        features_scaled = self.scaler.fit_transform(feature_data)

        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - sequence_length):
            X.append(features_scaled[i : i + sequence_length])
            # Target: 1 if price increases in next step, else 0
            if "price" in df.columns and i + sequence_length < len(df):
                current_price = df.iloc[i + sequence_length - 1]["price"]
                next_price = df.iloc[i + sequence_length]["price"]
                y.append(1 if next_price > current_price else 0)

        return np.array(X), np.array(y)

    def prepare_data(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            data: Historical data DataFrame
            sequence_length: Sequence length
            test_size: Test set size ratio

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        # Add technical indicators
        logger.info("Adding technical indicators")
        data = self.add_technical_indicators(data)

        # Drop NA values
        data = data.dropna()

        # Create sequences
        logger.info("Creating sequences")
        X, y = self.create_sequences(data, sequence_length)

        # Split into train and test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Prepared {len(X_train)} training and {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def save_scaler(self, path: str) -> None:
        """Save scaler to file."""
        joblib.dump(self.scaler, path)
        logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path: str) -> None:
        """Load scaler from file."""
        self.scaler = joblib.load(path)
        logger.info(f"Loaded scaler from {path}")

