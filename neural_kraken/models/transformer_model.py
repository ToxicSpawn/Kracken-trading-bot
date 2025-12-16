"""Transformer-based trading model with complete training pipeline."""

from __future__ import annotations

import logging
from typing import Tuple, Optional, Dict, Any
import os
import json

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.metrics import BinaryAccuracy, AUC
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

from sklearn.preprocessing import MinMaxScaler
import joblib

logger = logging.getLogger(__name__)


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu", kernel_regularizer=l2(0.01)),
            Dense(embed_dim, kernel_regularizer=l2(0.01)),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TimeSeriesTransformer:
    """Transformer model for time series trading prediction."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_heads: int = 4,
        ff_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        model_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize transformer model.

        Args:
            input_shape: (timesteps, features)
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_layers: Number of transformer layers
            dropout_rate: Dropout rate
            model_path: Path to load/save model
            config: Additional configuration
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        self.input_shape = input_shape
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model_path = model_path
        self.config = config or {}
        self.model: Optional[Model] = None
        self.scaler = MinMaxScaler()

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self) -> Model:
        """Build the transformer model."""
        inputs = Input(shape=self.input_shape)

        # Positional embedding
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(inputs)
        x = Dropout(self.dropout_rate)(x)

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(64, self.num_heads, self.ff_dim, self.dropout_rate)(x)

        # Output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=10000,
            decay_rate=0.9,
        )

        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=BinaryCrossentropy(),
            metrics=[
                BinaryAccuracy(),
                AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        self.model = model
        logger.info("Built transformer model")
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Train the transformer model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            save_path: Path to save model

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        save_path = save_path or self.model_path or "transformer_model.h5"

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=5,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=save_path,
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
        ]

        # Train
        logger.info(f"Training transformer model on {len(X_train)} samples...")
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Transformer model training complete")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() or train() first.")
        return self.model.predict(X, verbose=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model."""
        if self.model is None:
            raise RuntimeError("Model not built.")
        results = self.model.evaluate(X, y, verbose=0)
        metrics = self.model.metrics_names
        return dict(zip(metrics, results))

    def save_model(self, path: str) -> None:
        """Save model and configuration."""
        if self.model is None:
            raise RuntimeError("Model not built.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)

        # Save config
        config_path = os.path.join(os.path.dirname(path), "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "input_shape": self.input_shape,
                    "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim,
                    "num_layers": self.num_layers,
                    "dropout_rate": self.dropout_rate,
                    "config": self.config,
                },
                f,
                indent=2,
            )

        # Save scaler
        scaler_path = os.path.join(os.path.dirname(path), "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"Saved model to {path}")

    def load_model(self, path: str) -> None:
        """Load model and configuration."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required.")

        self.model = tf.keras.models.load_model(path)

        # Load config
        config_path = os.path.join(os.path.dirname(path), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.input_shape = tuple(config["input_shape"])
                self.num_heads = config["num_heads"]
                self.ff_dim = config["ff_dim"]
                self.num_layers = config["num_layers"]
                self.dropout_rate = config["dropout_rate"]
                self.config = config.get("config", {})

        # Load scaler
        scaler_path = os.path.join(os.path.dirname(path), "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        logger.info(f"Loaded model from {path}")

    def preprocess_data(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        fit_scaler: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for transformer.

        Args:
            data: DataFrame with features
            sequence_length: Sequence length
            fit_scaler: Whether to fit the scaler

        Returns:
            (X, y) arrays
        """
        # Select features
        feature_cols = ["price", "volume", "returns", "ma_5", "ma_20", "rsi", "volatility"]
        available_cols = [col for col in feature_cols if col in data.columns]

        if not available_cols:
            raise ValueError("No valid features found in data")

        feature_data = data[available_cols].copy()

        # Normalize
        if fit_scaler:
            feature_data = pd.DataFrame(
                self.scaler.fit_transform(feature_data),
                columns=available_cols,
                index=feature_data.index,
            )
        else:
            feature_data = pd.DataFrame(
                self.scaler.transform(feature_data),
                columns=available_cols,
                index=feature_data.index,
            )

        # Create sequences
        X = []
        y = []

        for i in range(len(feature_data) - sequence_length):
            X.append(feature_data.iloc[i : i + sequence_length].values)
            # Target: 1 if next price > current price
            if i + sequence_length < len(data):
                current_price = data.iloc[i + sequence_length - 1]["price"] if "price" in data.columns else data.iloc[i + sequence_length - 1].iloc[0]
                next_price = data.iloc[i + sequence_length]["price"] if "price" in data.columns else data.iloc[i + sequence_length].iloc[0]
                y.append(1 if next_price > current_price else 0)

        X = np.array(X)
        y = np.array(y) if y else None

        return X, y

