"""Machine learning strategies."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from strategies.base_strategy import BaseStrategy
from utils.logger import logger

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available, LSTM strategy will be disabled")

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, ML strategies will be disabled")


class LSTMStrategy(BaseStrategy):
    """LSTM-based price prediction strategy."""
    
    def __init__(self, config: Dict):
        super().__init__(config, "LSTM_Strategy")
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM strategy")
        
        ml_config = config.get("strategies", {}).get("ml", {})
        lstm_config = ml_config.get("lstm", {})
        self.symbols = lstm_config.get("symbols", ["BTC/USDT"])
        self.sequence_length = lstm_config.get("sequence_length", 60)
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.data_buffer = {symbol: [] for symbol in self.symbols}
    
    def _build_model(self) -> tf.keras.Model:
        """Build the LSTM model architecture."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 5)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def _prepare_data(self, data: List) -> Optional[np.ndarray]:
        """Prepare data for LSTM prediction."""
        if len(data) < self.sequence_length + 1:
            return None
        
        try:
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(df)
            
            # Create sequences
            X = []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
            
            return np.array(X)
        except Exception as e:
            self.log(f"Error preparing data: {e}")
            return None
    
    async def run(self, data: Dict[str, List]) -> Optional[Dict]:
        """Run the LSTM strategy on new data."""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        signals = {}
        
        for symbol, ohlcv_data in data.items():
            if not ohlcv_data or len(ohlcv_data) < self.sequence_length + 1:
                continue
            
            self.data_buffer[symbol].extend(ohlcv_data)
            if len(self.data_buffer[symbol]) > 2000:
                self.data_buffer[symbol] = self.data_buffer[symbol][-2000:]
            
            try:
                X = self._prepare_data(self.data_buffer[symbol])
                if X is None or len(X) < 1:
                    continue
                
                # Make prediction
                prediction = self.model.predict(X[-1].reshape(1, self.sequence_length, 5), verbose=0)
                predicted_price = self.scaler.inverse_transform(
                    np.concatenate([prediction, np.zeros((1, 4))], axis=1)
                )[0][0]
                
                current_price = ohlcv_data[-1][4]  # Close price
                price_change = (predicted_price - current_price) / current_price * 100
                
                self.log(f"{symbol} - Predicted price: {predicted_price:.2f} (Change: {price_change:.2f}%)")
                
                # Generate trading signal
                if price_change > 1.5:  # 1.5% predicted increase
                    signals[symbol] = {
                        "action": "buy",
                        "confidence": min(price_change/5, 1.0),
                        "predicted_price": float(predicted_price),
                        "current_price": float(current_price)
                    }
                elif price_change < -1.5:  # 1.5% predicted decrease
                    signals[symbol] = {
                        "action": "sell",
                        "confidence": min(abs(price_change)/5, 1.0),
                        "predicted_price": float(predicted_price),
                        "current_price": float(current_price)
                    }
            
            except Exception as e:
                self.log(f"Error in LSTM prediction for {symbol}: {e}")
        
        return signals if signals else None


class RandomForestStrategy(BaseStrategy):
    """Random Forest classification strategy."""
    
    def __init__(self, config: Dict):
        super().__init__(config, "RandomForest_Strategy")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest strategy")
        
        ml_config = config.get("strategies", {}).get("ml", {})
        rf_config = ml_config.get("random_forest", {})
        self.symbols = rf_config.get("symbols", ["BTC/USDT"])
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.data_buffer = {symbol: [] for symbol in self.symbols}
        self.min_data_points = 200
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Price returns
        df['returns'] = df['close'].pct_change()
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        df['macd'], df['signal'] = self._calculate_macd(df['close'])
        
        # ATR
        df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'], 14)
        
        # Bollinger Bands
        df['upper_band'], df['middle_band'], df['lower_band'] = self._calculate_bollinger_bands(df['close'], 20)
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9) -> tuple:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        middle_band = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, middle_band, lower_band
    
    async def run(self, data: Dict[str, List]) -> Optional[Dict]:
        """Run the Random Forest strategy on new data."""
        if not SKLEARN_AVAILABLE:
            return None
        
        signals = {}
        
        for symbol, ohlcv_data in data.items():
            if not ohlcv_data:
                continue
            
            self.data_buffer[symbol].extend(ohlcv_data)
            if len(self.data_buffer[symbol]) > 2000:
                self.data_buffer[symbol] = self.data_buffer[symbol][-2000:]
            
            if len(self.data_buffer[symbol]) < self.min_data_points:
                continue
            
            try:
                df = pd.DataFrame(self.data_buffer[symbol],
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = self._calculate_indicators(df)
                
                if len(df) < self.min_data_points:
                    continue
                
                # Prepare features and target
                features = df[['returns', 'sma_10', 'sma_50', 'rsi', 'macd', 'signal', 'atr',
                              'upper_band', 'middle_band', 'lower_band']]
                target = (df['returns'].shift(-1) > 0).astype(int)
                
                # Train the model if we have enough data
                if len(features) > self.min_data_points * 2:
                    X_train = features[:-1].values
                    y_train = target[1:-1].values
                    self.model.fit(X_train, y_train)
                    accuracy = self.model.score(X_train, y_train)
                    self.log(f"{symbol} - Model trained with accuracy: {accuracy:.2f}")
                
                # Make prediction
                last_features = features.iloc[-1].values.reshape(1, -1)
                prediction = self.model.predict_proba(last_features)[0][1]  # Probability of positive return
                
                self.log(f"{symbol} - Prediction probability: {prediction:.2f}")
                
                # Generate trading signal
                if prediction > 0.65:  # 65% probability of positive return
                    signals[symbol] = {
                        "action": "buy",
                        "confidence": float(prediction),
                        "indicators": features.iloc[-1].to_dict()
                    }
                elif prediction < 0.35:  # 35% probability of positive return
                    signals[symbol] = {
                        "action": "sell",
                        "confidence": float(1 - prediction),
                        "indicators": features.iloc[-1].to_dict()
                    }
            
            except Exception as e:
                self.log(f"Error in Random Forest prediction for {symbol}: {e}")
        
        return signals if signals else None

