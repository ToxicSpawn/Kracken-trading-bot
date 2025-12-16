"""Alternative data sources: news, social media, on-chain."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available for sentiment analysis")

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """News sentiment analyzer."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize news analyzer.

        Args:
            api_key: NewsAPI key (optional)
        """
        self.api_key = api_key
        if TRANSFORMERS_AVAILABLE:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
        else:
            self.sentiment_analyzer = None
            logger.warning("Sentiment analysis disabled. Install transformers.")

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment dictionary
        """
        if not self.sentiment_analyzer:
            return {"label": "NEUTRAL", "score": 0.0, "sentiment_score": 0.0}

        try:
            result = self.sentiment_analyzer(text[:512])  # Truncate to 512 tokens
            score = result[0]["score"]
            label = result[0]["label"]

            return {
                "label": label,
                "score": score,
                "sentiment_score": score if label == "POSITIVE" else -score,
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"label": "NEUTRAL", "score": 0.0, "sentiment_score": 0.0}

    def fetch_news(self, query: str = "bitcoin OR ethereum", hours: int = 24) -> List[Dict]:
        """
        Fetch news articles (placeholder - requires NewsAPI).

        Args:
            query: Search query
            hours: Hours to look back

        Returns:
            List of article dictionaries
        """
        # Placeholder - implement with NewsAPI
        logger.warning("News fetching not implemented. Requires NewsAPI key.")
        return []


class TwitterSentimentAnalyzer:
    """Twitter sentiment analyzer."""

    def __init__(self) -> None:
        """Initialize Twitter analyzer."""
        if TRANSFORMERS_AVAILABLE:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
        else:
            self.sentiment_analyzer = None

    def analyze_tweet(self, text: str) -> Dict:
        """
        Analyze tweet sentiment.

        Args:
            text: Tweet text

        Returns:
            Sentiment dictionary
        """
        if not self.sentiment_analyzer:
            return {"label": "NEUTRAL", "score": 0.0, "sentiment_score": 0.0}

        try:
            result = self.sentiment_analyzer(text[:512])
            score = result[0]["score"]
            label = result[0]["label"]

            return {
                "label": label,
                "score": score,
                "sentiment_score": score if label == "POSITIVE" else -score,
            }
        except Exception as e:
            logger.error(f"Tweet sentiment analysis error: {e}")
            return {"label": "NEUTRAL", "score": 0.0, "sentiment_score": 0.0}


class OnChainAnalyzer:
    """On-chain data analyzer."""

    def __init__(self, infura_key: Optional[str] = None) -> None:
        """
        Initialize on-chain analyzer.

        Args:
            infura_key: Infura API key (optional)
        """
        self.infura_key = infura_key
        self.w3 = None

        if infura_key:
            try:
                from web3 import Web3
                self.w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{infura_key}"))
                logger.info("Connected to Ethereum node")
            except ImportError:
                logger.warning("web3 not available. Install with: pip install web3")
            except Exception as e:
                logger.error(f"Failed to connect to Ethereum: {e}")

    def get_gas_price(self) -> Optional[Dict]:
        """Get current Ethereum gas price."""
        if not self.w3:
            return None

        try:
            gas_price = self.w3.eth.gas_price
            return {
                "timestamp": datetime.utcnow(),
                "gas_price_wei": gas_price,
                "gas_price_gwei": self.w3.from_wei(gas_price, "gwei"),
            }
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            return None

    def get_transaction_count(self) -> Optional[Dict]:
        """Get Ethereum transaction count."""
        if not self.w3:
            return None

        try:
            block = self.w3.eth.get_block("latest")
            return {
                "timestamp": datetime.utcnow(),
                "block_number": block.number,
                "transaction_count": len(block.transactions),
            }
        except Exception as e:
            logger.error(f"Error getting transaction count: {e}")
            return None

