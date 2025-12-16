"""Kafka consumer for real-time market data."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("kafka-python not available. Install with: pip install kafka-python")

logger = logging.getLogger(__name__)


class MarketDataConsumer:
    """Kafka consumer for market data streams."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topics: Optional[List[str]] = None,
        group_id: str = "neural-kraken-consumer",
    ) -> None:
        """
        Initialize Kafka consumer.

        Args:
            bootstrap_servers: Kafka broker addresses
            topics: List of topics to subscribe to
            group_id: Consumer group ID
        """
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is required. Install with: pip install kafka-python")

        self.bootstrap_servers = bootstrap_servers
        self.topics = topics or ["market.ticker", "market.trade", "market.orderbook"]
        self.group_id = group_id
        self.consumer: Optional[KafkaConsumer] = None
        self.callbacks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

    def start(self) -> None:
        """Start the consumer."""
        self.consumer = KafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id=self.group_id,
        )
        logger.info(f"Started Kafka consumer for topics: {self.topics}")

    def register_callback(self, topic: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for a topic.

        Args:
            topic: Topic name
            callback: Callback function
        """
        if topic not in self.callbacks:
            self.callbacks[topic] = []
        self.callbacks[topic].append(callback)
        logger.info(f"Registered callback for topic: {topic}")

    def process_messages(self) -> None:
        """Process messages from Kafka."""
        if not self.consumer:
            raise RuntimeError("Consumer not started. Call start() first.")

        logger.info("Processing Kafka messages...")
        for message in self.consumer:
            try:
                topic = message.topic
                data = message.value

                # Call registered callbacks
                if topic in self.callbacks:
                    for callback in self.callbacks[topic]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Callback error for topic {topic}: {e}")

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    def stop(self) -> None:
        """Stop the consumer."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer stopped")


class MarketDataProducer:
    """Kafka producer for publishing market data."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
    ) -> None:
        """
        Initialize Kafka producer.

        Args:
            bootstrap_servers: Kafka broker addresses
        """
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is required. Install with: pip install kafka-python")

        try:
            from kafka import KafkaProducer
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info(f"Started Kafka producer for {bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            raise

    def publish(self, topic: str, data: Dict[str, Any], key: Optional[str] = None) -> None:
        """
        Publish data to Kafka topic.

        Args:
            topic: Topic name
            data: Data to publish
            key: Optional message key
        """
        try:
            future = self.producer.send(topic, value=data, key=key.encode() if key else None)
            future.get(timeout=10)  # Wait for confirmation
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            raise

    def close(self) -> None:
        """Close the producer."""
        self.producer.close()
        logger.info("Kafka producer closed")

